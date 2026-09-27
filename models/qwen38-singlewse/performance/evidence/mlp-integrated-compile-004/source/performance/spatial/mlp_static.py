"""Static fabric for the complete original MLP geometry.

Keep gate/up trees live while down uses separate nearest-neighbor colors. No
native route word is rewritten. RAMP input queues and memory still have explicit
phase/credit dependencies in the worker and activation programs.
"""
from itertools import chain


class MlpStatic:
    def __init__(self, regions):
        self.regions=regions
        self.width,self.height=regions.width,regions.height

    def flows(self):
        """Yield complete static networks, with each network's PE list once.

        The encoded route byte is RX0..4 and TX bitmask RAMP/N/E/S/W. Words count
        the full MLP boundary, including both post-attention and successor RMS.
        Epoch control, external upstream transport and fullmodel reset are absent.
        """
        r=self.regions
        for s in chain(r.distribution(),r.down_edges(),r.native_edges()):
            a=tuple(s['source']);b=tuple(s.get('destination',s.get('end')))
            consume={b}
            if s['kind']=='header-fork':consume={tuple(v) for v in s['consumers']}
            if s['kind']=='vertical-multicast':
                consume={r.worker(g,s['worker_branch'],p,s['k']) for g in s['groups'] for p in range(64)}
            yield dict(id=s['id'],family=s['kind'],words=s['words'],color=s['color'],
                       source=a,consumers=consume,routes=self.line(a,b,consume))
        for g in range(136):
            x,y=r.origin(g)
            # Dedicated down broadcast: root -> north -> west, with south fanout.
            routes=[(x+41,y+1,0,2),(x+41,y,3,16)]
            for k in range(40):
                xx=x+40-k
                routes.append((xx,y,2,8|(16 if k<39 else 0)))
                for p in range(64):routes.append((xx,y+p+1,1,1|(8 if p<63 else 0)))
            yield dict(id=f'down-input:{g}',family='down-local-broadcast',words=65,color=18,
                       source=r.activation(g,0),consumers={r.worker(g,'gate',p,k) for p in range(64) for k in range(40)},routes=routes)
            a=r.activation(g,0);b=r.activation(g,63)
            yield dict(id=f'act-scale:{g}',family='activation-scale',words=1,color=6,
                       source=a,consumers={r.activation(g,p) for p in range(1,64)},
                       routes=self.line(a,b,{r.activation(g,p) for p in range(1,64)}))
            for p in range(1,64):
                a=r.activation(g,p);b=r.activation(g,p-1)
                yield dict(id=f'act-join:{g}:{p}',family='activation-max-and-encoded-gather',words=1+64-p,color=4+p%2,
                           source=a,consumers={b},routes=self.line(a,b,{b}))
            for k in range(40):
                for p in range(64):
                    a=r.worker(g,'gate',p,k);b=r.worker(g,'gate',p-1,k) if p else r.collector(g,k)
                    yield dict(id=f'down-local:{g}:{k}:{p}',family='down-local-gather',words=2*(64-p),color=2 if p%2==0 else 15,
                               source=a,consumers={b},routes=self.line(a,b,{b}))
        for k in range(1,40):
            a=(0,k);b=(0,k-1)
            yield dict(id=f'norm-squares:{k}',family='two-rms-reductions',words=2,color=4+k%2,
                       source=a,consumers={b},routes=self.line(a,b,{b}))
        a=(0,0);b=(0,39);consume={(0,k) for k in range(1,40)}
        yield dict(id='norm-inverse',family='two-rms-broadcasts',words=2,color=6,
                   source=a,consumers=consume,routes=self.line(a,b,consume))

    @staticmethod
    def line(a,b,consumers):
        assert (a[0]==b[0]) != (a[1]==b[1])
        dx=(b[0]>a[0])-(b[0]<a[0]);dy=(b[1]>a[1])-(b[1]<a[1])
        direction={(0,-1):1,(1,0):2,(0,1):3,(-1,0):4}[dx,dy]
        opposite={1:3,2:4,3:1,4:2}[direction]
        length=abs(b[0]-a[0])+abs(b[1]-a[1])
        return [(a[0]+i*dx,a[1]+i*dy,0 if i==0 else opposite,
                 (1<<direction if i<length else 0)|(1 if (a[0]+i*dx,a[1]+i*dy) in consumers else 0))
                for i in range(length+1)]

    def document(self):
        return dict(schema='wse-static-mlp-fabric-v1',application=[self.width,self.height],
                    original_dimensions=dict(hidden=5120,intermediate=17408,k_groups=40,mlp_groups=136),
                    strategy='Concurrent static gate/up trees and down local gather/broadcast; no route word rewrites or forwarding-drain fence is needed for this fabric.',
                    local_down_colors=[2,15],down_broadcast_color=18,activation_join_colors=[4,5],activation_scale_color=6,
                    worker_transition=['consume complete65-word initial packet','complete own native dot and exact children',
                                       'native outgoing send callback','rebind initial input queue from3 to18 and child queue to static down-local color',
                                       'consume complete65-word down packet','down native dot','send own pair then credit-forward child pairs'],
                    activation_transition=['both packed gate/up operands','BF16 SiLU/multiply and local absmax',
                                           'group absmax arrival and outgoing callback','scale broadcast','encode packed pair',
                                           'ordered gather with callback credits','root sends65-word down packet'],
                    fabric_routes_static=True,device_epoch_controller_implemented=False,full_model_reset_qualified=False,
                    full_subgraph_executable=False,compiled_sram_qualified=False,physical=False,full_model=False)
