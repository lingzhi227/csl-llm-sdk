"""Independent single-RX, connected multicast and full static-color admission."""
from array import array
from collections import Counter


def audit(fabric, flows=None):
    w,h=fabric.width,fabric.height
    assert (w,h)==(747,1080)
    occupied=bytearray(w*h*18);loads=array('I',[0])*(w*h*4)
    counts=Counter();hops=Counter();consumers=Counter();entries=0
    move={1:(0,-1),2:(1,0),3:(0,1),4:(-1,0)};opposite={1:3,2:4,3:1,4:2}
    for f in (fabric.flows() if flows is None else flows):
        color=f['color'];words=f['words'];family=f['family'];source=tuple(f['source'])
        assert 2<=color<=19 and 0<words<=128
        network={}
        for x,y,rx,tx in f['routes']:
            assert 0<=x<w and 0<=y<h and 0<=rx<5 and 0<tx<32
            assert (x,y) not in network
            key=(y*w+x)*18+color-2
            assert occupied[key]==0,('static route collision',f['id'],x,y,color)
            occupied[key]=1;network[x,y]=(rx,tx);entries+=1
        assert source in network and network[source][0]==0
        seen=set();todo=[source];actual=set()
        while todo:
            x,y=todo.pop();assert (x,y) not in seen,('cycle or merge',f['id'],x,y)
            seen.add((x,y));rx,tx=network[x,y]
            if tx&1:actual.add((x,y))
            for d,(dx,dy) in move.items():
                if not tx&(1<<d):continue
                next_xy=x+dx,y+dy
                assert next_xy in network and network[next_xy][0]==opposite[d],('disconnected link',f['id'],x,y,d)
                todo.append(next_xy);loads[(y*w+x)*4+d-1]+=words;hops[family]+=words
        assert seen==set(network) and actual==set(f['consumers'])
        assert sum(rx==0 for rx,tx in network.values())==1
        counts[family]+=1;consumers[family]+=len(actual)
    expected={'header-fork':40,'vertical-multicast':720,'native-tree':136*64*2*39,
              'packed-activation-operand':136*64*2,'vertical-reduction':(136-9)*40,
              'transpose-to-header':9*40,'horizontal-reduction':8*40,'residual-consumer':40,
              'down-local-broadcast':136,'activation-scale':136,
              'activation-max-and-encoded-gather':136*63,'down-local-gather':136*40*64,
              'two-rms-reductions':39,'two-rms-broadcasts':1}
    assert counts==expected,(counts,expected)
    assert consumers['down-local-broadcast']==348160 and consumers['vertical-multicast']==696320
    assert consumers['packed-activation-operand']==17408
    return dict(passed=True,application=[w,h],static_flows=sum(counts.values()),route_entries=entries,
                flows_by_family=dict(counts),consumer_deliveries_by_family=dict(consumers),
                word_hops_by_family=dict(hops),directed_links=sum(v>0 for v in loads),
                maximum_link_words=max(loads),static_single_rx_and_no_color_alias=True,
                native_route_rewrites=0,all_listed_streams_connected=True,
                scope='Static complete-dimensional MLP data paths plus two RMS phases. Does not prove PE queue phases, asynchronous liveness, external predecessor transport, epoch controller, termination, SRAM or execution.',
                runtime_liveness_qualified=False,compiled_sram_qualified=False,full_subgraph_executable=False,physical=False,full_model=False)
