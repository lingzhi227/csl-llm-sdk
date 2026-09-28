"""Freeze all16 original frontend groups and full native multicast copies.

Projected BF16 values come from the accepted complete-matrix independent
reference. Frozen FP64 recurrent results are injected only to qualify post-core
gated normalization; this fixture does not execute or admit a GDN device graph.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from weights import OriginalWeights
from reference.frontend_oracle import bits,expand,frontend_step,gated_interval


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def main():
    started=time.monotonic();config=json.loads(Path('reference-input.json').read_text());root=Path(config['root'])
    if sha(root/'fixture.npz')!=config['fixture_sha256']:raise ValueError('Original projection oracle changed')
    original=dict(np.load(root/'fixture.npz',allow_pickle=False))
    metadata=json.loads(Path('tensors.json').read_text());pins={'layers-0.safetensors':'07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'}
    reader=OriginalWeights(Path('/srv/model-storage/qwen38-singlewse/model'),metadata['tensors'],pins)
    prefix='model.language_model.layers.0.linear_attn.'
    conv=reader.small_tensor(prefix+'conv1d.weight').reshape(10240,4)
    gain=reader.small_tensor(prefix+'norm.weight').reshape(128)
    a_log=reader.small_tensor(prefix+'A_log').reshape(48);bias=reader.small_tensor(prefix+'dt_bias').reshape(48)
    network=json.loads(Path('frontend-network.json').read_text());setups=dict((tuple(pe),s) for pe,s in json.loads(Path('mixer-setups.json').read_text()))
    matrices=[original[f'bf16_{case}_{mi}'] for case in range(4) for mi in range(4)]
    if any(x.dtype!=np.uint16 for x in matrices):raise ValueError('Original projected BF16 encoding')
    sequence=[0,2,1,0,2,0];arrays={};weights=[];params=[];channels=[]
    for group in range(16):
        channel=np.r_[128*group+np.arange(128),2048+128*group+np.arange(128),4096+384*group+np.arange(384)]
        channels.append(channel);weights.append(conv[channel]);params.append(np.r_[a_log[3*group:3*group+3],bias[3*group:3*group+3]])
    arrays['weights']=np.array(weights,np.uint16);arrays['parameters']=np.array(params,np.uint16);arrays['gain']=np.repeat(gain[None],16,axis=0)
    history=np.zeros((16,640,3),np.uint16);state=np.zeros((16,3,128,128),np.float64)
    records=[];max_packet_ratio=0.;max_gated_ratio=0.
    for position,case in enumerate(sequence):
        projected=[original[f'bf16_{case}_{mi}'] for mi in range(4)];token=position+1
        packet=np.zeros((16,3,386),np.float64);bound=np.zeros_like(packet);core=np.zeros((16,384),np.uint16)
        lower=np.zeros((16,384),np.float64);upper=np.zeros_like(lower)
        for group in range(16):
            raw=np.r_[projected[0][channels[group]],projected[1][384*group:384*group+384],projected[2][3*group:3*group+3],projected[3][3*group:3*group+3]]
            packet[group],bound[group],_=frontend_step(raw,arrays['weights'][group],arrays['parameters'][group],gain,history[group])
            for head in range(3):
                p=packet[group,head];q,k,v=p[:128],p[128:256],p[256:384]
                state[group,head]*=p[384]
                delta=(v-k@state[group,head])*p[385]
                state[group,head]+=k[:,None]*delta
                values=bits(q@state[group,head]);at=slice(128*head,128*(head+1));core[group,at]=values
                lower[group,at],upper[group,at]=gated_interval(values,raw[640+128*head:640+128*(head+1)],gain)
            ratio=float(np.linalg.norm(bound[group])/max(np.linalg.norm(packet[group]),1e-30));max_packet_ratio=max(max_packet_ratio,ratio)
        streams=[[] for _ in range(16)]
        # Reverse projection order alternately to exercise gates/Z arriving
        # before Q/K/V, independently of callback ready notifications.
        matrix_order=[3,1,2,0] if position%2 else [0,2,1,3]
        for mi in matrix_order:
            for producer in reversed(network['producers']):
                s=setups[tuple(producer['pe'])];_,count,first,stride,key,_,_,rows=s[8*mi:8*mi+8]
                if key!=0:raise ValueError('Original projection source key')
                for i in range(count):
                    row=first+i*stride;value=int(projected[mi][row])
                    if rows==2:value|=int(projected[mi][row+1])<<16
                    frame=[token,row,rows,value,mi]
                    for group in producer['groups']:streams[group].append(frame)
        for group,frames in enumerate(streams):
            if len(frames)!=network['consumer_projected_packets'][group]:raise ValueError('Native drain count')
            arrays[f'projected_{position}_{group}']=np.array(frames,np.uint32)
        returns=np.zeros((16,192,5),np.uint32)
        for group in range(16):
            for i in range(192):returns[group,i]=[token,384*group+2*i,2,int(core[group,2*i])|(int(core[group,2*i+1])<<16),5]
        for name,value in dict(packet=packet,bound=bound,history=history.copy(),core=core,lower=lower,upper=upper,returns=returns).items():arrays[f'{name}_{position}']=value
        for group in range(16):
            ratio=float(np.linalg.norm(upper[group]-lower[group])/max(np.linalg.norm((upper[group]+lower[group])/2),1e-30));max_gated_ratio=max(max_gated_ratio,ratio)
        if not all(np.isfinite(x).all() for x in (packet,bound,lower,upper)):raise ValueError('Nonfinite independent reference')
        records.append(dict(position=position,source_case=case,token=token,projection_order=matrix_order))
    if max_packet_ratio>=.02 or max_gated_ratio>=.03:raise ValueError('A priori frontend interval too broad')
    with Path('fixture.npz').open('xb') as stream:np.savez(stream,**arrays)
    meta=dict(passed=True,physical=False,groups=16,heads=48,positions=len(sequence),reset_replay_positions=2,
        revision=metadata['revision'],original_shards=pins,projection_reference_sha256=config['fixture_sha256'],
        sequence=records,consumer_projected_packets=network['consumer_projected_packets'],
        maximum_packet_bound_norm_ratio=max_packet_ratio,maximum_gated_interval_norm_ratio=max_gated_ratio,
        fixture_sha256=sha(Path('fixture.npz')),seconds=time.monotonic()-started,
        criterion='All original frontend groups: predeclared FP64/BF16 intervals, exact three-sample histories, parameter retention, native packet/token drain and bit-exact reset replay.',
        scope='Actual original projection-reference inputs and original conv/gate/norm weights; standalone CSL arrival frontend. Frozen recurrent results are injected to isolate gated normalization. No actual GDN, complete layer, conversation or speed admission.')
    Path('fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)


if __name__=='__main__':main()
