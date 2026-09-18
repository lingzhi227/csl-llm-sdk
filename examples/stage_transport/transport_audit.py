"""Offline exact transport audit after device/runtime release, synthetic only."""
import hashlib
import json
from pathlib import Path
import numpy as np


def require(value,message):
    if not value:raise ValueError(message)


def supplied(epoch,phase,index,length):
    if index>=length:return np.float32(0)
    return np.float32((((epoch-1)*7+phase*3+index%17)%23-11)/16)


def rounded(values):
    raw=np.asarray(values,np.float32).view(np.uint32)
    return ((raw+np.uint32(0x7fff)+((raw>>16)&1))>>16).astype(np.uint16)


def audit(root):
    root=Path(root);plan=json.loads((root/'transport-plan.json').read_bytes())
    capture=json.loads((root/'capture.json').read_bytes())
    require(capture['status']=='captured' and capture['normal_stop'] and capture['complete_epochs']==[1,2],'Released complete capture')
    for name,spec in capture['files'].items():
        p=root/'evidence'/name;require(p.stat().st_size==spec['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==spec['sha256'],'Captured archive identity')
    counts=dict(matrix_inputs=0,local_outputs=0,line_broadcast_outputs=0,BF16_root_consumer_values=0)
    for epoch in (1,2):
        with np.load(root/'evidence'/('epoch-'+str(epoch)+'.npz'),allow_pickle=False) as actual:
            require(np.array_equal(actual['origin'],[3*epoch,4*epoch,264*epoch,69888*epoch,epoch-1]),'Observed early READY/full frame/chunk/retire counts')
            for s in plan['stripes']:
                tag=str(s['group']);c=s['columns'];partial=np.zeros((c,128),np.float32)
                final_input=np.array([[supplied(epoch,3,96*k+i,17408) for i in range(96)] for k in range(c)],np.float32)
                require(np.array_equal(actual[tag+'_input'].reshape(c,96),final_input),'Every lane receives its own ordinal in final fixed period')
                for k in range(c):
                    valid=min(96,s['input_words']-96*k)
                    for row in range(s['rows']):partial[k,row]=supplied(epoch,s['phase'],96*k+row%valid,s['input_words'])
                require(np.array_equal(actual[tag+'_partial'].reshape(c,128),partial),'Active phase and exact local input/coefficient')
                total=partial[-1].copy()
                for k in range(c-2,-1,-1):total=np.add(total,partial[k],dtype=np.float32)
                require(np.array_equal(actual[tag+'_result'].reshape(c,128),np.broadcast_to(total,(c,128))),'Stripe-local right-to-left chain and all-recipient broadcast')
                bf16=rounded(total)
                for side in ('root','consumer'):require(np.array_equal((actual[tag+'_'+side]&65535).astype(np.uint16),bf16),'BF16 final cast, packet group/order and consumer retention')
                require(np.array_equal(actual[tag+'_counters'].reshape(c,4),np.broadcast_to([epoch,epoch,epoch,4*epoch],(c,4))),'All-lane phase/own-line/token completion')
                counts['matrix_inputs']+=c*96;counts['local_outputs']+=c*128
                counts['line_broadcast_outputs']+=c*128;counts['BF16_root_consumer_values']+=256
    result=dict(status='passed',scope='connected_dense_stage_transport_synthetic',original_model=False,
        exact_comparisons=counts,early_READY_events=6,repeated_tokens=2,
        full_layer_inference=False,physical_execution_claim_requires_runtime_evidence=True)
    (root/'audit.json').write_text(json.dumps(result,indent=2)+'\n');return result


if __name__=='__main__':print(json.dumps(audit(Path(__file__).resolve().parent)))
