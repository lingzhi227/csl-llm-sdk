"""Execute complete resident requests and retain candidate evidence, not acceptance.

The host uploads prompt IDs once per request, triggers dependent steps, and
reads IDs/timing. Neural intermediates and next-token IDs never enter H2D.
"""
import hashlib,json,math,time
from pathlib import Path
import numpy as np
from backend import runtime
from resident_loader import upload
from source_gate import verify

def main():
    root=Path.cwd();verify();acceptance=json.loads((root/'full-acceptance-v1.json').read_text())
    frozen=json.loads((root/'full-acceptance-frozen.json').read_text())
    assert hashlib.sha256((root/'full-acceptance-v1.json').read_bytes()).hexdigest()==frozen['sha256']
    assert acceptance['candidate_full_outputs_seen'] is False
    sram=json.loads((root/'sram.json').read_text());assert sram['passed'] and sram['application_pes']==870000
    workloads={w['name']:w for w in acceptance['workloads']};runs=[];first_story=None
    opened=time.monotonic_ns();model=Path('/srv/qwen38-singlewse-hardware/model')
    with runtime(True) as (runner,dtype,order):
        ready=time.monotonic_ns();ids,initialization=upload(runner,dtype,order,root,model)
        opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def put(name,data):
            value=np.array(data,np.uint32);runner.memcpy_h2d(ids[name],value,749,1159,1,1,value.size,data_type=dtype.MEMCPY_32BIT,**opts)
        def get(name,count,x=749,y=1159,width=1,height=1,bits=32):
            result=np.zeros(count*width*height,np.uint32)
            assert result.nbytes<16<<20
            runner.memcpy_d2h(result,ids[name],x,y,width,height,count,
                data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**opts)
            return result&65535 if bits==16 else result
        roles=json.loads((root/'role-plan-96.json').read_text());capture_xy=roles['validation']['coordinates']
        runner.launch('start',nonblock=False);request_calls=0
        for request_index,spec in enumerate(acceptance['run_order']):
            workload=workloads[spec['workload']];folder=root/f'request-{request_index:02d}';folder.mkdir()
            prompt=workload['input_ids'];limit=workload['max_new_tokens'];tokens=[];steps=[]
            assert len(prompt)+limit-1<=96
            begin=time.monotonic_ns()
            put('prompt',prompt);put('controls',[len(prompt),limit,workload['eos_id'],int(spec['capture'])])
            previous=begin
            for index in range(limit):
                launched=time.monotonic_ns();runner.launch('begin' if index==0 else 'step',nonblock=False)
                metadata=get('result',5);finished=time.monotonic_ns();request_calls+=1
                token,count,position,done,best_bits=map(int,metadata)
                assert 0<=token<248320 and count==index+1 and position==len(prompt)+index and done in (0,1)
                best=np.array(best_bits,np.uint32).view(np.float32).item();assert math.isfinite(best)
                tokens.append(token)
                steps.append(dict(index=index,token=token,position=position,done=bool(done),maximum_logit=best,
                    dependent_step_seconds=(finished-launched)/1e9,observed_token_interval_seconds=(finished-previous)/1e9,
                    elapsed_seconds=(finished-begin)/1e9));previous=finished
                (folder/'step-progress.json').write_text(json.dumps(dict(generated_ids=tokens,steps=steps))+'\n')
                if done:break
            assert np.array_equal(get('generated',len(tokens)),np.array(tokens,np.uint32))
            eos=tokens[-1]==workload['eos_id']
            positions=len(prompt)+len(tokens)-1
            row=dict(request=request_index,workload=workload['name'],capture_enabled=spec['capture'],physical=True,
                prompt_ids=prompt,generated_ids=tokens,eos_reached=eos,processed_positions=positions,all_layers=64,vocabulary=248320,
                ttft_seconds=steps[0]['elapsed_seconds'],end_to_end_seconds=steps[-1]['elapsed_seconds'],steps=steps,
                no_hidden_or_next_token_upload=True,numerical_acceptance_pending=True)
            (folder/'generation.json').write_text(json.dumps(row,indent=2)+'\n');runs.append(row)
            # Check every endpoint has completed the same number of SDK calls
            # and seen every ordered global frame. Read only after timing ends.
            low=None;high=None
            with (folder/'endpoint-trace.u32').open('xb') as out:
                for y in range(0,1160,4):
                    trace=get('trace',6,0,y,750,4).reshape(4,750,6)
                    assert np.all(trace[:,:,4]==request_calls)
                    current=trace[:,:,:].reshape(-1,6)
                    if y==1156:
                        assert np.array_equal(trace[3,749,[0,5]],np.zeros(2,np.uint32));current=current[:-1]
                    assert np.array_equal(current[:,0],current[:,5])
                    lo=int(current[:,5].min());hi=int(current[:,5].max())
                    low=lo if low is None else min(low,lo);high=hi if high is None else max(high,hi)
                    out.write(trace.astype('<u4').tobytes())
            assert low==high and low>0
            (folder/'trace-check.json').write_text(json.dumps(dict(all870000_endpoints_checked=True,request_calls=request_calls,endpoint_global_sequence=low))+'\n')
            if spec['capture']:
                for name,base,elements in [('layers',0,positions*64*5120),('logits',1794,positions*248320),('final_norm',3154,positions*5120)]:
                    digest=hashlib.sha256();written=0
                    with (folder/(name+'.bf16')).open('xb') as out:
                        for index in range((elements+17535)//17536):
                            count=min(17536,elements-written);x,y=capture_xy[base+index]
                            bits=get('data',count,x,y,bits=16).astype('<u2');raw=bits.tobytes();out.write(raw);digest.update(raw);written+=count
                    assert written==elements
                    (folder/(name+'-capture.json')).write_text(json.dumps(dict(elements=elements,bytes=elements*2,sha256=digest.hexdigest(),candidate_outputs_only=True))+'\n')
            # Retain candidate IDs and captures before rejecting sequence checks.
            assert eos,'EOS absent at fixed request limit'
            if workload['name']=='sunny_morning':
                assert len(tokens)-1>=acceptance['sequence']['sunny_morning_min_non_eos_tokens']
                if first_story is None:first_story=tokens.copy()
                else:assert tokens==first_story,'Request reset or capture mode changed dependent output'
            print(json.dumps(dict(request=request_index,workload=workload['name'],capture=spec['capture'],tokens=tokens,ttft_seconds=row['ttft_seconds'],end_to_end_seconds=row['end_to_end_seconds'],numerical_acceptance_pending=True)),flush=True)
            (root/'candidate-progress.json').write_text(json.dumps(dict(completed_requests=len(runs),full_model_accepted=False))+'\n')
    verify()
    result=dict(candidate_execution_complete=True,full_model_accepted=False,physical=True,normal_stop=True,
        requires='Independent frozen-criterion sequential-prefix CPU comparison, decoded sentence checks and correlated hardware release',
        runtime_initialization_seconds=(ready-opened)/1e9,original_weight_initialization=initialization,runs=runs,
        acceptance_sha256=frozen['sha256'])
    (root/'CANDIDATE.json').write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
