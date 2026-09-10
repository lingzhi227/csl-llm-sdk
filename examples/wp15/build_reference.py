"""Frozen original-projection CPU sequence with persistent uncertain K/V sources."""
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
import torch

ROOT=Path.cwd();sys.path.insert(0,str(ROOT/'core'))
from qwen38.projected_attention_numerics import POLICY,SourceCache
from qwen38.trained_qk_rope_numerics import Interval,report,spans,input_interval
from qwen38.rms_numerics import bits
from pinned_reference import PinnedReference

torch.set_num_threads(1);torch.set_num_interop_threads(1)
plan=json.loads((ROOT/'reference-plan.json').read_text())
assert plan['policy']==POLICY
for name,h in plan['frozen_inputs'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h


def decode(path,shape,offset=0):
    words=np.memmap(path,dtype='<u2',mode='r',offset=offset,shape=shape)
    return (np.asarray(words,dtype=np.uint32)<<16).view(np.float32)


def floats(value):
    return value.detach().float().numpy().astype(np.float64).reshape(-1)


weights_dir=Path(plan['weights_directory'])
weights={name:decode(weights_dir/file,(rows,5120),offset) for name,file,rows,offset in
         [('q_proj','q-with-norm.bf16',512,512),('k_proj','k-with-norm.bf16',256,512),('v_proj','v-row-major.bf16',256,0)]}
norm_weights=[decode(weights_dir/file,(256,)) for file in ('q-with-norm.bf16','k-with-norm.bf16')]
hidden=decode(ROOT/'hidden.bf16',(4,5120))
projection=np.load(ROOT/'projection-observations.npz',allow_pickle=False)
official=PinnedReference((ROOT/'modeling_qwen3_5.py').read_bytes(),weights,norm_weights)
assert [bits(v) for v in official.frequencies.tolist()]==plan['inverse_frequency_bits']

source_arrays={};source_steps=[];cache=SourceCache()


def collect_intervals(prefix,obj):
    if isinstance(obj,list) and obj and all(isinstance(v,Interval) for v in obj):
        source_arrays[prefix]=np.array([(v.lo,v.hi) for v in obj],dtype=np.float64)
    elif isinstance(obj,dict):
        for name,value in obj.items():collect_intervals(prefix+'__'+name,value)
    elif isinstance(obj,list):
        for i,value in enumerate(obj):collect_intervals(prefix+'__'+str(i),value)


# Every source enclosure is completed and serialized before CPU projection
# observations. Original intervals persist in the source cache across tokens.
for i,case in enumerate(POLICY['cases']):
    centers=projection[case+'__nominal'].tolist();radii=projection[case+'__radius'].tolist()
    step=cache.step(centers,radii,np.concatenate(norm_weights).tolist(),official.frequencies.tolist(),
                    POLICY['request_generations'][i],POLICY['positions'][i])
    source_steps.append(step);collect_intervals(case,step)
    source_arrays[case+'__projection']=np.array([(x.lo,x.hi) for x in [input_interval(c,e) for c,e in zip(centers,radii)]])
np.savez(ROOT/'source-intervals.npz',**source_arrays)
source_archive_sha256=hashlib.sha256((ROOT/'source-intervals.npz').read_bytes()).hexdigest()
print('SOURCE_INTERVALS_FROZEN',source_archive_sha256,flush=True)

arrays={'norm_weights':np.concatenate(norm_weights),'inverse_frequencies':official.frequencies.numpy()}
rows=[];history=[];generation=0
for i,case in enumerate(POLICY['cases']):
    gen,pos=POLICY['request_generations'][i],POLICY['positions'][i]
    if gen!=generation:generation=gen;history=[]
    source=source_steps[i];pre=source['preprocessing'];ar=source['attention']
    item=official.project(hidden[i],pos)
    projected=torch.cat([item['captured'][n].flatten() for n in ('q_proj','k_proj','v_proj')])
    projection_values=floats(projected)
    assert np.array_equal(projection_values,projection[case+'__official'])
    assert np.array_equal(floats(item['gate']),projection_values[256:512])
    assert np.array_equal(floats(item['v']),projection_values[768:])
    bounds=[Interval(*pair) for pair in source_arrays[case+'__projection']]
    checks={'projection':report(projection_values.tolist(),bounds,projection[case+'__nominal'].tolist())}
    for h,name in enumerate(('q','k')):
        checks[name+'_normalized']=report(floats(item['captured'][name+'_norm']).tolist(),pre['heads'][h]['rms']['output'],pre['heads'][h]['rms']['nominal'])
        checks[name+'_rotated']=report(floats(item[name]).tolist(),pre['heads'][h]['output'],pre['heads'][h]['nominal'])
        for p,label in enumerate(official.rotary_labels[name+'_embed']):
            interval=pre['heads'][h]['product'+str(p)] if p<2 else pre['heads'][h]['output'][:64]
            checks[name+'_rotary_boundary'+str(p)]=report(floats(item['trace']['rotary:'+label]).tolist(),interval)
        assert torch.equal(item[name].flatten()[64:],item['captured'][name+'_norm'].flatten()[64:])
    checks['sine']=report(floats(item['sine']).tolist(),pre['sine']*2,pre['sine_nominal']*2)
    checks['cosine']=report(floats(item['cosine']).tolist(),pre['cosine']*2,pre['cosine_nominal']*2)
    rms_dtypes={k:str(v.dtype) for k,v in item['trace'].items() if k.startswith(('q_norm:','k_norm:'))}
    rotary_dtypes={k:str(v.dtype) for k,v in item['trace'].items() if k.startswith('rotary:')}
    assert rms_dtypes and all(v=='torch.float32' for v in rms_dtypes.values())
    assert len(rotary_dtypes)==6 and all(v=='torch.bfloat16' for v in rotary_dtypes.values())
    assert all(v.dtype==torch.bfloat16 for v in item['captured'].values())
    history.append(item);keys=[h['k'] for h in history];values=[h['v'] for h in history]
    result=official.attend(item['q'],keys,values,item['gate'])
    final=floats(result['output']);attention_values=floats(result['attention'])
    assert len(history)==source['valid_tokens']==pos+1
    stage_dtypes={n:str(v.dtype) for n,v in result['stages'].items()}
    assert stage_dtypes=={'dot_bf16':'torch.bfloat16','scaled_bf16':'torch.bfloat16','probability':'torch.float32',
                          'attention_bf16':'torch.bfloat16','gate_bf16':'torch.bfloat16','output':'torch.bfloat16'}
    for name,value in result['stages'].items():
        values_list=floats(value).tolist()
        checks['attention_'+name]=report(values_list,ar[name],ar['nominal'].get(name))
    checks['attention_probability_bf16']=report(floats(result['probability']).tolist(),ar['probability_bf16'],ar['nominal']['probability_bf16'])

    case_arrays=dict(projection=projection_values,output=final,attention=attention_values,
                     probability=floats(result['probability']),source_nominal_output=np.array(ar['nominal']['output']),
                     cache_keys=floats(torch.cat(keys,dim=2)).reshape(len(history),256),
                     cache_values=floats(torch.cat(values,dim=2)).reshape(len(history),256))
    for name in ('q','k','v','gate','sine','cosine'):case_arrays[name]=floats(item[name])
    for name,value in item['captured'].items():case_arrays['captured_'+name]=floats(value)
    for name,value in item['trace'].items():case_arrays['prefix_trace_'+name]=floats(value)
    for name,value in result['stages'].items():case_arrays['attention_stage_'+name]=floats(value)

    countervalues={
        'zero':np.zeros(256), 'cyclic_dimension_permutation':np.roll(final,-1), 'no_sigmoid_gate':attention_values,
        'current_KV_only':floats(official.attend(item['q'],keys[-1:],values[-1:],item['gate'])['output']),
        'ignore_QK_uniform_scores':floats(official.attend(torch.zeros_like(item['q']),keys,values,item['gate'])['output']),
        'negate_current_query':floats(official.attend(-item['q'],keys,values,item['gate'])['output']),
        'wrong_V_routed_from_rawgate':floats(official.attend(item['q'],keys,[h['gate'].reshape(1,1,1,256) for h in history],item['gate'])['output']),
        'reverse_aligned_cache_pairs':floats(official.attend(item['q'],keys[::-1],values[::-1],item['gate'])['output']),
    }
    countervalues['reset_cache_each_token']=countervalues['current_KV_only'].copy()
    if len(history)==2:
        first,second=history
        _,initial_key,_,_=official.rotate(second['captured']['q_norm'],second['captured']['k_norm'],0)
        reversed_query,reversed_key,_,_=official.rotate(first['captured']['q_norm'],first['captured']['k_norm'],1)
        countervalues['reverse_two_hidden_inputs_and_positions']=floats(official.attend(reversed_query,[initial_key,reversed_key],[second['v'],first['v']],first['gate'])['output'])
    counters={name:report(value.tolist(),ar['output']) for name,value in countervalues.items()}
    case_arrays.update({'counter_'+name:value for name,value in countervalues.items()})
    arrays.update({case+'__'+name:value for name,value in case_arrays.items()})
    widths=np.array([v.hi-v.lo for v in ar['output']])
    norm=float(np.linalg.norm(final));relative_width=float(np.linalg.norm(widths)/norm) if norm else None
    passed=all(check['passed'] for check in checks.values())
    rows.append(dict(case=case,generation=gen,position=pos,valid_cache_tokens=len(history),passed=passed,checks=checks,
                     rms_dtypes=rms_dtypes,rotary_dtypes=rotary_dtypes,eager_dtypes=stage_dtypes,
                     source_shift_magnitude_upper=ar['source_shift_magnitude_upper'],output_spans=ar['output_spans'],
                     full_width_L2_over_output_L2=relative_width,counterexamples=counters,
                     counterexample_notes='Current-KV-only and reset-per-token are the same counterexample. Reversing aligned cached K/V pairs is mathematically permutation-invariant; physical cache order needs separate metadata/bit checks. Reversing the two original hidden inputs is a distinct wrong-request-order counterexample.'))
    print('CASE',case,'passed',passed,'cache',len(history),'max_output_width',float(widths.max()),flush=True)

np.savez(ROOT/'observations.npz',**arrays)
result=dict(passed=all(r['passed'] for r in rows),scope='Four original-hidden CPU references:three consecutive same-request tokens then new-request zero; no SDK execution',
            policy=POLICY,cases=rows,source_identity=official.identity,
            source_sha256=hashlib.sha256((ROOT/'modeling_qwen3_5.py').read_bytes()).hexdigest(),
            projection_oracle_sha256=plan['frozen_inputs']['projection-observations.npz'],
            source_intervals_sha256=source_archive_sha256,observations_sha256=hashlib.sha256((ROOT/'observations.npz').read_bytes()).hexdigest(),
            output_bytes={n:(ROOT/n).stat().st_size for n in ('source-intervals.npz','observations.npz')},
            inverse_frequency_bits=[bits(v) for v in official.frequencies.tolist()],
            backend=dict(torch=torch.__version__,numpy=np.__version__,device='cpu',threads=torch.get_num_threads(),configuration=torch.__config__.show()))
(ROOT/'result.json').write_text(json.dumps(result,indent=2)+'\n')
assert result['passed'],'Original-hidden source enclosure failure; retain artifacts before any changed candidate'
print('PASS four original-hidden persistent attention CPU references',flush=True)
