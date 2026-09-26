"""Offline complete sequential-prefix oracle for frozen physical candidates."""
import argparse,ctypes,gc,hashlib,json,math,re,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from tokenizers import Tokenizer
from pinned import load,original_cache
from checkpoint import Checkpoint
from source_gate import verify

ROOT=Path.cwd();MODEL=Path('/srv/model-storage/qwen38-singlewse/model')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()

def expand(bits):return (np.asarray(bits,dtype=np.uint32)<<16).view(np.float32).astype(np.float64)

def metrics(candidate,reference,criteria):
    left=expand(candidate).reshape(-1);right=expand(reference).reshape(-1)
    assert left.shape==right.shape and np.isfinite(left).all() and np.isfinite(right).all()
    delta=left-right;norm=np.linalg.norm(right);other=np.linalg.norm(left);den=max(float(norm),1e-8)
    cosine=1.0 if norm==0 and other==0 else float(left@right)/max(float(norm*other),1e-8)
    rms=float(np.sqrt(np.mean(right*right)));error=float(np.max(np.abs(delta)))
    result=dict(relative_l2=float(np.linalg.norm(delta))/den,cosine=cosine,max_abs_error=error,
        rms_abs_error=float(np.sqrt(np.mean(delta*delta))),max_abs_error_over_reference_rms=error/max(rms,1e-8))
    passed=result['relative_l2']<=criteria['relative_l2_max'] and result['cosine']>=criteria['cosine_min']
    for key in ['rms_abs_error','max_abs_error','max_abs_error_over_reference_rms']:
        if key+'_max' in criteria:passed=passed and result[key]<=criteria[key+'_max']
    result['passed']=bool(passed);return result

def selection(candidate,reference,selected,criteria):
    observed=expand(candidate);values=expand(reference)
    assert observed.shape==values.shape and observed.ndim==1
    assert np.isfinite(observed).all() and np.isfinite(values).all()
    assert selected==int(np.argmax(observed)), 'Device lowest-ID argmax mismatch'
    best=int(np.argmax(values));value=float(values[best]);raw=int(reference[best]);adjacent=raw+1 if value>=0 else raw-1
    ulp=abs(float(expand(np.array([adjacent],np.uint16))[0])-value)
    band=min(criteria['largest_allowed_reference_gap'],2*ulp)
    gap=value-float(values[selected]);assert gap<=band, 'Reference winner band exceeded'
    return dict(selected=selected,reference_best=best,reference_gap=gap,allowed_bf16_band=band,exact_match=selected==best)

def main():
    verify()
    p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--await-candidate',action='store_true');p.add_argument('--collect-all',action='store_true');a=p.parse_args()
    assert torch.__version__=='2.6.0+cpu' and np.__version__=='2.2.6'
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    criteria=json.loads((ROOT/'full-acceptance-v1.json').read_text());frozen=json.loads((ROOT/'full-acceptance-frozen.json').read_text())
    assert sha(ROOT/'full-acceptance-v1.json')==frozen['sha256'] and not criteria['candidate_full_outputs_seen']
    workload=next(x for x in criteria['workloads'] if x['name']=='sunny_morning')
    ids=workload['input_ids'].copy();positions=None;candidate={};generation=None;text=None
    if not a.await_candidate:assert a.candidate.is_dir()
    pins=json.loads((MODEL/'COMPLETE.json').read_text())
    for name in ['config.json','tokenizer.json','generation_config.json']:
        record=next(x for x in pins['files'] if x['file']==name)
        raw=(MODEL/name).read_bytes()
        digest=hashlib.sha256(raw).hexdigest() if 'sha256' in record else hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
        assert digest==record.get('sha256',record.get('git_blob_sha1'))
    tokenizer=Tokenizer.from_file(str(MODEL/'tokenizer.json'))
    assert tokenizer.encode(workload['rendered_prompt'],add_special_tokens=False).ids==workload['input_ids']
    cfg=json.loads((MODEL/'config.json').read_text())['text_config'];cfg['_attn_implementation']='eager';config=SimpleNamespace(**cfg)
    official=load(ROOT/'sources',config,observe=False);store=Checkpoint(MODEL,ROOT/'tensors.json')
    cache=original_cache(official,config.layer_types);rotary=official['Qwen3_5TextRotaryEmbedding'](config)
    qualification=dict(physical=False,scope='Offline sequential CPU oracle; no physical operands are injected',source_extraction=official['extraction'],
        acceptance_sha256=frozen['sha256'],prompt_precomputation_allowed=True)
    (ROOT/'source-qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
    position=0;checks=0;started=time.monotonic();token_checks=[];failure=None;violations=[]
    evidence=ROOT/'reference';evidence.mkdir();log=(ROOT/'comparisons.jsonl').open('x')
    def compare(at,kind,index,bits,reference):
        nonlocal checks
        limit=criteria['full_vocabulary_logits'] if kind=='logits' else criteria['per_position_all64_layers_and_final_norm']
        result=metrics(bits,reference,limit);checks+=1
        record=dict(position=at,kind=kind,index=index,**result);log.write(json.dumps(record)+'\n');log.flush()
        if not result['passed']:
            violations.append(record)
            if not a.collect_all:raise ValueError('Frozen numerical criterion failed: '+json.dumps(record))
    def check_token(at,bits):
        if at>=len(workload['input_ids'])-1:
            index=at-len(workload['input_ids'])+1;selected=generation['generated_ids'][index]
            try:token_checks.append(dict(position=at,**selection(candidate['logits'][at],bits,selected,criteria['token_selection'])))
            except AssertionError as error:
                if not a.collect_all:raise
                record=dict(position=at,kind='token_selection',selected=selected,passed=False,error=str(error))
                token_checks.append(record);violations.append(record)
    def adopt_candidate(computed):
        nonlocal generation,positions,ids,text
        generation=json.loads((a.candidate/'generation.json').read_text())
        assert generation['physical'] and generation['capture_enabled'] and generation['workload']=='sunny_morning'
        assert generation['prompt_ids']==workload['input_ids']
        assert all(0<=i<248320 for i in generation['generated_ids'])
        ids=generation['prompt_ids']+generation['generated_ids'][:-1];positions=len(ids)
        assert computed<=positions==generation['processed_positions']<=96
        assert generation['all_layers']==64 and generation['vocabulary']==248320
        for name,shape in [('layers',(positions,64,5120)),('logits',(positions,248320)),('final_norm',(positions,5120))]:
            path=a.candidate/(name+'.bf16');record=json.loads((a.candidate/(name+'-capture.json')).read_text())
            assert record['elements']==math.prod(shape) and path.stat().st_size==math.prod(shape)*2 and sha(path)==record['sha256']
            candidate[name]=np.memmap(path,dtype='<u2',mode='r',shape=shape)
        text=tokenizer.decode(generation['generated_ids'],skip_special_tokens=True).strip()
        qualification.update(candidate_generation_sha256=sha(a.candidate/'generation.json'),positions=positions,decoded_text=text,
            prompt_positions_precomputed_before_capture_arrival=computed)
        (ROOT/'qualification.json').write_text(json.dumps(qualification,indent=2)+'\n')
        # Every precomputed prompt output is still checked under frozen bounds.
        for at in range(computed):
            for layer in range(64):compare(at,'layer',layer,candidate['layers'][at,layer],np.load(evidence/f'position{at}-layer{layer}.npy',allow_pickle=False))
            compare(at,'final_norm',0,candidate['final_norm'][at],np.load(evidence/f'position{at}-norm.npy',allow_pickle=False))
            bits=np.load(evidence/f'position{at}-logits.npy',allow_pickle=False)
            compare(at,'logits',0,candidate['logits'][at],bits);check_token(at,bits)
    class StreamingLayer(nn.Module):
        def __init__(self,index):super().__init__();self.index=index
        def forward(self,hidden_states,**kwargs):
            with torch.device('meta'):module=official['Qwen3_5DecoderLayer'](config,self.index)
            module=store.bind_layer(module,self.index);output=module(hidden_states,**kwargs)
            bits=output.view(torch.uint16).cpu().numpy().reshape(-1)
            np.save(evidence/f'position{position}-layer{self.index}.npy',bits)
            if candidate:compare(position,'layer',self.index,candidate['layers'][position,self.index],bits)
            del module;gc.collect();ctypes.CDLL(None).malloc_trim(0);return output
    def embedding(tokens):return store.tensor('model.language_model.embed_tokens.weight',int(tokens[0,0]),1)[None]
    with torch.device('meta'):norm=official['Qwen3_5RMSNorm'](5120,eps=config.rms_norm_eps)
    norm.weight=nn.Parameter(store.tensor('model.language_model.norm.weight'),requires_grad=False)
    model=SimpleNamespace(config=config,embed_tokens=embedding,layers=[StreamingLayer(i) for i in range(64)],rotary_emb=rotary,norm=norm)
    try:
        with torch.inference_mode():
            while True:
                if not candidate and a.candidate.is_dir():adopt_candidate(position)
                if not candidate and position==len(workload['input_ids']):
                    (ROOT/'waiting-candidate.json').write_text(json.dumps(dict(prompt_positions_completed=position,seconds=time.monotonic()-started))+'\n')
                    deadline=time.monotonic()+7200
                    while not a.candidate.is_dir():
                        if time.monotonic()>deadline:raise TimeoutError('Candidate capture arrival deadline')
                        time.sleep(5)
                    adopt_candidate(position)
                if position==len(ids):break
                token=ids[position]
                assert cache.get_seq_length()==position
                mask=torch.zeros((1,1,1,position+1),dtype=torch.bfloat16)
                output=official['text_model_forward'](model,input_ids=torch.tensor([[token]]),past_key_values=cache,
                    attention_mask={'full_attention':mask,'linear_attention':None},use_cache=True)
                hidden=output.last_hidden_state;bits=hidden.view(torch.uint16).cpu().numpy().reshape(-1)
                np.save(evidence/f'position{position}-norm.npy',bits)
                if candidate:compare(position,'final_norm',0,candidate['final_norm'][position],bits)
                logits=torch.empty(248320,dtype=torch.bfloat16)
                for start in range(0,248320,1024):
                    count=min(1024,248320-start);head=store.tensor('lm_head.weight',start,count)
                    logits[start:start+count]=torch.nn.functional.linear(hidden,head).reshape(-1)
                bits=logits.view(torch.uint16).numpy();np.save(evidence/f'position{position}-logits.npy',bits)
                if candidate:
                    compare(position,'logits',0,candidate['logits'][position],bits);check_token(position,bits)
                assert cache.get_seq_length()==position+1;store.check_unchanged()
                record=dict(completed_positions=position+1,total_positions=positions,checks=checks,candidate_loaded=bool(candidate),violations=len(violations),seconds=time.monotonic()-started)
                (ROOT/'progress.json').write_text(json.dumps(record)+'\n');print(json.dumps(record),flush=True)
                position+=1
        assert candidate and store.read_names==set(store.tensors) and checks==positions*66
        assert generation['eos_reached'] and generation['generated_ids'][-1]==248046
        sentences=[s.strip() for s in re.findall(r'[^.!?]+[.!?](?:\s|$)',text)]
        assert len(sentences)>=2 and all(len(s.split())>=3 for s in sentences[:2]) and text[-1] in '.!?'
        assert all(word in text.lower() for word in criteria['sequence']['required_topics'])
        assert len(generation['generated_ids'])-1>=16
        verify()
        assert not violations, 'Frozen numerical criteria failed; all requested comparisons retained'
    except BaseException as exc:
        failure=repr(exc);raise
    finally:
        log.close()
        result=dict(passed=failure is None,error=failure,physical=False,comparison_checks=checks,expected_comparison_checks=positions*66 if positions is not None else None,
            token_checks=token_checks,decoded_text=text,seconds=time.monotonic()-started,acceptance_sha256=frozen['sha256'],
            collect_all_requested=a.collect_all,violation_count=len(violations),first_violation=violations[0] if violations else None,
            scope='Offline full sequential-prefix numerical comparison. Final acceptance still requires correlated physical job release and source/artifact binding.')
        (ROOT/('COMPLETE.json' if failure is None else 'FAILURE.json')).write_text(json.dumps(result,indent=2)+'\n')

if __name__=='__main__':main()
