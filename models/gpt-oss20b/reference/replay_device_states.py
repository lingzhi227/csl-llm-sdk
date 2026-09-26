"""Offline original-operator replay on captured device inputs, never inference H2D.

For the first position, replay each attention/MLP independently on its actual
device input. This separates local operator error from accumulated BF16 drift.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from run_reference import Checkpoint, ModelConfig, RMSNorm, attention, mlp, dense, from_bits, parameter


def bits(tensor):
    return tensor.detach().contiguous().view(torch.int16).numpy().view(np.uint16)


def compare(actual, expected):
    a = (actual.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    b = (expected.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    delta = a-b
    def ordered(x):
        x=x.astype(np.int32)
        return np.where(x & 32768, -(x & 32767), x)
    ulp=np.abs(ordered(actual)-ordered(expected))
    different=np.flatnonzero(actual.reshape(-1)!=expected.reshape(-1))
    return dict(finite=bool(np.isfinite(a).all() and np.isfinite(b).all()),
        bit_exact=int(np.count_nonzero(actual==expected)),values=int(a.size),
        max_abs_error=float(np.max(np.abs(delta))),rms_error=float(np.sqrt(np.mean(delta**2))),
        relative_l2=float(np.linalg.norm(delta)/max(np.linalg.norm(b),1e-30)),
        max_bf16_ulp=int(ulp.max()),
        first_differences=[dict(index=int(i),device=float(a.reshape(-1)[i]),
          replay=float(b.reshape(-1)[i])) for i in different[:8]])


@torch.inference_mode()
def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True)
    p.add_argument('--actual',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    torch.set_num_threads(2);torch.set_num_interop_threads(1)
    receipt=json.loads((args.model/'COMPLETE.json').read_text())
    assert receipt['revision']=='6cee5e81ee83917806bbde320786a8fb61efebee'
    args.output.mkdir(parents=True,exist_ok=False)
    with np.load(args.actual,allow_pickle=False) as archive:
        actual={name:archive[name] for name in ('attention','output','ids')}
        counters=archive['counters']
        assert int(counters[0,0,0])==1, 'This replay covers position zero only'
    del counters
    assert actual['attention'].shape==actual['output'].shape==(24,2880)
    c=Checkpoint(args.model);config=ModelConfig(num_hidden_layers=24,num_experts=32)
    embedding=np.array(c.row_slice('model.embed_tokens.weight',13225)[0],copy=True)
    started=time.monotonic();records=[]
    for layer in range(24):
        previous=embedding if layer==0 else actual['output'][layer-1]
        module=attention(c,config,layer)
        expected_attention=bits(module(from_bits(previous[None])))
        del module
        module=mlp(c,config,layer);routing=[]
        def trace_gate(module,operands,logits):
            selected=torch.topk(logits,4,dim=-1,sorted=True)
            routing.append(dict(ids=selected.indices[0].tolist(),
              logits=logits[0].float().tolist(),selected_logits=selected.values[0].float().tolist()))
        hook=module.gate.register_forward_hook(trace_gate)
        expected_output=bits(module(from_bits(actual['attention'][layer][None])))
        hook.remove();del module
        record=dict(layer=layer,attention=compare(actual['attention'][layer][None],expected_attention),
          mlp=compare(actual['output'][layer][None],expected_output),router=routing[0],
          device_ids=actual['ids'][layer].tolist(),
          routing_matches_original_on_actual_input=actual['ids'][layer].tolist()==routing[0]['ids'])
        records.append(record)
        with (args.output/'layers.jsonl').open('a') as out:out.write(json.dumps(record)+'\n')
        print(json.dumps(dict(layer=layer,attention_relative_l2=record['attention']['relative_l2'],
          mlp_relative_l2=record['mlp']['relative_l2'],ids_exact=record['routing_matches_original_on_actual_input'])),flush=True)
    norm=RMSNorm(2880,device='meta')
    norm.scale=parameter(dense(c,'model.norm.weight').float())
    hidden=norm(from_bits(actual['output'][-1:]))
    head=dense(c,'lm_head.weight');logits=torch.nn.functional.linear(hidden,head);del head
    best=torch.topk(logits[0],8)
    report=dict(cpu_diagnostic=True,physical_inference=False,
      scope='unchanged OpenAI operators independently replayed on position-zero device inputs; accumulated drift is not fed into an acceptance threshold',
      revision=receipt['revision'],actual_sha256=hashlib.sha256(args.actual.read_bytes()).hexdigest(),
      seconds=time.monotonic()-started,records=records,
      replay_head_top_ids=best.indices.tolist(),replay_head_logits=best.values.float().tolist())
    (args.output/'COMPLETE.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)


if __name__=='__main__':main()
