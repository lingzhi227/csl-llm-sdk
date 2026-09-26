"""Exercise every weight transfer against actual checkpoint and coordinate roles."""
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from checkpoint import Checkpoint
import resident_geometry as geo
from resident_loader import audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    checkpoint=Checkpoint(a.model);digest=hashlib.sha256();start=time.monotonic();transfers=0
    matrix_owners=set();tensor_owners={}
    def consume(item):
        nonlocal transfers
        digest.update(json.dumps([item.symbol,item.x,item.y,item.width,item.height,item.count,item.tensor,item.source_bytes],separators=(',',':')).encode())
        digest.update(item.data.tobytes())
        for y in range(item.y,item.y+item.height):
            for x in range(item.x,item.x+item.width):
                role=geo.role_at(x,y)
                if item.symbol=='weights':
                    assert (x,y) not in matrix_owners;matrix_owners.add((x,y))
                    if '.experts.gate_up_proj_' in item.tensor:assert role.kind=='gate'
                    elif '.experts.down_proj_' in item.tensor:assert role.kind=='down'
                    elif item.tensor=='model.embed_tokens.weight':assert role.kind=='embedding'
                    elif item.tensor=='lm_head.weight':assert role.kind=='head'
                    elif '.router.' in item.tensor:assert role.kind=='router'
                    elif '.o_proj.' in item.tensor:assert role.kind=='o'
                    else:assert role.kind=='qkv'
                if item.symbol=='scales':assert role.kind in ('gate','down')
                if item.symbol=='gain':assert role.kind in ('attention_controller','moe_controller','head_controller')
                if item.symbol=='sinks':assert role.kind=='kv'
                if role.layer>=0:assert item.tensor.startswith(f'model.layers.{role.layer}.')
        transfers+=1
        if transfers%5000==0:print(json.dumps(dict(transfers=transfers,seconds=time.monotonic()-start)),flush=True)
    result=audit(checkpoint,consume)
    expected=sum(geo.role_at(x,y).kind in ('embedding','head','qkv','o','router','gate','down') for x in range(geo.WIDTH) for y in range(geo.HEIGHT))
    assert len(matrix_owners)==expected==562236
    result.update(unique_matrix_owners=len(matrix_owners),transfer_stream_sha256=digest.hexdigest(),seconds=time.monotonic()-start,
                  geometry='27 columns per layer, all 24 layers; full original parameters; first runtime KV capacity 96')
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:f.write(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
