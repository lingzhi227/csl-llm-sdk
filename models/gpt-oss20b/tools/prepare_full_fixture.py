"""Small original-reference acceptance data; no checkpoint tensors duplicated."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference',type=Path,required=True)
    p.add_argument('--transformer',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    receipt=json.loads((a.reference/'COMPLETE.json').read_text());assert receipt['tokens']==[13225,11,5922]
    tm=json.loads((a.transformer/'transformer-fixture.json').read_text())
    path=a.transformer/'transformer-fixture.npz';assert hashlib.sha256(path.read_bytes()).hexdigest()==tm['fixture_sha256']
    with np.load(path,allow_pickle=False) as f:frequencies=f['frequencies']
    expected={}
    for kind in ('attention','output'):
        expected[kind]=np.stack([np.stack([np.load(a.reference/f'step{step:02}-layer{layer:02}-{kind}.npy',allow_pickle=False)[-1] for layer in range(24)]) for step in range(2)])
    records=[json.loads(line) for line in (a.reference/'layers.jsonl').read_text().splitlines()]
    ids=np.zeros((2,24,4),np.uint16)
    for record in records:ids[record['step'],record['layer']]=record['routing'][0]['experts'][-1]
    a.output.mkdir(parents=True,exist_ok=False);path=a.output/'full-fixture.npz'
    with path.open('xb') as f:np.savez(f,frequencies=frequencies,expected_ids=ids,**expected)
    meta=dict(model=receipt['model'],revision=receipt['revision'],upstream_commit=receipt['upstream_commit'],
      initial_tokens=[13225],expected_generated_tokens=[11,5922],reference=receipt,fixture_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
      scope='full original 24-layer two-token greedy generation; saved layer states used only for post-generation comparison')
    (a.output/'full-fixture.json').write_text(json.dumps(meta,indent=2)+'\n');print(json.dumps(meta),flush=True)

if __name__=='__main__':main()
