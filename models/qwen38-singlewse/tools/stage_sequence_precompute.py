"""Begin independent prompt arithmetic while the physical request is running."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--candidate',required=True);p.add_argument('--collect-all',action='store_true');a=p.parse_args()
assert re.fullmatch('[0-9]{3}',a.attempt) and re.fullmatch('resident-generation-hw-[0-9]{3}',a.candidate)
name='reference-sequence-'+a.attempt;dest='/srv/model-storage/qwen38-singlewse/runs/'+name
remote='/srv/qwen38-singlewse-hardware/'+a.candidate
files={'run.py':(ROOT/'reference/sequence/run.py').read_bytes()}
for n in ['pinned.py','checkpoint.py']:files[n]=(ROOT/'reference/full'/n).read_bytes()
for f in (ROOT/'reference/sources').glob('*'):
    if f.is_file():files['sources/'+f.name]=f.read_bytes()
for n in ['tensors.json','full-acceptance-v1.json']:files[n]=(ROOT/'configs'/n).read_bytes()
files['full-acceptance-frozen.json']=(ROOT/'evidence/full-acceptance-frozen.json').read_bytes()
files['source_gate.py']=(ROOT/'runtime/source_gate.py').read_bytes()
files['candidate-origin.json']=(json.dumps(dict(remote=remote,request='request-00',generation_type='Physical candidate, not yet accepted'))+'\n').encode()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for n,v in files.items():
        e=tarfile.TarInfo(n);e.size=len(v);tar.addfile(e,io.BytesIO(v))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=30)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax=26G','--property=MemorySwapMax=0',
     '--property=CPUAffinity=6 7','--property=CPUQuota=200%','--property=TasksMax=64','--property=RuntimeMaxSec=10800',
     '--property=TimeoutStopSec=5','--property=LimitCORE=0','--property=StandardOutput=append:'+dest+'/run.log',
     '--property=StandardError=append:'+dest+'/run.log','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=2',
     'flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','-u','run.py','--candidate',dest+'/candidate','--await-candidate']
if a.collect_all:cmd.append('--collect-all')
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
receipt=dict(unit=unit,remote=dest,candidate=remote,physical=False,deadline_seconds=10800,prompt_precompute=True,candidate_capture_transfer_pending=True,collect_all_requested=a.collect_all)
(out/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
