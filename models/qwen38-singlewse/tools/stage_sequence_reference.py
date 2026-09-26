"""Relay bounded candidate captures to Mass1 and launch an offline CPU oracle."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--candidate',required=True);a=p.parse_args()
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
names=['generation.json','trace-check.json','layers.bf16','logits.bf16','final_norm.bf16','layers-capture.json','logits-capture.json','final_norm-capture.json']
sender='root='+repr(remote)+'\nnames='+repr(names)+'\n'+'''
import sys,tarfile,json
from pathlib import Path
p=Path(root)/'request-00'
for name in names:assert (p/name).is_file() and not (p/name).is_symlink()
assert sum((p/name).stat().st_size for name in names)<128<<20
assert json.loads((p/'generation.json').read_text())['physical']
assert json.loads((p/'trace-check.json').read_text())['all870000_endpoints_checked']
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
 for name in names:tar.add(p/name,arcname=name,recursive=False)
'''
receiver='names='+repr(names)+'\n'+'''
import sys,tarfile,json,hashlib
from pathlib import Path
root=Path('candidate.partial');root.mkdir();seen=set();total=0
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for m in tar:
  tar.members.clear();assert m.isfile() and m.name in names and m.name not in seen
  total+=m.size;assert total<128<<20;seen.add(m.name)
  with (root/m.name).open('xb') as sink:
   stream=tar.extractfile(m);left=m.size
   while left:
    raw=stream.read(min(left,1<<20));assert raw;sink.write(raw);left-=len(raw)
assert seen==set(names)
for name in ['layers','logits','final_norm']:
 receipt=json.loads((root/(name+'-capture.json')).read_text());h=hashlib.sha256();path=root/(name+'.bf16')
 with path.open('rb') as f:
  for raw in iter(lambda:f.read(1<<20),b''):h.update(raw)
 assert path.stat().st_size==receipt['bytes'] and h.hexdigest()==receipt['sha256']
root.rename('candidate');print(json.dumps(dict(passed=True,files=len(seen),bytes=total,payload_stored_on_mac=False)))
'''
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--quiet','--wait','--pipe','--unit='+unit+'-transfer','--property=WorkingDirectory='+dest,
     '--property=MemoryMax=256M','--property=MemorySwapMax=0','--property=CPUAffinity=6 7','--property=CPUQuota=50%',
     '--property=TasksMax=32','--property=RuntimeMaxSec=180','--property=TimeoutStopSec=5','--property=LimitCORE=0',
     'flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','-c',receiver]
with (out/'transfer.log').open('x') as log:
    send=subprocess.Popen(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(sender)],stdout=subprocess.PIPE,stderr=log)
    try:
        received=subprocess.run(['ssh','workstation',shlex.join(cmd)],stdin=send.stdout,stdout=log,stderr=log,timeout=190)
        send.stdout.close();assert send.wait(timeout=20)==0 and received.returncode==0
    finally:
        if send.poll() is None:send.terminate();send.wait(timeout=10)
cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax=26G','--property=MemorySwapMax=0',
     '--property=CPUAffinity=6 7','--property=CPUQuota=200%','--property=TasksMax=64','--property=RuntimeMaxSec=10800',
     '--property=TimeoutStopSec=5','--property=LimitCORE=0','--property=StandardOutput=append:'+dest+'/run.log',
     '--property=StandardError=append:'+dest+'/run.log','--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=2',
     'flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','-u','run.py','--candidate',dest+'/candidate']
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
receipt=dict(unit=unit,remote=dest,candidate=remote,physical=False,deadline_seconds=10800)
(out/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
