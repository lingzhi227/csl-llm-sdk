"""Relay <=128MiB immutable capture beside the active bounded CPU oracle."""
import argparse,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('--attempt',required=True);a=p.parse_args()
assert re.fullmatch('reference-sequence-[0-9]{3}',a.reference) and re.fullmatch('[0-9]{3}',a.attempt)
receipt=json.loads((ROOT/'evidence'/a.reference/'dispatch.json').read_text());dest=receipt['remote'];remote=receipt['candidate']
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
assert not Path('candidate').exists()
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
unit='qwen38-single-'+a.reference+'-capture-'+a.attempt
# This is a <=128MiB streaming file relay, not a second heavy model worker.
# The existing oracle retains the shared heavy-job lock; relay has a256MiB cap.
cmd=['systemd-run','--user','--quiet','--wait','--pipe','--unit='+unit,'--property=WorkingDirectory='+dest,
     '--property=MemoryMax=256M','--property=MemorySwapMax=0','--property=CPUAffinity=6 7','--property=CPUQuota=50%',
     '--property=TasksMax=32','--property=RuntimeMaxSec=180','--property=TimeoutStopSec=5','--property=LimitCORE=0',
     '/usr/bin/python3','-c',receiver]
with (ROOT/'evidence'/a.reference/('capture-transfer-'+a.attempt+'.log')).open('x') as log:
    send=subprocess.Popen(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(sender)],stdout=subprocess.PIPE,stderr=log)
    try:
        received=subprocess.run(['ssh','workstation',shlex.join(cmd)],stdin=send.stdout,stdout=log,stderr=log,timeout=190)
        send.stdout.close();assert send.wait(timeout=20)==0 and received.returncode==0
    finally:
        if send.poll() is None:send.terminate();send.wait(timeout=10)
print(json.dumps(dict(capture_transferred=True,reference=a.reference,physical_source=remote)))
