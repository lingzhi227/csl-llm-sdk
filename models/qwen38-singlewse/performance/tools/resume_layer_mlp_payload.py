"""Resume an interrupted fixture transfer after verifying every existing byte.

No hardware is allocated. Source and destination stream at most1MiB per read;
partial files are never treated as admitted fixtures, and the full immutable
oracle manifest is checked again before writing the staging completion receipt.
"""
import argparse,hashlib,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
SESSION=['sh','/path/to/alcf-session.sh','host']


def remote(connection,script,timeout=30):
    return subprocess.run(connection+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=timeout).stdout


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
    if not re.fullmatch('[0-9]{3}',args.attempt):raise ValueError('Attempt')
    out=ROOT/'performance/evidence'/('layer-mlp-hw-'+args.attempt)
    staging=json.loads((out/'staging.json').read_text());dest=staging['remote']
    if not dest.endswith('/layer-mlp-hw-'+args.attempt) or (out/'payload-staging.json').exists():raise ValueError('Already admitted or wrong destination')
    meta=json.loads((ROOT/'performance/evidence/layer-mlp-reference-001/fixture.json').read_text())
    source='/srv/model-storage/qwen38-singlewse/runs/layer-mlp-reference-001';records=[]
    for name in ['banks.npy','fixture.npz','silu-table.npy']:
        # Refuse live receiving writers instead of racing or replacing one.
        check='root='+repr(dest)+'\nname='+repr(name)+'\n'+r'''import hashlib,json,os
from pathlib import Path
r=Path(root)
assert not (r/'SUPERVISOR.json').exists(), 'Never mutate a dispatched fixture'
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  args=(p/'cmdline').read_bytes().split(b'\0')
  if args and args[0].split(b'/')[-1]==b'tar' and root.encode() in args:raise ValueError('Existing tar receiver is still active')
 except (FileNotFoundError,PermissionError):pass
f=r/name;h=hashlib.sha256();size=f.stat().st_size if f.exists() else 0
if f.exists():
 with f.open('rb') as stream:
  for b in iter(lambda:stream.read(1048576),b''):h.update(b)
print(json.dumps(dict(bytes=size,sha256=h.hexdigest())))
'''
        prefix=json.loads(remote(SESSION,check));offset=prefix['bytes']
        src_check='path='+repr(source+'/'+name)+'\noffset='+str(offset)+'\n'+'''import hashlib,json
from pathlib import Path
p=Path(path);assert p.stat().st_size>=offset;h=hashlib.sha256();left=offset
with p.open('rb') as f:
 while left:
  b=f.read(min(1048576,left));assert b;h.update(b);left-=len(b)
print(json.dumps(dict(bytes=p.stat().st_size,prefix_sha256=h.hexdigest())))
'''
        original=json.loads(remote(['ssh','workstation'],src_check))
        if original['prefix_sha256']!=prefix['sha256']:raise ValueError('Existing prefix differs; refusing overwrite')
        length=original['bytes']-offset
        if length:
            sender='path='+repr(source+'/'+name)+'\noffset='+str(offset)+'\n'+'''import sys
with open(path,'rb') as f:
 f.seek(offset)
 while True:
  b=f.read(1048576)
  if not b:break
  sys.stdout.buffer.write(b)
'''
            receiver='path='+repr(dest+'/'+name)+'\noffset='+str(offset)+'\nlength='+str(length)+'\n'+'''import os,sys
from pathlib import Path
p=Path(path);assert (p.stat().st_size if p.exists() else 0)==offset
left=length
with p.open('ab') as f:
 while left:
  b=sys.stdin.buffer.read(min(1048576,left));assert b,'Incomplete stream';f.write(b);left-=len(b)
 assert not sys.stdin.buffer.read(1),'Oversized stream'
 f.flush();os.fsync(f.fileno())
'''
            reader=subprocess.Popen(['ssh','workstation','python3 -c '+shlex.quote(sender)],stdout=subprocess.PIPE)
            try:subprocess.run(SESSION+['python3 -c '+shlex.quote(receiver)],stdin=reader.stdout,check=True,timeout=600)
            finally:
                reader.stdout.close()
                try:
                    if reader.wait(timeout=20)!=0:raise RuntimeError('Source stream failed')
                except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
        records.append(dict(file=name,verified_existing_bytes=offset,transferred_bytes=length,total_bytes=original['bytes']))
        print(json.dumps(records[-1]),flush=True)
    verify='root='+repr(dest)+'\nhashes='+repr(meta['hashes'])+'\n'+'''import hashlib
from pathlib import Path
for name,expected in hashes.items():
 h=hashlib.sha256()
 with (Path(root)/name).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 assert h.hexdigest()==expected,name
'''
    remote(SESSION,verify)
    with (out/'payload-staging.json').open('x') as f:json.dump(dict(payload_verified=True,resumed=True,files=records,allocated_bank_bytes=meta['allocated_bank_bytes']),f,indent=2);f.write('\n')


if __name__=='__main__':main()
