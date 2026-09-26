"""Freeze a bounded reference run; retain all payloads on Mass1."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--full',action='store_true');a=p.parse_args()
assert re.fullmatch(r'[0-9]{3}',a.attempt)
kind='full' if a.full else 'qualify'
dest='/srv/model-storage/qwen38-singlewse/runs/reference-'+kind+'-'+a.attempt
files={n.name:n.read_bytes() for n in (ROOT/'reference/full').glob('*') if n.is_file()}
files.update({'sources/'+n.name:n.read_bytes() for n in (ROOT/'reference/sources').glob('*') if n.is_file()})
files['tensors.json']=(ROOT/'configs/tensors.json').read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as t:
    for n,v in files.items():
        info=tarfile.TarInfo(n);info.size=len(v);t.addfile(info,io.BytesIO(v))
subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=20)
out=ROOT/'evidence'/('reference-'+kind+'-'+a.attempt);out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
command=['/usr/bin/python3','-u','run.py']+([] if a.full else ['--qualify'])
unit='qwen38-single-reference-'+kind+'-'+a.attempt
cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax='+('26G' if a.full else '6G'),'--property=MemorySwapMax=0',
     '--property=CPUAffinity=6 7','--property=TasksMax=64','--property=RuntimeMaxSec='+('7200' if a.full else '120'),
     '--property=TimeoutStopSec=5','--property=LimitCORE=0',
     '--property=StandardOutput=append:'+dest+'/run.log','--property=StandardError=append:'+dest+'/run.log',
     '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=2','flock','-n','/srv/cerebras-workstation/heavy.lock',*command]
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
receipt=dict(unit=unit,remote=dest,physical=False,kind=kind)
(out/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
