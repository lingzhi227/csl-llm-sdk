"""One fresh bounded transfer; relay pipe holds bytes in memory only."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert re.fullmatch(r'[0-9]{3}',a.attempt)
SESSION=['sh','/path/to/alcf-session.sh','host']
name='checkpoint-transfer-'+a.attempt
ws='/srv/model-storage/qwen38-singlewse/runs/'+name
alcf='/srv/qwen38-singlewse-hardware/'+name
files={'stream.py':(ROOT/'tools/checkpoint_stream.py').read_bytes(),'hub.json':(ROOT/'configs/hub.json').read_bytes()}
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
buf=io.BytesIO()
with tarfile.open(fileobj=buf,mode='w') as tar:
    for n,v in files.items():
        ent=tarfile.TarInfo(n);ent.size=len(v);tar.addfile(ent,io.BytesIO(v))
for prefix,dest in [(['ssh','workstation'],ws),(SESSION,alcf)]:
    subprocess.run(prefix+['mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=buf.getvalue(),check=True,timeout=25)
out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
unit='qwen38-single-'+name
cmd=['systemd-run','--user','--quiet','--wait','--pipe','--unit='+unit,'--property=WorkingDirectory='+ws,
     '--property=MemoryMax=256M','--property=MemorySwapMax=0','--property=CPUAffinity=6 7','--property=CPUQuota=50%',
     '--property=TasksMax=32','--property=RuntimeMaxSec=4200','--property=TimeoutStopSec=5','--property=LimitCORE=0',
     'flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','-u','stream.py','send']
sender=None;receiver=None;started=time.monotonic();error=None
try:
    with (out/'sender.log').open('x') as log:
        sender=subprocess.Popen(['ssh','workstation',shlex.join(cmd)],stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log)
        receiver=subprocess.Popen(SESSION+['cd '+shlex.quote(alcf)+' && timeout --signal=TERM --kill-after=5 4200 python3 -u stream.py receive'],stdin=sender.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        sender.stdout.close()
        (out/'ACTIVE.json').write_text(json.dumps(dict(sender_pid=sender.pid,receiver_pid=receiver.pid,unit=unit,ws=ws,alcf=alcf,physical_job=False))+'\n')
        raw,err=receiver.communicate(timeout=4250)
        (out/'receiver.log').write_bytes(raw+err)
        assert receiver.returncode==0,('receiver',receiver.returncode)
        assert sender.wait(timeout=20)==0,'sender'
except BaseException as exc:
    error=repr(exc)
    for child in (receiver,sender):
        if child is not None and child.poll() is None:
            child.terminate()
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:child.kill();child.wait(timeout=10)
    raise
finally:
    (out/'receipt.json').write_text(json.dumps(dict(passed=error is None,error=error,seconds=time.monotonic()-started,sender_code=sender.poll() if sender else None,receiver_code=receiver.poll() if receiver else None,payload_stored_on_mac=False),indent=2)+'\n')
