import datetime,fcntl,hashlib,json,resource,signal
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(134217728,134217728));resource.setrlimit(resource.RLIMIT_CPU,(15,15));signal.alarm(20)
f=open('/srv/cerebras-workstation/heavy.lock','a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
r=Path('/srv/model-storage/qwen38-singlewse/runs/layer-mlp-compile-014');files={};size=0
for p in sorted(r.glob('out/bin/*.elf')):
 b=p.read_bytes();size+=len(b);files[str(p.relative_to(r))]=dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),attempt='layer-mlp-compile-014',source_manifest_sha256=hashlib.sha256((r/'source-manifest.json').read_bytes()).hexdigest(),files=files,total_bytes=size,complete=False,accepted=False,scope='Post-release identity of every partial ELF still present in the immutable failed compile directory. Does not certify missing PE coverage or SRAM admission.')))
