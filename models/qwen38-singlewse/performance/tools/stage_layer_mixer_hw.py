"""Freeze complete original mixer qualification and bounded remote payload relay.

Uses the admitted018 CSL unchanged. No WSE job is submitted by this staging tool.
Large banks/oracle arrays pass through bounded pipes, never Mac payload files.
"""
import argparse, ast, hashlib, io, json, re, shlex, subprocess, sys, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'performance/runtime'))
from mixer_qualification import validate, SHAPES


def digest(raw):return hashlib.sha256(raw).hexdigest()


def archive(files):
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as tar:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
    return payload.getvalue()


def freeze(files):
    for n,b in files.items():
        if n.endswith('.py'):ast.parse(b)
    files['source-manifest.json']=(json.dumps(dict(files={n:digest(b) for n,b in files.items()}),indent=2)+'\n').encode()


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--reuse-payload');a=p.parse_args()
    if not re.fullmatch('[0-9]{3}',a.attempt):raise ValueError('Attempt')
    if a.reuse_payload is not None and not re.fullmatch('[0-9]{3}',a.reuse_payload):raise ValueError('Payload origin')
    e=ROOT/'performance/evidence';name='layer-mixer-hw-'+a.attempt;out=e/name
    source_name='layer-mixer-payload-'+a.attempt;source_out=e/source_name
    if out.exists() or source_out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Active workstation owner: '+active.stdout)
    compiled=e/'layer-mlp-compile-018';banks=e/'layer-joint-reference-003';oracle=e/'layer-mixer-reference-001'
    for r in [compiled,banks,oracle]:
        if not json.loads((r/'workstation-release.json').read_text())['workstation_released']:raise ValueError('Unreleased prerequisite')
    if not json.loads((compiled/'COMPLETE.json').read_text())['sram_passed']:raise ValueError('Complete SRAM admission')
    if (banks/'source/compiled-source-manifest.json').read_bytes()!=(compiled/'source-manifest.json').read_bytes():raise ValueError('Bank compile provenance')
    proof=json.loads((banks/'bank-remap-proof.json').read_text());meta=json.loads((oracle/'fixture.json').read_text())
    if not proof['passed'] or not meta['passed']:raise ValueError('Original bank/oracle gate')
    files={n:(compiled/'source'/n).read_bytes() for n in json.loads((compiled/'source-manifest.json').read_text())['files'] if n.endswith('.csl')}
    for n in ['profiles.json','joint-stage.json','joint-bank-placement.json','mixer-setups.json','device-network.json']:
        files[n]=(compiled/'source'/n).read_bytes()
    declared=json.loads((compiled/'source-manifest.json').read_text())['files']
    if any(digest(b)!=declared[n] for n,b in files.items()):raise ValueError('Compiled source identity changed')
    files['compiled-source-manifest.json']=(compiled/'source-manifest.json').read_bytes()
    files['fixture.json']=(oracle/'fixture.json').read_bytes();files['bank-index.json']=(banks/'bank-index.json').read_bytes()
    plan=validate(*(json.loads(files[n]) for n in ['joint-stage.json','device-network.json','mixer-setups.json','profiles.json','bank-index.json']))
    if meta['original_matrices']!=5 or meta['output_values_per_case']!=21600 or meta['cases_count']!=4:raise ValueError('Complete independent oracle')
    hashes=dict(proof['hashes']);hashes['fixture.npz']=meta['fixture_sha256'];hashes['fixture.json']=digest(files['fixture.json'])
    if a.reuse_payload:
        prior=e/('layer-mixer-hw-'+a.reuse_payload)
        payload=json.loads((prior/'payload-staging.json').read_text())
        audit=json.loads((prior/'compile-audit.json').read_text())
        if not payload['payload_verified'] or any(payload['hashes'][n]!=hashes[n] for n in ['banks.npy','fixture.npz']):
            raise ValueError('Prior payload identity')
        if audit['cleanup_errors'] or not audit['jobs'] or any(not j['released'] for j in audit['jobs']):
            raise ValueError('Prior compiler release')
        if (prior/'run-audit.json').exists() or (prior/'COMPLETE.json').exists():
            raise ValueError('Payload reuse requires a pre-runtime attempt')
    config=dict(complete_original_mixer=True,stage='layer_00',revision=meta['revision'],application=[78,146],application_pes=11388,
                mixer_shapes=[list(s) for s in SHAPES],fabric_offset=[67,1],logical_origin=[63,0],artifact_single_message_limit=128<<20,
                run_progress_timeout_seconds=30,payload_hashes=hashes,acceptance=meta['criterion'],full_model=False,
                scope='All five complete original layer0 mixer contractions, real full-K reduction, zero/change/replay, full original bank retention and drain. Diagnostic host root consumption; no conv/GDN/complete-layer/model execution or serving-speed acceptance.')
    files['experiment.json']=(json.dumps(config,indent=2)+'\n').encode()
    for n in ['bounded_client.py','run_hw.py','job_capture.py','source_gate.py','elf_inventory.py']:
        files[n]=(ROOT/'runtime'/n).read_bytes()
    for n in ['backend.py','bounded_compiler.py','supervise.py','check_sram.py','placement.py','progress.py','progress_lifecycle.py','device_client.py','mixer_qualification.py','run_layer_mlp.py']:
        files[n]=(ROOT/'performance/runtime'/n).read_bytes()
    files['run.py']=(ROOT/'performance/runtime/run_layer_mixer.py').read_bytes()
    files['compile_hw.py']=(ROOT/'performance/runtime/compile_layer_mlp.py').read_bytes()
    files['supervise.py']=files['supervise.py'].replace(b'from runtime.lifecycle import stage, query, TERMINAL',b'from progress_lifecycle import stage, query, TERMINAL')
    for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
    files['stager.py']=Path(__file__).read_bytes();freeze(files)
    destination='/srv/qwen38-singlewse-hardware/'+name
    session=['sh','/path/to/alcf-session.sh','host']
    subprocess.run(session+['mkdir '+shlex.quote(destination)+' && tar -xf - -C '+shlex.quote(destination)],input=archive(files),check=True,timeout=30)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'staging.json').write_text(json.dumps(dict(remote=destination,physical_dispatched=False,compiled='layer-mlp-compile-018',banks='layer-joint-reference-003',oracle='layer-mixer-reference-001'),indent=2)+'\n')
    if a.reuse_payload:
        original='/srv/qwen38-singlewse-hardware/layer-mixer-hw-'+a.reuse_payload
        script='root_name='+repr(destination)+'\norigin='+repr(original)+'\nhashes='+repr({n:hashes[n] for n in ['banks.npy','fixture.npz']})+'\n'+'''import hashlib,json,os,resource
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(256<<20,256<<20));resource.setrlimit(resource.RLIMIT_CPU,(30,30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
root=Path(root_name);source=Path(origin);sizes={}
assert (source/'FAILURE.json').is_file() and not (source/'run.jobs').exists()
audit=json.loads((source/'compile-audit.json').read_text())
assert not audit['cleanup_errors'] and audit['jobs'] and all(j['released'] for j in audit['jobs'])
for n,expected in hashes.items():
 p=source/n;s=p.stat();assert p.is_file() and 0<s.st_size<=512<<20
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 assert h.hexdigest()==expected and p.stat()==s
 os.link(p,root/n);assert (root/n).stat().st_ino==s.st_ino;sizes[n]=s.st_size
print(json.dumps(dict(payload_verified=True,physical_job_submitted=False,hashes=hashes,bytes=sizes,reused_from=origin,method='verified immutable payload hardlinks; no duplicate payload storage')))
'''
        result=subprocess.run(session+['timeout --signal=TERM --kill-after=5 45 python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=55)
        receipt=json.loads(result.stdout);(out/'payload-staging.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(name=name,payload_verified=True,physical_dispatched=False,reused_from=original)))
        return
    source_root='/srv/model-storage/qwen38-singlewse/runs/'+source_name
    paths={'banks.npy':json.loads((banks/'dispatch.json').read_text())['remote']+'/banks.npy',
           'fixture.npz':json.loads((oracle/'dispatch.json').read_text())['remote']+'/fixture.npz'}
    sf={'payload.json':(json.dumps(dict(paths=paths,hashes={n:hashes[n] for n in paths}),indent=2)+'\n').encode()}
    sf['stream.py']=rb'''import hashlib,json,sys,tarfile,time
from pathlib import Path
started=time.monotonic();config=json.loads(Path('payload.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
try:
 for n,h in json.loads(Path('source-manifest.json').read_text())['files'].items():assert sha(Path(n))==h
 stamps={}
 for n,p in config['paths'].items():
  path=Path(p);assert 0<path.stat().st_size<=512<<20
  assert sha(path)==config['hashes'][n];stamps[n]=(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
  for n,p in config['paths'].items():tar.add(p,arcname=n,recursive=False)
 sys.stdout.buffer.flush()
 for n,p in config['paths'].items():
  t=Path(p).stat();assert (t.st_ino,t.st_size,t.st_mtime_ns)==stamps[n]
 for n,h in json.loads(Path('source-manifest.json').read_text())['files'].items():assert sha(Path(n))==h
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,streamed_bytes=sum(s[1] for s in stamps.values()),hashes=config['hashes'],seconds=time.monotonic()-started))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
'''
    freeze(sf)
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(source_root)+' && tar -xf - -C '+shlex.quote(source_root)],input=archive(sf),check=True,timeout=30)
    unit='qwen38-single-'+source_name
    source_out.mkdir();(source_out/'source-manifest.json').write_bytes(sf['source-manifest.json'])
    (source_out/'dispatch.json').write_text(json.dumps(dict(remote=source_root,unit=unit,physical=False),indent=2)+'\n')
    cmd=['systemd-run','--user','--wait','--pipe','--unit='+unit,'--property=MemoryMax=256M','--property=MemorySwapMax=0','--property=TasksMax=16',
         '--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=600','--property=TimeoutStopSec=5','--property=KillMode=control-group',
         '--property=LimitFSIZE=1048576','--property=LimitCORE=0','--working-directory='+source_root,
         '/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','stream.py']
    receiver='root_name='+repr(destination)+'\nhashes='+repr({n:hashes[n] for n in paths})+'\n'+'''import hashlib,json,os,resource,sys,tarfile
from pathlib import Path
resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20));resource.setrlimit(resource.RLIMIT_CORE,(0,0));resource.setrlimit(resource.RLIMIT_CPU,(45,45));resource.setrlimit(resource.RLIMIT_FSIZE,(512<<20,512<<20))
root=Path(root_name);seen=set();sizes={}
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|') as tar:
 for member in tar:
  assert member.name in hashes and member.name not in seen and member.isfile() and 0<member.size<=512<<20
  seen.add(member.name);h=hashlib.sha256();size=0
  with tar.extractfile(member) as source,(root/member.name).open('xb') as target:
   for b in iter(lambda:source.read(1<<20),b''):target.write(b);h.update(b);size+=len(b)
  assert size==member.size and h.hexdigest()==hashes[member.name];sizes[member.name]=size
assert seen==set(hashes)
print(json.dumps(dict(payload_verified=True,physical_job_submitted=False,hashes=hashes,bytes=sizes)))
'''
    # Both ends have independent finite bounds; only pipe buffers are local.
    with (source_out/'relay.log').open('x') as log:
        reader=subprocess.Popen(['ssh','workstation',shlex.join(cmd)],stdout=subprocess.PIPE,stderr=log)
        try:
            target=subprocess.run(session+['timeout --signal=TERM --kill-after=5 630 python3 -c '+shlex.quote(receiver)],stdin=reader.stdout,capture_output=True,check=True,timeout=650)
        finally:
            reader.stdout.close()
            try:code=reader.wait(timeout=20)
            except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
        if code:raise RuntimeError('Bounded payload sender failed')
    receipt=json.loads(target.stdout);(out/'payload-staging.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(name=name,payload_verified=True,physical_dispatched=False,internal_banks=len(plan['device_banks']),root_ports=sum(map(len,plan['roots'])))))


if __name__=='__main__':main()
