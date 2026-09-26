"""Stage a new full-matrix experiment; retain model bytes on remote hosts only."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args();assert re.fullmatch(r'[0-9]{3}',a.attempt)
DEST='/srv/qwen38-singlewse-hardware/fp8-matrix-hw-'+a.attempt
SESSION=['sh','/path/to/alcf-session.sh','host']
files={n:(ROOT/'experiments/fp8_matrix'/n).read_bytes() for n in ['layout.csl','pe.csl','run.py','prepare.py','experiment.json']}
for n in ['fp8_matrix_block.csl','fp8_codec.csl','fp8_activation.csl']:files[n]=(ROOT/'csl'/n).read_bytes()
for n in ['backend.py','bounded_client.py','compile_hw.py','run_hw.py','supervise.py','job_capture.py','source_gate.py','check_sram.py','elf_inventory.py']:
    files[n]=(ROOT/'runtime'/n).read_bytes()
for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
stream=io.BytesIO()
with tarfile.open(fileobj=stream,mode='w') as t:
    for n,v in files.items():
        info=tarfile.TarInfo(n);info.size=len(v);t.addfile(info,io.BytesIO(v))
subprocess.run(SESSION+['mkdir '+shlex.quote(DEST)+' && tar -xf - -C '+shlex.quote(DEST)],input=stream.getvalue(),check=True,timeout=30)
out=ROOT/'evidence'/('fp8-matrix-hw-'+a.attempt);out.mkdir()
(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
if a.attempt=='001':
    reader=subprocess.Popen(['ssh','workstation','cat /srv/model-storage/qwen38-singlewse/model/layers-0.safetensors'],stdout=subprocess.PIPE)
    try:
        subprocess.run(SESSION+['cat > '+shlex.quote(DEST+'/layers-0.safetensors.partial')],stdin=reader.stdout,check=True,timeout=180)
    finally:
        reader.stdout.close();assert reader.wait(timeout=15)==0
else:
    subprocess.run(SESSION+['ln /srv/qwen38-singlewse-hardware/fp8-matrix-hw-001/layers-0.safetensors '+shlex.quote(DEST+'/layers-0.safetensors.partial')],check=True,timeout=20)
script='root_name='+repr(DEST)+'\n'+'''
import hashlib,json,os,resource,subprocess
from pathlib import Path
root=Path(root_name);os.chdir(root)
partial=root/'layers-0.safetensors.partial';h=hashlib.sha256()
with partial.open('rb') as f:
    for chunk in iter(lambda:f.read(4<<20),b''):h.update(chunk)
assert h.hexdigest()=='07f700e293baeaf3cd4240c3df1a948c4403f16961ea7979e86c8d6a9f8fd466'
partial.rename(root/'layers-0.safetensors')
def limit():
    resource.setrlimit(resource.RLIMIT_AS,(3<<30,3<<30))
    resource.setrlimit(resource.RLIMIT_CPU,(300,300))
    resource.setrlimit(resource.RLIMIT_FSIZE,(128<<20,128<<20))
env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
with (root/'prepare.log').open('x') as log:
    proc=subprocess.Popen(['timeout','360','/opt/cerebras/venv/bin/python','-u','prepare.py'],
        cwd=root,env=env,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True,preexec_fn=limit)
print(json.dumps(dict(root=str(root),prepare_pid=proc.pid,physical_dispatched=False)))
'''
r=subprocess.run(SESSION+['python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=60)
receipt=json.loads(r.stdout);(out/'staging.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
