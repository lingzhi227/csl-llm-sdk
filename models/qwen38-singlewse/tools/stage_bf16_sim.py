"""Stage immutable BF16 resident source and bound every workstation phase."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--phase',choices=['stage','prepare','compile','run'],default='stage');a=p.parse_args()
assert len(a.attempt)==3 and a.attempt.isdigit()
name='resident-bf16-sim-'+a.attempt;dest='/srv/model-storage/qwen38-singlewse/runs/'+name
out=ROOT/'evidence'/name
if a.phase=='stage':
    files={n:(ROOT/'experiments/resident_bf16'/n).read_bytes() for n in ['layout.csl','pe.csl','prepare.py','run.py','experiment.json']}
    files['bf16_resident.csl']=(ROOT/'csl/bf16_resident.csl').read_bytes()
    for n in ['backend.py','bounded_client.py','check_sram.py','elf_inventory.py','run_micro_sim.py','source_gate.py']:
        files[n]=(ROOT/'runtime'/n).read_bytes()
    files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for n,v in files.items():
            item=tarfile.TarInfo(n);item.size=len(v);archive.addfile(item,io.BytesIO(v))
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
else:
    if a.phase=='prepare':command=['/usr/bin/python3','-u','prepare.py']
    elif a.phase=='compile':
        command=['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=16,7','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out']
    else:command=['/opt/cerebras/sdk/2.10.1/cs_python','run_micro_sim.py']
    unit='qwen38-single-bf16-'+a.phase+'-'+a.attempt
    cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax=2G','--property=MemorySwapMax=0',
        '--property=CPUAffinity=6 7','--property=TasksMax=128','--property=RuntimeMaxSec=300','--property=TimeoutStopSec=5','--property=LimitCORE=0',
        '--property=StandardOutput=append:'+dest+'/'+a.phase+'.log','--property=StandardError=append:'+dest+'/'+a.phase+'.log',
        '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','flock','-n','/srv/cerebras-workstation/heavy.lock',*command]
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    (out/(a.phase+'-dispatch.json')).write_text(json.dumps(dict(unit=unit,remote=dest,physical=False),indent=2)+'\n')
print(json.dumps(dict(phase=a.phase,remote=dest,physical=False)))
