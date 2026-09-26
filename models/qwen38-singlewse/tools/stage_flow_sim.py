"""Bounded FP8 flow experiment with frozen inputs and exact source hashes."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE='/srv/model-storage/qwen38-singlewse'
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--run',action='store_true');a=p.parse_args()
assert re.fullmatch(r'[0-9]{3}',a.attempt)
dest=BASE+'/runs/fp8-flow-sim-'+a.attempt
if not a.run:
    files={n:(ROOT/'experiments/fp8_flow'/n).read_bytes() for n in ['layout.csl','pe.csl','prepare.py','run.py','experiment.json']}
    files.update({n:(ROOT/'csl'/n).read_bytes() for n in ['fp8_codec.csl','fp8_activation.csl','fp8_block_gemv.csl']})
    files.update({n:(ROOT/'runtime'/n).read_bytes() for n in ['backend.py','check_sram.py','elf_inventory.py','run_micro_sim.py']})
    files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as t:
        for n,v in files.items():
            info=tarfile.TarInfo(n);info.size=len(v);t.addfile(info,io.BytesIO(v))
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=20)
    (ROOT/'evidence'/('fp8-flow-sim-'+a.attempt)).mkdir()
    (ROOT/'evidence'/('fp8-flow-sim-'+a.attempt)/'source-manifest.json').write_bytes(files['source-manifest.json'])
    command=['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=10,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'];phase='compile'
else:
    # Sequential preparation and simulation, under one finite owner and lock.
    command=['/bin/sh','-c','/usr/bin/python3 prepare.py && /opt/cerebras/sdk/2.10.1/cs_python run_micro_sim.py'];phase='run'
unit='qwen38-single-flow-'+phase+'-'+a.attempt
cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax=2G','--property=MemorySwapMax=0',
     '--property=CPUAffinity=6 7','--property=TasksMax=128','--property=RuntimeMaxSec=300','--property=TimeoutStopSec=5','--property=LimitCORE=0',
     '--property=StandardOutput=append:'+dest+'/'+phase+'.log','--property=StandardError=append:'+dest+'/'+phase+'.log',
     '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','flock','-n','/srv/cerebras-workstation/heavy.lock',*command]
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20);print(json.dumps(dict(unit=unit,remote=dest)))
