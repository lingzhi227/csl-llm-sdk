"""Bounded workstation compile/simulation of the resident bus, after owner exit."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE='/srv/model-storage/qwen38-singlewse'
p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--phase',choices=['stage','compile','run'],default='stage');args=p.parse_args()
assert re.fullmatch(r'[0-9]{3}',args.attempt)
name='resident-bus-sim-'+args.attempt;dest=BASE+'/runs/'+name
if args.phase=='stage':
    files={n:(ROOT/'experiments/resident_bus'/n).read_bytes() for n in ['layout.csl','pe.csl','run.py','experiment.json']}
    for n in ['fp8_matrix_block.csl','fp8_codec.csl','fp8_activation.csl']:files[n]=(ROOT/'csl'/n).read_bytes()
    for n in ['backend.py','bounded_client.py','check_sram.py','elf_inventory.py','run_micro_sim.py','source_gate.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
    files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for n,v in files.items():
            item=tarfile.TarInfo(n);item.size=len(v);archive.addfile(item,io.BytesIO(v))
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=20)
    # Pipe the small existing frozen fixture between remotes; do not save payloads on Mac.
    source='/srv/qwen38-singlewse-hardware/resident-bus-hw-004'
    reader=subprocess.Popen(['sh','/path/to/alcf-session.sh','host','tar -cf - -C '+shlex.quote(source)+' fixture.npz fixture.json'],stdout=subprocess.PIPE)
    try:subprocess.run(['ssh','workstation','tar -xf - -C '+shlex.quote(dest)],stdin=reader.stdout,check=True,timeout=30)
    finally:reader.stdout.close();assert reader.wait(timeout=20)==0
    out=ROOT/'evidence'/name;out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    print(json.dumps(dict(remote=dest,staged=True,started=False)));raise SystemExit
if args.phase=='compile':
    command=['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=16,7','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out']
else:command=['/opt/cerebras/sdk/2.10.1/cs_python','run_micro_sim.py']
unit='qwen38-single-bus-'+args.phase+'-'+args.attempt
cmd=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax=2G','--property=MemorySwapMax=0',
    '--property=CPUAffinity=6 7','--property=TasksMax=128','--property=RuntimeMaxSec=300','--property=TimeoutStopSec=5','--property=LimitCORE=0',
    '--property=StandardOutput=append:'+dest+'/'+args.phase+'.log','--property=StandardError=append:'+dest+'/'+args.phase+'.log',
    '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','flock','-n','/srv/cerebras-workstation/heavy.lock',*command]
subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
receipt=dict(unit=unit,remote=dest,phase=args.phase,physical=False)
(ROOT/'evidence'/name/(args.phase+'-dispatch.json')).write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
