"""Freeze an expert simulator attempt on Mass1; no payloads/binaries on Mac."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
BASE='/srv/model-storage/gpt-oss20b'

def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--run',action='store_true')
    p.add_argument('--full',action='store_true');a=p.parse_args()
    if not re.fullmatch(r'[0-9]{3}',a.attempt):raise ValueError('Numbered fresh attempt required')
    size='full' if a.full else 'mini';stem=f'expert-{size}-sim-{a.attempt}';dest=BASE+'/runs/'+stem
    gr,dr=(36,18) if a.full else (2,1)
    if a.run:
        # The full geometry is compile-audited, not admitted to expensive simulation.
        if a.full:raise ValueError('Full expert simulation has no bounded execution profile yet')
        cmd=['/opt/cerebras/sdk/2.10.1/cs_python','run_micro_sim.py'];seconds=960;phase='run'
    else:
        files={n:(ROOT/'experiments/expert'/n).read_bytes() for n in ('layout.csl','pe.csl','run.py')}
        files.update({n:(ROOT/'csl'/n).read_bytes() for n in ('mxfp4_gemv.csl','moe_math.csl')})
        files.update({n:(ROOT/'tools'/n).read_bytes() for n in ('run_micro_sim.py','check_sram.py','elf_inventory.py')})
        files['backend.py']=(ROOT/'runtime/backend.py').read_bytes()
        files['experiment.json']=(json.dumps(dict(application_pes=23*gr,full_expert=a.full,
                                  compiler_params=f'gate_rows:{gr},down_rows:{dr}'))+'\n').encode()
        files['source-manifest.json']=(json.dumps({'files':{n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
        stream=io.BytesIO()
        with tarfile.open(fileobj=stream,mode='w') as archive:
            for name,data in files.items():
                entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
        subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True)
        fixture=BASE+'/fixtures/expert-'+size+'-001'
        subprocess.run(['ssh','workstation','cp '+fixture+'/expert-fixture.npz '+fixture+'/expert-fixture.json '+shlex.quote(dest)+'/'],check=True)
        cmd=['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3',f'--fabric-dims=30,{gr+2}',
             '--fabric-offsets=4,1',f'--params=gate_rows:{gr},down_rows:{dr}','--memcpy','--channels=1','--max-parallelism=2','-o','out']
        seconds=600 if a.full else 180;phase='compile'
    unit=f'gptoss20b-expert-{size}-{phase}-{a.attempt}'
    launch=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,'--property=MemoryMax=3G',
            '--property=MemorySwapMax=0','--property=CPUAffinity='+('0 1' if a.full else '2 3'),'--property=TasksMax=128',
            '--property=RuntimeMaxSec='+str(seconds),'--property=TimeoutStopSec=5','--property=LimitCORE=0',
            '--property=StandardOutput=append:'+dest+'/'+phase+'.log','--property=StandardError=append:'+dest+'/'+phase+'.log',
            '--setenv=OPENBLAS_NUM_THREADS=1',*cmd]
    subprocess.run(['ssh','workstation',shlex.join(launch)],check=True)
    print(json.dumps(dict(remote=dest,phase=phase,seconds=seconds,unit=unit)))

if __name__=='__main__':main()
