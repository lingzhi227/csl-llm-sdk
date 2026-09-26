"""Freeze a fresh bounded full MoE-layer compilation on the Mass1 workstation."""
import argparse, hashlib, io, json, re, shlex, subprocess, tarfile
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
BASE = '/srv/model-storage/gpt-oss20b/runs'

def main():
    p = argparse.ArgumentParser(); p.add_argument('attempt');p.add_argument('--kind',choices=('moe','attention','full','roles','decoder'),default='moe'); a = p.parse_args()
    if not re.fullmatch(r'[0-9]{3}', a.attempt): raise ValueError('Fresh numbered attempt required')
    stem,folder,width={'moe':('moe-layer','moe_layer',27),'attention':('attention-prefix','attention_prefix',4),'full':('full-model','full_model',750),'roles':('resident-roles','resident_roles',180),'decoder':('decoder-layer','decoder_layer',27)}[a.kind]
    height=1 if a.kind=='roles' else 1160
    dest = BASE + '/' + stem + '-compile-' + a.attempt
    files = {f.name: f.read_bytes() for f in (ROOT/'csl/resident').glob('*.csl')}
    files.update({f.name: f.read_bytes() for f in (ROOT/'csl').glob('*.csl')})
    files.update({f.name:f.read_bytes() for f in (ROOT/'experiments'/folder).glob('*.csl')})
    files['experiment.json']=(json.dumps(dict(shared_programs=True,physical_application_pes=width*height,full_moe_layer=a.kind in ('moe','decoder'),full_attention=a.kind in ('attention','decoder'),full_model=a.kind=='full',full_decoder_layer=a.kind=='decoder'))+'\n').encode()
    files.update({n: (ROOT/'tools'/n).read_bytes() for n in ('check_sram.py','elf_inventory.py')})
    files['source-manifest.json'] = (json.dumps({'files': {n:hashlib.sha256(v).hexdigest() for n,v in files.items()}},indent=2)+'\n').encode()
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w') as archive:
        for name,data in files.items():
            entry=tarfile.TarInfo(name);entry.size=len(data);archive.addfile(entry,io.BytesIO(data))
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(dest)+' && tar -xf - -C '+shlex.quote(dest)],input=stream.getvalue(),check=True,timeout=30,cwd=ROOT)
    unit='gptoss20b-'+stem+'-compile-'+a.attempt
    launch=['systemd-run','--user','--unit='+unit,'--property=WorkingDirectory='+dest,
      '--property=MemoryMax='+('3G' if a.kind=='full' else '2G' if a.kind in ('roles','decoder') else '8G'),'--property=MemorySwapMax=0','--property=CPUAffinity='+('2 3' if a.kind in ('full','roles','decoder') else '0 1 2 3'),
      '--property=TasksMax=256','--property=RuntimeMaxSec='+('1800' if a.kind=='full' else '600'),'--property=TimeoutStopSec=5',
      '--property=LimitCORE=0','--property=StandardOutput=append:'+dest+'/compile.log',
      '--property=StandardError=append:'+dest+'/compile.log',
      '/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3',
      '--fabric-dims='+str(width+7)+','+str(height+2),'--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism='+('2' if a.kind=='full' else '1' if a.kind in ('roles','decoder') else '4'),'-o','out']
    subprocess.run(['ssh','workstation',shlex.join(launch)],check=True,timeout=30,cwd=ROOT)
    print(json.dumps(dict(remote=dest,unit=unit,physical_pes=width*height,shared_programs=True)))

if __name__=='__main__': main()
