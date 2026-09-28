"""Paired actual-cohost compiler costs for full-key FP32 GDN columns.

All samples have4096-word calibration banks. An alternate start entry keeps the
math reachable while leasing existing decoder/partial arrays. This neither
executes a neural graph nor admits full banks or new fabric transport.
"""
import argparse
import ast
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'performance')]


def build():
    source=ROOT/'performance/evidence/layer-mlp-compile-022/source'
    manifest=json.loads((source.parent/'source-manifest.json').read_text())['files']
    names=[p.name for p in source.glob('*.csl')]+['source_gate.py','check_sram.py','elf_inventory.py','placement.py']
    files={n:(source/n).read_bytes() for n in names}
    for n in names+['profiles.json']:
        if hashlib.sha256((source/n).read_bytes()).hexdigest()!=manifest[n]:raise ValueError('Frozen P42 source identity')
    profiles=json.loads((source/'profiles.json').read_text())['profiles'];selected=[]
    for rows in (4,8):
        selected.append(next(p for p in profiles if p['source']=='layer_projection.csl' and p['parameters']['rows']==rows and not p['parameters']['root'] and not p['parameters'].get('fusion_actor',False) and not p['parameters'].get('mlp_sender',False)))
    for root in (False,True):
        selected.append(next(p for p in profiles if p['source']=='device_mixer_layer_mlp_standby.csl' and p['parameters']['mixer_can_root']==root))
    files['gdn_value.csl']=(ROOT/'performance/csl/gdn_value.csl').read_bytes()
    samples=[];width=len(selected)*2
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});','layout {',f' @set_rectangle({width},1);']
    for index,p in enumerate(selected):
        old=p['source'];raw=files[old];marker=b'fn start() void {sys.unblock_cmd_stream();}'
        if raw.count(marker)!=1:raise ValueError('Callable calibration anchor')
        if old=='layer_projection.csl':cache='native.decoded';result='native.partial'
        else:cache='mixer.native.decoded';result='mixer.partial'
        invocation=('const gdn=@import_module("gdn_value.csl");\nfn start() void {\n'
            ' const state=@ptrcast([*]f32,&bank[bank_words-128]);\n'
            ' const vector=@ptrcast([*]f32,&'+cache+');const result:[*]f32=&'+result+';\n'
            ' gdn.key(state,vector,@bitcast(f32,audit[0]),@bitcast(f32,audit[1]),@bitcast(f32,audit[2]),result);\n'
            ' gdn.reduce(state,vector,result);sys.unblock_cmd_stream();\n}').encode()
        changed='gdn_math_'+old;files[changed]=raw.replace(marker,invocation)
        for variant,name in enumerate([old,changed]):
            x=2*index+variant;params=deepcopy(p['parameters']);original=params['bank_words'];params['bank_words']=4096
            fields=','.join('.%s=%s'%(k,v if isinstance(v,str) else str(v).lower()) for k,v in params.items())
            layout.append(f' @set_tile_code({x},0,"{name}",.{{.memcpy_params=memcpy.get_params({x}),{fields}}});')
            samples.append(dict(pe=[x,0],pair=index,variant='math' if variant else 'baseline',source=name,parameters=params,
                original_pe=p['pes'][0],original_bank_words=original,payload_changed=True))
    exported={v.split(',')[-1].strip().strip('"') for v in re.findall(r'@export_symbol\(([^;]+)\);',files['layer_projection.csl'].decode())}
    for declaration in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode()):
        if re.search(r'@export_name\("([^"]+)"',declaration).group(1) in exported:layout.append(' '+declaration)
    files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
    files['profiles.json']=(json.dumps(dict(samples=samples,application=[width,1],base='layer-mlp-compile-022',
        scope='Paired actual-cohost compiler cost of full-key FP32 state columns reusing existing decoder/partial arenas. Calibration start entry only; no new transport, numerical execution or full-bank admission.',
        executed=False,physical=False,full_bank_admission=False,base_source_manifest_sha256=hashlib.sha256((source.parent/'source-manifest.json').read_bytes()).hexdigest()),indent=2)+'\n').encode()
    return files,samples


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
    if not re.fullmatch('[0-9]{3}',a.attempt): raise ValueError('Attempt')
    name='layer-backend-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    files,samples=build();width=len(samples)
    script=r'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=FABRIC_WIDTH,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=120)
   if code:raise RuntimeError('Compiler exit '+str(code))
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=(APPLICATION_WIDTH,1),fabric=(FABRIC_WIDTH,3));verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,full_bank_admission=False,full_stage_admission=False,sram_passed=result['passed'],seconds=time.monotonic()-started))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
'''.replace('APPLICATION_WIDTH',str(width)).replace('FABRIC_WIDTH',str(width+7))
    ast.parse(script);files['execute.py']=script.encode();files['stager.py']=Path(__file__).read_bytes()
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=1G','--property=MemorySwapMax=0','--property=TasksMax=64',
         '--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=140','--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    print(json.dumps(dict(remote=remote,unit=unit,samples=width,physical=False,full_bank_admission=False)))


if __name__=='__main__':main()
