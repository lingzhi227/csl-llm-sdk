"""Bounded all-head original-reference recurrence through actual paired ports."""
import argparse,ast,hashlib,io,json,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'performance')]
from spatial.gdn_columns import paired_port
from tools.stage_layer_mlp_compile import check_host_memory

def build():
 frozen=ROOT/'performance/evidence/layer-mlp-compile-027';manifest=json.loads((frozen/'source-manifest.json').read_text())['files']
 files={}
 for name in ['gdn_columns.csl','qwen_math.csl','gdn-bank-placement.json']:
  raw=(frozen/'source'/name).read_bytes()
  if hashlib.sha256(raw).hexdigest()!=manifest[name]:raise ValueError('P45 source identity')
  files[name]=raw
 workers=json.loads(files.pop('gdn-bank-placement.json'))['workers'];width,height=251,6
 if len(workers)!=753 or sum(w['columns'] for w in workers)!=6144:raise ValueError('All original P45 value slices required')
 for n in ['source.csl','worker.csl','prepare.py','run.py']:files[n]=(ROOT/'performance/probes/gdn_columns'/n).read_bytes()
 port=paired_port();adapted=port.replace(b'GDN_WORKSPACE',b'@ptrcast([*]f32,&decoder)').replace(b'GDN_VALUES',b'@ptrcast([*]f32,&values)').replace(b'GDN_COHOST_IDLE',b'')
 files['worker.csl']=files['worker.csl'].replace(b'PAIRED_GDN_PORT',adapted)
 files['reference/__init__.py']=b''
 for n in ['frontend_oracle.py','gdn_columns_oracle.py']:files['reference/'+n]=(ROOT/'performance/reference'/n).read_bytes()
 for n in ['backend.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
 for n in ['check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
 fixture=ROOT/'performance/evidence/mixer-frontend-sim-007'
 files['reference-input.json']=(json.dumps(dict(root=json.loads((fixture/'dispatch.json').read_text())['remote'],fixture_sha256=json.loads((fixture/'fixture.json').read_text())['fixture_sha256']))+'\n').encode()
 layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});','layout {',f' @set_rectangle({width},{height});']
 for i,w in enumerate(workers):
  x,y=i%width,2*(i//width)
  layout.append(f' @set_tile_code({x},{y},"source.csl",.{{.memcpy_params=memcpy.get_params({x}),.columns={w["columns"]}}});')
  layout.append(f' @set_tile_code({x},{y+1},"worker.csl",.{{.memcpy_params=memcpy.get_params({x}),.gdn_columns={w["columns"]},.gdn_first={w["first"]},.gdn_head={w["head"]},.gdn_state_word=0,.gdn_input_color=3,.gdn_output_color=4,.gdn_input_queue=2}});')
  for row,color,rx,tx in [(y,3,'RAMP','SOUTH'),(y+1,3,'NORTH','RAMP'),(y+1,4,'RAMP','NORTH'),(y,4,'SOUTH','RAMP')]:
   layout.append(f' @set_color_config({x},{row},@get_color({color}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}}});')
 for n in ['state','status','wire','frames']:layout.append(f' @export_name("{n}",[*]u32,false);')
 for n in ['begin','clear','inspect','exchange']:layout.append(f' @export_name("{n}",fn()void);')
 files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
 files['probe-plan.json']=(json.dumps(dict(application=[width,height],workers=workers,original_placement='layer-mlp-compile-027',original_manifest_sha256=hashlib.sha256((frozen/'source-manifest.json').read_bytes()).hexdigest(),paired_port_sha256=hashlib.sha256(port).hexdigest(),arithmetic_sha256=manifest['gdn_columns.csl'],
  diagnostic_bank_words=4096,full_bank_admission=False,actual_frontend_execution=False,physical=False),indent=2)+'\n').encode()
 return files

def main():
 parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
 if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
 name='gdn-columns-sim-'+args.attempt;out=ROOT/'performance/evidence'/name
 if out.exists():raise ValueError('Frozen attempt')
 live=subprocess.check_output(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],text=True,timeout=20)
 if live.strip():raise ValueError('Live workstation owner: '+live)
 probe="from pathlib import Path;print(next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))"
 admission=check_host_memory(int(subprocess.check_output(['ssh','workstation','python3 -c '+shlex.quote(probe)],text=True,timeout=20)),16)
 files=build()
 files['execute.py']=b'''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[('prepare',['/usr/bin/python3','prepare.py'],120),('compile',[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=258,8','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],1800),('run',[sdk+'cs_python','run.py','--smoke'],600)]
try:
 for phase,cmd,seconds in commands:
  with Path(phase+'.log').open('x') as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+seconds
   try:
    while proc.poll() is None:
     if time.monotonic()>deadline:raise TimeoutError(phase+' deadline')
     if phase=='run' and Path('sim.log').exists():
      if Path('sim.log').stat().st_size>8388608:raise RuntimeError('Simulator log bound')
      if 'FATAL:' in Path('sim.log').read_text():raise RuntimeError('Simulator fatal')
     time.sleep(.2)
    if proc.returncode:raise RuntimeError(phase+' exit '+str(proc.returncode))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':check(Path.cwd(),application=(251,6),fabric=(258,8))
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\\n');raise
'''
 files['stager.py']=Path(__file__).read_bytes();files['gdn_columns_composition.py']=(ROOT/'performance/spatial/gdn_columns.py').read_bytes()
 for n,b in files.items():
  if n.endswith('.py'):ast.parse(b)
 files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
 archive=io.BytesIO()
 with tarfile.open(fileobj=archive,mode='w') as tar:
  for n,b in files.items():item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
 remote='/srv/model-storage/qwen38-singlewse/runs/'+name
 subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=archive.getvalue(),check=True,timeout=30)
 unit='qwen38-single-'+name
 cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=16G','--property=MemorySwapMax=0','--property=TasksMax=128','--property=CPUQuota=200%','--property=AllowedCPUs=6,7','--property=RuntimeMaxSec=2550','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=536870912','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6,7','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False,resource_admission=admission),indent=2)+'\n')
 subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20);print(json.dumps(dict(name=name,workers=753,heads=48,physical=False)))

if __name__=='__main__':main()
