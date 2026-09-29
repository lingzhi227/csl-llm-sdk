"""Bounded overlap/marker test on an exact original-extent bridge cohost."""
import argparse,ast,hashlib,io,json,re,shlex,subprocess,sys,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT/'performance'))
from tools.build_frontend_stage import verified_frozen,encoded


def build(reverse_ut=1,trace=False,resource_compile="layer-backend-compile-043",group=15):
 if resource_compile not in ('layer-backend-compile-042','layer-backend-compile-043','layer-backend-compile-044','layer-backend-compile-045'):raise ValueError('Reviewed resource source required')
 base=ROOT/'performance/evidence'/resource_compile;m=json.loads((base/'source-manifest.json').read_text())['files']
 if not json.loads((base/'COMPLETE.json').read_text())['sram_passed']:raise ValueError('Actual bridge cohost not admitted')
 files={n:verified_frozen(base/'source'/n,h)for n,h in m.items()if n.endswith('.csl')or n in ['check_sram.py','placement.py','elf_inventory.py','source_gate.py']}
 samples=json.loads(verified_frozen(base/'source/profiles.json',m['profiles.json']))['samples'];sample=samples[1 if group==15 else 3]
 assert group in (14,15)and sample['parameters']['bridge_group']==group
 params=sample['parameters'];frames=params['bridge_frames'];workers=params['bridge_workers']
 incoming=params['bridge_input'];back=params['bridge_return_output'];outgoing=params['bridge_output'];returned=params['bridge_return_input']
 original=files[sample['source']]
 if reverse_ut not in (1,6):raise ValueError('Diagnostic reverse microthread')
 actual=original if reverse_ut==1 else original.replace(b'@get_ut_id(1)',b'@get_ut_id(6)')
 if reverse_ut==6 and original.count(b'@get_ut_id(1)')!=3:raise ValueError('Unexpected cohost specialization')
 files[sample['source']]=actual+b'''\nfn probe_arm() void {bridge_arm();}
fn probe_send() void {sys.unblock_cmd_stream();}
fn probe_wait() void {sys.unblock_cmd_stream();}
comptime {@export_symbol(probe_arm);@export_symbol(probe_send);@export_symbol(probe_wait);}
'''
 for n in ['source.csl','sink.csl','run.py']:files[n]=(ROOT/'performance/probes/gdn_bridge'/n).read_bytes()
 if trace:
  for name in [sample['source'],'source.csl','sink.csl']:
   body=files[name].decode();body='const bridge_trace=@import_module("<simprint>",.{.enable=true});\n'+body
   if name==sample['source']:
    body=body.replace('fn bridge_inspect() void {','fn bridge_inspect() void {bridge_trace.print_string("bridge inspect\\n");')
    body=body.replace('bridge_status[7]+=1;','bridge_status[7]+=1;bridge_trace.print_string("bridge drained\\n");')
    body=body.replace('bridge_markers+=1;','bridge_markers+=1;bridge_trace.print_string("bridge marker ");bridge_trace.print_u16_decimal(bridge_markers);bridge_trace.print_string("\\n");')
    body=body.replace('bridge_chunks+=1;','bridge_chunks+=1;bridge_trace.print_string("bridge chunk ");bridge_trace.print_u16_decimal(bridge_chunks);bridge_trace.print_string("\\n");')
    body=body.replace('bridge_sent+=1;','bridge_sent+=1;if(bridge_sent==bridge_frames){bridge_trace.print_string("bridge all frames\\n");}')
   elif name=='source.csl':
    body=body.replace('waiting=false;@block(@get_data_task_id(iq));sys.unblock_cmd_stream();','waiting=false;@block(@get_data_task_id(iq));bridge_trace.print_string("source completed\\n");sys.unblock_cmd_stream();')
    body=body.replace('fn probe_wait() void {','fn probe_wait() void {bridge_trace.print_string("source wait\\n");')
    body=body.replace('task marked(count:u16) void {','task marked(count:u16) void {bridge_trace.print_string("source marker\\n");')
   else:body=body.replace('phase=0;audit[0]=1;','phase=0;bridge_trace.print_string("sink completed\\n");audit[0]=1;')
   files[name]=body.encode()
 files['backend.py']=(ROOT/'runtime/backend.py').read_bytes()
 lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=3,.height=1});','layout {',' @set_rectangle(3,1);',
 f' @set_tile_code(0,0,"source.csl",.{{.memcpy_params=memcpy.get_params(0),.total_frames={frames},.total_workers={workers},.send_color={incoming},.return_color={back}}});']
 fields=','.join('.%s=%s'%(k,v if isinstance(v,str)else str(v).lower())for k,v in sample['parameters'].items())
 lines.append(' @set_tile_code(1,0,"'+sample['source']+'",.{.memcpy_params=memcpy.get_params(1),'+fields+'});')
 lines.append(f' @set_tile_code(2,0,"sink.csl",.{{.memcpy_params=memcpy.get_params(2),.group={group},.total_frames={frames},.total_workers={workers},.input_color={outgoing},.output_color={returned}}});')
 for x,c,rx,tx in [(0,incoming,'RAMP','EAST'),(1,incoming,'WEST','RAMP'),(1,outgoing,'RAMP','EAST'),(2,outgoing,'WEST','RAMP'),(2,returned,'RAMP','WEST'),(1,returned,'EAST','RAMP'),(1,back,'RAMP','WEST'),(0,back,'EAST','RAMP')]:
  extra=',.switches=.{.pos1=.{.rx=RAMP},.pop_mode=.{.pop_on_advance=true}}'if x==2 and c==returned else''
  lines.append(f' @set_color_config({x},0,@get_color({c}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}{extra}}});')
 lines+=[' '+d for d in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode())]
 for n in ('wire','received','probe_audit'):lines.append(f' @export_name("{n}",[*]u32,false);')
 for n in ('probe_arm','probe_send','probe_wait'):lines.append(f' @export_name("{n}",fn()void);')
 files['layout.csl']=('\n'.join(lines+['}'])+'\n').encode()
 files['probe-plan.json']=encoded(dict(application=[3,1],group=group,frames=frames,workers=workers,original_cohost=sample['original_pe'],bank_words=sample['parameters']['bank_words'],
 actual_cohost_source_sha256=hashlib.sha256(original).hexdigest(),probe_cohost_source_sha256=hashlib.sha256(actual).hexdigest(),reverse_ut=reverse_ut,diagnostic_microthread_change=reverse_ut!=1,diagnostic_simprint=trace,resource_compile=resource_compile,
 synthetic_wire=True,bank_values_are_sentinels=True,neural_execution=False,physical=False,
 scope='Exact bank extent and old kernel bodies, three diagnostic RPC wrappers; optional explicitly recorded microthread substitution. Synthetic packet identity and overlap only.'))
 return files


def main():
 p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--reverse-ut',type=int,choices=[1,6],default=1);p.add_argument('--trace',action='store_true');p.add_argument('--group',type=int,choices=[14,15],default=15);p.add_argument('--stream',action='store_true');a=p.parse_args()
 if not re.fullmatch('[0-9]{3}',a.attempt):raise ValueError('Attempt')
 name='gdn-bridge-sim-'+a.attempt;out=ROOT/'performance/evidence'/name
 if out.exists():raise ValueError('Frozen attempt')
 live=subprocess.check_output(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],text=True,timeout=20)
 if live.strip():raise ValueError('Live workstation owner: '+live)
 if a.stream:
  if a.reverse_ut!=1 or a.trace:raise ValueError('Direct stream qualification has no diagnostic substitutions')
  from tools.stage_gdn_stream_sim import build as stream_build
  files=stream_build(a.group)
 else:files=build(a.reverse_ut,a.trace,group=a.group)
 files['stager.py']=Path(__file__).read_bytes()
 script='''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 for phase,cmd,bound in [('compile',['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=10,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],120),('run',['/opt/cerebras/sdk/2.10.1/cs_python','run.py'],180)]:
  with Path(phase+'.log').open('x')as log:
   proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:
    code=proc.wait(timeout=bound)
    if code:raise RuntimeError(phase+' exit '+str(code))
   finally:
    if proc.poll()is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':check(Path.cwd(),application=(3,1),fabric=(10,3))
 verify();result=json.loads(Path('result.json').read_text());assert result['passed']and result['normal_stop']
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,seconds=time.monotonic()-started))+'\\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\\n');raise
'''
 ast.parse(script);files['execute.py']=script.encode();files['source-manifest.json']=encoded(dict(files={n:hashlib.sha256(b).hexdigest()for n,b in files.items()}))
 payload=io.BytesIO()
 with tarfile.open(fileobj=payload,mode='w')as archive:
  for n,b in files.items():
   item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
 remote='/srv/model-storage/qwen38-singlewse/runs/'+name;unit='qwen38-single-'+name
 subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
 cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=2G','--property=MemorySwapMax=0','--property=TasksMax=64','--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=330','--property=TimeoutStopSec=5','--property=KillMode=control-group','--property=LimitFSIZE=67108864','--property=LimitCORE=0','--working-directory='+remote,'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1','/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'dispatch.json').write_bytes(encoded(dict(remote=remote,unit=unit,physical=False)))
 subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20);print(json.dumps(dict(attempt=name,unit=unit)))

if __name__=='__main__':main()
