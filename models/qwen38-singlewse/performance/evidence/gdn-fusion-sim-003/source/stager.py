"""Bounded connected frontend/recurrence execution using original value slices."""
import argparse, ast, hashlib, io, json, shlex, subprocess, sys, tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'performance')]
from spatial.gdn_fusion import frontend, switch_worker, replace_one
from tools.stage_layer_mlp_compile import check_host_memory

EXPORTS={'u16':['weights','history','gain','parameters','output','core'],
         'u32':['wire','packet','audit','status','state']}
FUNCTIONS=['send','inspect','clear','arm','await_done']


def build(smoke=False, reference_root='/srv/model-storage/qwen38-singlewse/runs/mixer-frontend-sim-007'):
    frozen=ROOT/'performance/evidence/layer-mlp-compile-027'
    manifest=json.loads((frozen/'source-manifest.json').read_text())['files'];files={}
    for name in ['gdn_columns.csl','qwen_math.csl','mixer_frontend_native.csl','gdn-bank-placement.json']:
        raw=(frozen/'source'/name).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=manifest[name]:raise ValueError('P45 source identity')
        files[name]=raw
    workers=json.loads(files.pop('gdn-bank-placement.json'))['workers']
    files['mixer_frontend_native.csl'],proof=frontend(files['mixer_frontend_native.csl'])
    groups=[0] if smoke else list(range(16));width=len(groups)
    selected=[]
    for x,group in enumerate(groups):
        local=sorted((w for w in workers if w['head']//3==group),key=lambda w:(w['head'],w['first']))
        for i,w in enumerate(local):selected.append(dict(w,diagnostic_pe=[x,i+2]))
    height=max(w['diagnostic_pe'][1] for w in selected)+1
    for n in ['consumer.csl','worker.csl','prepare.py','run.py']:files[n]=(ROOT/'performance/probes/gdn_fusion'/n).read_bytes()
    port=(ROOT/'performance/runtime/mixer_frontend_native_port.cslpart').read_bytes()
    port=replace_one(port,b'@get_input_queue(1)',b'@get_input_queue(2)')
    port=replace_one(port,b'task frontend_packet_ready() void {frontend_snapshot();}\ntask frontend_output_ready() void {frontend_snapshot();}',b'')
    port=replace_one(port,b'frontend_deliver();frontend_cursor=0;',b'frontend_deliver();frontend_cursor=0;progress();')
    files['consumer.csl']=replace_one(files['consumer.csl'],b'FRONTEND_NATIVE_PORT',port)
    worker=switch_worker().replace(b'GDN_WORKSPACE',b'@ptrcast([*]f32,&decoder)').replace(b'GDN_VALUES',b'@ptrcast([*]f32,&values)').replace(b'GDN_COHOST_IDLE',b'')
    files['worker.csl']=replace_one(files['worker.csl'],b'SWITCH_GDN_PORT',worker)
    # Keep the established projected-frame transmitter; exports below unify
    # only the standalone harness interface.
    source=(ROOT/'performance/probes/mixer_frontend/source.csl').read_bytes()
    source=source[:source.index(b'comptime {')]+b'''fn arm() void {sys.unblock_cmd_stream();}
fn await_done() void {sys.unblock_cmd_stream();}
comptime {
 @initialize_queue(oq,.{.color=@get_color(3)});@bind_local_task(sent,complete);
'''
    for ty,names in EXPORTS.items():
        for name in names:source+=f' @export_symbol({"hp" if ty=="u16" else "ap" if name=="audit" else "wp"},"{name}");\n'.encode()
    source+=(''.join(f' @export_symbol({n});\n' for n in FUNCTIONS+['begin'])+'}\n').encode();files['source.csl']=source
    empty=b'param memcpy_params;const sys=@import_module("<memcpy/memcpy>",memcpy_params);var zero=@zeros([1]u32);var short=@zeros([1]u16);const p:[*]u32=&zero;const h:[*]u16=&short;\n'
    empty+=(''.join(f'fn {n}() void {{sys.unblock_cmd_stream();}}\n' for n in FUNCTIONS)+'fn begin(token:u32) void {sys.unblock_cmd_stream();}\ncomptime {\n').encode()
    for ty,names in EXPORTS.items():
        for name in names:empty+=f' @export_symbol({"h" if ty=="u16" else "p"},"{name}");\n'.encode()
    empty+=(''.join(f' @export_symbol({n});\n' for n in FUNCTIONS+['begin'])+'}\n').encode();files['empty.csl']=empty
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});','layout {',f' @set_rectangle({width},{height});']
    def route(x,y,c,rx,tx,extra=''):
        layout.append(f' @set_color_config({x},{y},@get_color({c}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}{extra}}});')
    for x,group in enumerate(groups):
        local=[w for w in selected if w['head']//3==group];by_y={w['diagnostic_pe'][1]:w for w in local}
        layout.append(f' @set_tile_code({x},0,"source.csl",.{{.memcpy_params=memcpy.get_params({x})}});')
        layout.append(f' @set_tile_code({x},1,"consumer.csl",.{{.memcpy_params=memcpy.get_params({x}),.frontend_group={group},.frontend_projected_packets=518,.worker_count={len(local)},.delay_last_release=true}});')
        route(x,0,3,'RAMP','SOUTH');route(x,1,3,'NORTH','RAMP');route(x,1,5,'RAMP','SOUTH');route(x,1,4,'SOUTH','RAMP')
        for y in range(2,height):
            if y not in by_y:
                layout.append(f' @set_tile_code({x},{y},"empty.csl",.{{.memcpy_params=memcpy.get_params({x})}});');continue
            w=by_y[y];h=w['head']%3
            layout.append(f' @set_tile_code({x},{y},"worker.csl",.{{.memcpy_params=memcpy.get_params({x}),.gdn_columns={w["columns"]},.gdn_first={w["first"]},.gdn_head={w["head"]},.gdn_state_word=0,.gdn_input_color=5,.gdn_output_color=4,.gdn_input_queue=2}});')
            counter=(1161-387*h)%1161
            route(x,y,5,'NORTH','RAMP'+(',SOUTH' if y<max(by_y) else ''),f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter={counter},.limit1=1160,.max_counter=386}}')
            route(x,y,4,'RAMP','NORTH',',.switches=.{.pos1=.{.rx=SOUTH},.pop_mode=.{.always_pop=true}}')
    for ty,names in EXPORTS.items():
        for name in names:layout.append(f' @export_name("{name}",[*]{ty},false);')
    layout+= [f' @export_name("{n}",fn()void);' for n in FUNCTIONS]+[' @export_name("begin",fn(u32)void);','}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    plan=dict(application=[width,height],groups=groups,workers=selected,frontend_proof=proof,source_manifest_sha256=hashlib.sha256((frozen/'source-manifest.json').read_bytes()).hexdigest(),
              actual_frontend_execution=True,actual_gdn_execution=True,host_injected_recurrent_results=False,
              original_state_values=sum(128*w['columns'] for w in selected),full_original_value_coverage=not smoke,
              full_bank_admission=False,physical=False,delayed_final_source_callback=True,
              scope='Connected frontend/recurrence component with original slices; diagnostic placement and terminal output observer. No output projection, complete layer or model-speed admission.')
    files['probe-plan.json']=(json.dumps(plan,indent=2)+'\n').encode()
    fixture=ROOT/'performance/evidence/mixer-frontend-sim-007'
    files['reference-input.json']=(json.dumps(dict(root=reference_root,fixture_sha256=json.loads((fixture/'fixture.json').read_text())['fixture_sha256']))+'\n').encode()
    files['reference/__init__.py']=b''
    for n in ['frontend_oracle.py','gdn_columns_oracle.py']:files['reference/'+n]=(ROOT/'performance/reference'/n).read_bytes()
    for n in ['backend.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
    for n in ['check_sram.py','placement.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
    return files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--smoke',action='store_true');parser.add_argument('--compile-only',action='store_true');args=parser.parse_args()
    if len(args.attempt)!=3 or not args.attempt.isdigit():raise ValueError('Attempt')
    name='gdn-fusion-sim-'+args.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    live=subprocess.check_output(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],text=True,timeout=20)
    if live.strip():raise ValueError('Live workstation owner: '+live)
    probe="from pathlib import Path;print(next(int(x.split()[1])*1024 for x in Path('/proc/meminfo').read_text().splitlines() if x.startswith('MemAvailable:')))"
    admission=check_host_memory(int(subprocess.check_output(['ssh','workstation','python3 -c '+shlex.quote(probe)],text=True,timeout=20)),16)
    reference=json.loads((ROOT/'performance/evidence/mixer-frontend-sim-007/dispatch.json').read_text())['remote']
    files=build(args.smoke,reference);plan=json.loads(files['probe-plan.json']);w,h=plan['application'];fabric=(w+7,h+2)
    commands=[('prepare',['/usr/bin/python3','prepare.py'],120),('compile',['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3',f'--fabric-dims={fabric[0]},{fabric[1]}','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],1800)]
    if not args.compile_only:commands.append(('run',['/opt/cerebras/sdk/2.10.1/cs_python','run.py','--smoke'] if args.smoke else ['/opt/cerebras/sdk/2.10.1/cs_python','run.py'],600))
    execution='''import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
commands=COMMANDS
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
  if phase=='compile':check(Path.cwd(),application=APPLICATION,fabric=FABRIC)
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,seconds=time.monotonic()-started,compile_only=COMPILE_ONLY))+'\\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\\n');raise
'''
    files['execute.py']=execution.replace('COMMANDS',repr(commands)).replace('APPLICATION',repr((w,h))).replace('FABRIC',repr(fabric)).replace('COMPILE_ONLY',repr(args.compile_only)).encode()
    files['stager.py']=Path(__file__).read_bytes();files['fusion_composition.py']=(ROOT/'performance/spatial/gdn_fusion.py').read_bytes()
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
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20);print(json.dumps(dict(name=name,application=[w,h],physical=False)))


if __name__=='__main__':main()
