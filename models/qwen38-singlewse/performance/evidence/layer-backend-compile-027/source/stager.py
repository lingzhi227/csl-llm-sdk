"""Bounded actual-cohost frontend compiler calibration, never a neural benchmark.

Retain the selected consumer's complete P40 bank. Other cohosts explicitly use
at most4096 calibration words so ELF code/queue costs can be measured before a
new whole-stage auxiliary-page reservation. No physical allocation or simulator.
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
from spatial.mixer_frontend_composition import compose_frontend


def build(native=False,compact=False):
    source = ROOT/'performance/evidence/layer-mlp-compile-021/source'
    manifest = json.loads((source.parent/'source-manifest.json').read_text())['files']
    names = [p.name for p in source.glob('*.csl')]+['source_gate.py', 'check_sram.py', 'elf_inventory.py', 'placement.py']
    files = {n: (source/n).read_bytes() for n in names}
    bound = ['profiles.json', 'joint-stage.json', 'mixer-setups.json']
    for n in names+bound:
        if hashlib.sha256((source/n).read_bytes()).hexdigest() != manifest[n]:
            raise ValueError('Frozen P40 source identity changed')
    profiles = [dict(pe=pe, source=c['source'], parameters=deepcopy(c['parameters']))
                for c in json.loads((source/'profiles.json').read_text())['profiles'] for pe in c['pes']]
    region = next(r for r in json.loads((source/'joint-stage.json').read_text())['regions'] if r['role']=='mix')
    setups = json.loads((source/'mixer-setups.json').read_text())
    if compact:
        if not native:raise ValueError('Compact calibration requires native frames')
        from spatial.compact_mixer import compact_csl
        compact_csl(files)
        files['compact_mixer.py']=(ROOT/'performance/spatial/compact_mixer.py').read_bytes()
    compose_frontend(files, profiles, region, setups, native=native)
    selected = {}
    for p in profiles:
        if not p['source'].startswith('frontend_'): continue
        # Compile each actual source/queue-pattern combination, both vertical
        # parities, and both native reduction rank parities. Group arithmetic
        # is checked for every one of the16 original consumer groups.
        key = (p['source'], *(p['parameters'].get(k) for k in ('frontend_horizontal0', 'frontend_horizontal1',
               'frontend_horizontal2', 'frontend_vertical', 'frontend_incoming', 'frontend_source_color', 'mixer_rank_parity', 'frontend_group')))
        selected.setdefault(key, p)
    layout = [f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={len(selected)},.height=1}});',
              'layout {', f' @set_rectangle({len(selected)},1);']
    samples = []
    for x, p in enumerate(selected.values()):
        params = dict(p['parameters']); actual = params['bank_words']
        if 'frontend_group' not in params: params['bank_words'] = min(actual, 4096)
        fields = ','.join('.%s=%s'%(k, v if isinstance(v,str) else str(v).lower()) for k,v in params.items())
        layout.append(f' @set_tile_code({x},0,"{p["source"]}",.{{.memcpy_params=memcpy.get_params({x}),{fields}}});')
        samples.append(dict(pe=[x,0], original_pe=p['pe'], source=p['source'], parameters=params,
                            original_bank_words=actual, payload_changed=params['bank_words']!=actual))
    files['layout.csl'] = ('\n'.join(layout+['}'])+'\n').encode()
    files['profiles.json'] = (json.dumps(dict(samples=samples, application=[len(selected),1],
        scope='Actual original cohosts with automatic root release, packet merges and16-head-group frontend implementation; selected calibration banks, no network execution.',
        base='layer-mlp-compile-021', base_sources={n:manifest[n] for n in names+bound},
        physical=False, executed=False, full_bank_admission=False, full_stage_admission=False),indent=2)+'\n').encode()
    for name in ('mixer_frontend_composition','mixer_frontend_network'):
        files[name+'.py'] = (ROOT/'performance/spatial'/(name+'.py')).read_bytes()
    return files, samples


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');p.add_argument('--native',action='store_true');p.add_argument('--compact',action='store_true');a=p.parse_args()
    if not re.fullmatch('[0-9]{3}',a.attempt): raise ValueError('Attempt')
    name='layer-backend-compile-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Live workstation owner: '+active.stdout)
    files,samples=build(native=a.native,compact=a.compact);width=len(samples)
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
