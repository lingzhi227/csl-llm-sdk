"""Freeze source, compact observations and release of a completed owned run."""
import argparse,base64,hashlib,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def collect(attempt):
 if not re.fullmatch(r'\d{3}',attempt):raise ValueError('Attempt')
 out=ROOT/'evidence'/('gdn-fusion-sim-'+attempt);dispatch=json.loads((out/'dispatch.json').read_text())
 remote=dispatch['remote'];unit=dispatch['unit']+'.service'
 script='root='+repr(remote)+'\nunit='+repr(unit)+'\n'+'''import base64,datetime,hashlib,json,os,subprocess
from pathlib import Path
r=Path(root);manifest=json.loads((r/'source-manifest.json').read_text())
fields=['ActiveState','SubState','Result','ExecMainStatus','MemoryPeak','CPUUsageNSec','ControlGroup','MainPID']
show=subprocess.check_output(['systemctl','--user','show',unit]+['--property='+n for n in fields],text=True)
release=dict(line.split('=',1) for line in show.splitlines() if '=' in line)
if release['ActiveState'] not in ('inactive','failed') or release['MainPID']!='0':raise ValueError('Owner still active')
matching=[];members=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  cwd=str((p/'cwd').resolve(strict=True));groups=(p/'cgroup').read_text()
  if cwd==root or cwd.startswith(root+'/') or '/'+unit in groups:matching.append(int(p.name))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
if release.get('ControlGroup'):
 c=Path('/sys/fs/cgroup')/release['ControlGroup'].lstrip('/')
 if c.exists():
  for p in c.rglob('cgroup.procs'):members.extend(p.read_text().split())
if matching or members:raise ValueError('Invocation still has processes')
release.update(matching_processes=matching,cgroup_members=members)
if not ((r/'COMPLETE.json').exists() or (r/'FAILURE.json').exists() or (r/'CANCELLED.json').exists()):raise ValueError('No terminal receipt')
release.update(unit=unit,released=True,observed_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
source={n:base64.b64encode((r/n).read_bytes()).decode() for n in manifest['files']}
names=['COMPLETE.json','FAILURE.json','CANCELLED.json','result.json','sram.json','fixture.json','compile.log','run.log','prepare.log','sim.log']
receipts={};log_metadata={}
for n in names:
 if not (r/n).is_file():continue
 raw=(r/n).read_bytes()
 try:receipts[n]=raw.decode('utf-8')
 except UnicodeDecodeError:
  if not n.endswith('.log'):raise
  text=raw[:65536].decode('utf-8',errors='replace')
  receipts[n+'.sanitized.txt']=''.join(c if c.isprintable() or c in '\\n\\t' else '\ufffd' for c in text)
  log_metadata[n]=dict(original_bytes=len(raw),original_sha256=hashlib.sha256(raw).hexdigest(),original_retained_remote=True,export='First65536bytes, UTF-8 replacement and nonprinting-character replacement')
if log_metadata:receipts['log-collection.json']=json.dumps(log_metadata,indent=2)+'\\n'
if sum(map(len,source.values()))+sum(map(len,receipts.values()))>8388608:raise ValueError('Compact collection bound')
print(json.dumps(dict(source=source,receipts=receipts,release=release)))
'''
 data=json.loads(subprocess.check_output(['ssh','workstation','python3 -c '+shlex.quote(script)],text=True,timeout=30))
 manifest=json.loads((out/'source-manifest.json').read_text())['files']
 def freeze(path,raw):
  path.parent.mkdir(parents=True,exist_ok=True)
  if path.exists():
   if path.read_bytes()!=raw:raise ValueError('Frozen file differs: '+str(path))
  else:path.write_bytes(raw)
 for n,s in data['source'].items():
  raw=base64.b64decode(s)
  if hashlib.sha256(raw).hexdigest()!=manifest[n]:raise ValueError('Source digest')
  freeze(out/'source'/n,raw)
 for n,s in data['receipts'].items():freeze(out/n,s.encode())
 if not (out/'release.json').exists():freeze(out/'release.json',(json.dumps(data['release'],indent=2)+'\n').encode())
 print(json.dumps(dict(attempt=attempt,receipts=list(data['receipts']),released=True)))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('attempt',nargs='+');a=p.parse_args()
 for attempt in a.attempt:collect(attempt)
