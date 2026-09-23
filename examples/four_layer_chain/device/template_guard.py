"""Check the generated platform container before requesting CSL compilation.

SdkCompiler.__enter__ creates/connects the owned job; compile() uploads source
and issues sdk_compile later. A single bounded metadata export is supervised
inside the same owned client process group and total compile phase deadline.
"""
from pathlib import Path
import getpass,hashlib,json,os,resource,subprocess,sys,time,zipfile
from source_gate import ROOT
from hw00.store import atomic_json

def memory_bytes(value):
 value=str(value)
 for suffix,factor in [('Ki',1<<10),('Mi',1<<20),('Gi',1<<30),('Ti',1<<40)]:
  if value.endswith(suffix):return int(value[:-len(suffix)])*factor
 return int(value)

def verify_template():
 ids=(ROOT/'compile.jobs').read_text().splitlines()
 ids={j for j in ids if j.startswith('wsjob-')}
 if len(ids)!=1:raise ValueError('One invocation-correlated compile job')
 jid=ids.pop();out=ROOT/'template-inspection';out.mkdir(exist_ok=False)
 def query(*args):
  p=subprocess.run(['/usr/local/bin/csctl','get',*args,'-o','json'],capture_output=True,text=True,check=True,timeout=10)
  if len(p.stdout)>1<<20:raise ValueError('Resource metadata bound')
  return json.loads(p.stdout)
 job=query('job',jid)
 if job['spec']['user']['username']!=getpass.getuser():raise ValueError('Owned job required')
 command=['/usr/local/bin/csctl','log-export',jid,'--binaries=false','--compile-artifacts=false','--tasks','coordinator-0','--timeout','30','--target-type','local','--path',str(out)]
 wrapper='import os,resource,sys;resource.setrlimit(resource.RLIMIT_FSIZE,(8<<20,8<<20));resource.setrlimit(resource.RLIMIT_CPU,(10,10));os.execv(sys.argv[1],sys.argv[1:])'
 atomic_json(out/'REQUEST.json',dict(command=command,metadata_only=True,CSL_compile_invocations=0,num_lines_default=0,reason='Server rejects num-lines with skipClusterLogs'))
 def inventory():
  files=[p for p in out.rglob('*') if p.is_file()]
  if any(p.is_symlink() for p in out.rglob('*')) or len(files)>128 or sum(p.stat().st_size for p in files)>16<<20 or any(p.stat().st_size>8<<20 for p in files):raise ValueError('Bounded template inspection directory')
  if (out/'export.log').stat().st_size>1<<20:raise ValueError('Template console limit')
 started=time.monotonic();proc=None
 try:
  with (out/'export.log').open('xb') as log:
   proc=subprocess.Popen([sys.executable,'-B','-c',wrapper,*command],stdout=log,stderr=subprocess.STDOUT)
   while proc.poll() is None:
    if time.monotonic()-started>40:raise TimeoutError('Template export40s')
    inventory();time.sleep(.1)
   inventory()
   if proc.returncode:raise RuntimeError('Template export exit '+str(proc.returncode))
 finally:
  if proc is not None:
   if proc.poll() is None:
    proc.terminate()
    try:proc.wait(timeout=3)
    except subprocess.TimeoutExpired:proc.kill();proc.wait(timeout=3)
   atomic_json(out/'EXPORT-EXIT.json',dict(child_pid=proc.pid,child_reaped=proc.poll() is not None,exit_code=proc.returncode,seconds=time.monotonic()-started,process_group=os.getpgrp(),same_owned_compile_group=True))
 archives=list(out.glob('*.zip'))
 if len(archives)!=1:raise ValueError('One immutable metadata archive')
 archive=archives[0]
 with zipfile.ZipFile(archive) as z:
  entries=z.infolist()
  if len(entries)>128 or sum(i.file_size for i in entries)>16<<20:raise ValueError('Template archive inventory')
  candidates=[i for i in entries if i.filename.endswith('/'+jid+'.yaml')]
  if len(candidates)!=1 or candidates[0].file_size>65536:raise ValueError('Exact generated job template')
  raw=z.read(candidates[0])
  try:document=json.loads(raw)
  except json.JSONDecodeError:
   import yaml
   document=yaml.safe_load(raw)
 spec=document['spec'];coordinator=spec['wsReplicaSpecs']['Coordinator']
 containers=coordinator['template']['spec']['containers']
 if spec['numWafers']!=0 or len(containers)!=1 or coordinator['replicas']!=1:raise ValueError('One coordinator and no physical wafer')
 actual=containers[0]['resources'];limits=actual.get('limits',{});requests=actual.get('requests',{})
 maximum=90<<30
 if not 0<memory_bytes(limits['memory'])<=maximum or not 0<memory_bytes(requests['memory'])<=maximum:raise ValueError('Platform memory exceeds authorized90GiB or lacks hard limit')
 if requests.get('cpu')!='500m':raise ValueError('Platform CPU request changed from reviewed500m')
 report=dict(status='actual_template_admitted_before_compile',job_id=jid,num_wafers=0,replicas=1,actual_resources=actual,memory_limit_bytes=memory_bytes(limits['memory']),memory_request_bytes=memory_bytes(requests['memory']),CPU_limit_displayed='cpu' in limits,actual_peak_unknown=True,SDK_requested_memory_bytes=8<<30,SDK_requested_cpu_millicores=1000,actual_template_sha256=hashlib.sha256(raw).hexdigest(),archive=dict(file=str(archive.relative_to(ROOT)),bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest()),metadata_export_seconds=time.monotonic()-started,CSL_compile_invocations=0)
 atomic_json(ROOT/'actual-template.json',report);return report
