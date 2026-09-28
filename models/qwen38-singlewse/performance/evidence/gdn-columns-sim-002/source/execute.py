import json,os,signal,subprocess,time
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
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
