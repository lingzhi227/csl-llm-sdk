import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 for phase,cmd,bound in [('compile',['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=12,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],120),('run',['/opt/cerebras/sdk/2.10.1/cs_python','run.py'],180)]:
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
  if phase=='compile':check(Path.cwd(),application=(5,1),fabric=(12,3))
 verify();result=json.loads(Path('result.json').read_text());assert result['passed']and result['normal_stop']
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,seconds=time.monotonic()-started))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
