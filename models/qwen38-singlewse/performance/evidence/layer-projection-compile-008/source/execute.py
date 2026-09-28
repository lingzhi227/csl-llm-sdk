import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic()
try:
 with Path('compile.log').open('x') as log:
  proc=subprocess.Popen(['/opt/cerebras/sdk/2.10.1/cslc','layout.csl','--arch=wse3','--fabric-dims=24,3','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=proc.wait(timeout=180)
   if code:raise RuntimeError('Compiler exit '+str(code))
  finally:
   if proc.poll() is None:
    os.killpg(proc.pid,signal.SIGTERM)
    try:proc.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
 result=check(Path.cwd(),application=(17,1),fabric=(24,3));verify()
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,executed=False,application=[17,1],sram_passed=result['passed'],seconds=time.monotonic()-started))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
