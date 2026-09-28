import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
verify();started=time.monotonic()
try:
 with Path('prepare.log').open('x') as log:
  process=subprocess.Popen(['/usr/bin/python3','prepare.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   result=process.wait(timeout=300)
   if result:raise RuntimeError('Bank relocation exit '+str(result))
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,seconds=time.monotonic()-started,proof=json.loads(Path('bank-remap-proof.json').read_text())))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
