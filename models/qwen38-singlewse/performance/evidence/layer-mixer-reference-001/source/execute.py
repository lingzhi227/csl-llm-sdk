import json,os,signal,subprocess,time
from pathlib import Path
from source_gate import verify
verify();started=time.monotonic()
try:
 with Path('prepare.log').open('x') as log:
  process=subprocess.Popen(['/usr/bin/python3','prepare.py'],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  try:
   code=process.wait(timeout=300)
   if code:raise RuntimeError('Reference exit '+str(code))
  finally:
   if process.poll() is None:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=2)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait(timeout=2)
 verify();fixture=json.loads(Path('fixture.json').read_text());assert fixture['passed']
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_device_execution=False,seconds=time.monotonic()-started,fixture=fixture))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
