import subprocess,json,time
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=10,6','--fabric-offsets=4,1','--memcpy','--channels=1','--max-parallelism=1','-o','out'],[sdk+'cs_python','run.py']]
try:
 for phase,cmd in zip(['compile','run'],commands):
  with Path(phase+'.log').open('x') as out:subprocess.run(cmd,stdout=out,stderr=subprocess.STDOUT,check=True,timeout=240)
  if phase=='compile':assert check(Path.cwd())['application_pes']==12
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
