import subprocess,json,time,os,signal
from pathlib import Path
from source_gate import verify
from check_sram import check
verify();started=time.monotonic();sdk='/opt/cerebras/sdk/2.10.1/'
commands=[['prepare',['/usr/bin/python3','prepare.py']],['compile',[sdk+'cslc','layout.csl','--arch=wse3','--fabric-dims=638,4','--fabric-offsets=4,1','--memcpy','--channels=2','--max-parallelism=1','-o','out']],['run',[sdk+'cs_python','run.py']]]
try:
 for phase,cmd in commands:
  with Path(phase+'.log').open('x') as out:
   # 009 proves host submissions return before device initialization. A complete
   # 960-tile upload/readback needs a larger simulator budget than 20-PE probes.
   # Two memcpy channels and BF16-only simulation payload reduce serialized I/O.
   budget=900 if phase=='run' else 240
   proc=subprocess.Popen(cmd,stdout=out,stderr=subprocess.STDOUT,start_new_session=True);deadline=time.monotonic()+budget
   try:
    while proc.poll() is None:
     if time.monotonic()>deadline:raise TimeoutError(phase+' deadline')
     if phase=='run' and Path('sim.log').exists():
      if Path('sim.log').stat().st_size>8388608:raise RuntimeError('sim log budget')
      if 'FATAL:' in Path('sim.log').read_text():raise RuntimeError('Simulator fatal; frozen sim.log')
     time.sleep(.2)
    if proc.returncode:raise RuntimeError(phase+' exit '+str(proc.returncode))
   finally:
    if proc.poll() is None:
     os.killpg(proc.pid,signal.SIGTERM)
     try:proc.wait(timeout=2)
     except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=2)
  if phase=='compile':assert check(Path.cwd(),application=(631,2),fabric=(638,4))['application_pes']==1262
 verify();Path('COMPLETE.json').write_text(json.dumps(dict(physical=False,seconds=time.monotonic()-started,result=json.loads(Path('result.json').read_text())))+'\n')
except BaseException as e:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(e).__name__,message=str(e)))+'\n');raise
