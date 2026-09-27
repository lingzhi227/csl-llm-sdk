"""One bounded runtime after independently admitted existing compiler output."""
import fcntl,hashlib,json,signal
from pathlib import Path
from runtime.lifecycle import stage,query,TERMINAL
from runtime.store import atomic_json
from source_gate import verify
from check_sram import check

def interrupted(number,frame):raise RuntimeError('Supervisor signal '+str(number))

def main():
 root=Path.cwd();signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
 with Path('/srv/cerebras-hardware/hardware.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
  assert not (root/'ACTIVE.json').exists()
  assert not any(j['status']['phase'] not in TERMINAL for j in query('jobs','--my-jobs')['items'])
  try:
   config=json.loads((root/'experiment.json').read_text());reuse=json.loads((root/'reuse-compiler-output.json').read_text())
   assert config['application']==[631,2] and config['fabric_offset']==[123,706] and config['artifact_single_message_limit']==16<<20
   audit=json.loads((root/'reused-compile-audit.json').read_text())
   assert audit['exit_code']==1 and audit['error'] is None and not audit['cleanup_errors'] and not audit['uncorrelated_new_jobs']
   assert audit['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in audit['jobs'])
   admission=json.loads((root/'artifact-admission.json').read_text());framing=json.loads((root/'upload-framing.json').read_text())
   assert admission['passed'] and framing['passed'] and framing['candidate_protobuf_identical_to_original']
   artifact=json.loads((root/'artifact.json').read_text());assert artifact['sha256']==reuse['artifact_sha256']==admission['artifact_sha256']==framing['artifact_sha256']
   assert hashlib.sha256(Path(artifact['artifact']).read_bytes()).hexdigest()==artifact['sha256']
   assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csl')}==reuse['csl_hashes']
   assert check(root,offset=(123,706))['application_pes']==1262
   receipt=stage(root,'run','run_hw.py',300)
   result=json.loads((root/'result.json').read_text());assert result['passed'] and result['normal_stop'] and result['physical']
   verify();atomic_json(root/'COMPLETE.json',dict(scope=result['scope'],full_model=False,all_owned_jobs_released=True,
     stages=[dict(stage='compile_output_reused',source_attempt=reuse['source_attempt'],compiler_jobs=audit['jobs'],host_admission=admission),receipt]))
   atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
  except BaseException as error:
   atomic_json(root/'FAILURE.json',dict(error=type(error).__name__,message=str(error)));raise

if __name__=='__main__':main()
