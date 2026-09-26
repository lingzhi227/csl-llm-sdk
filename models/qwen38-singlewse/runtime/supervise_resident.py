"""One resident physical candidate, with admission and invocation-owned cleanup."""
import fcntl,json,signal
from pathlib import Path
from runtime.lifecycle import stage,query,TERMINAL
from runtime.store import atomic_json
from source_gate import verify

def interrupted(number,frame):raise RuntimeError('Supervisor signal '+str(number))

def main():
    root=Path.cwd();signal.signal(signal.SIGINT,interrupted);signal.signal(signal.SIGTERM,interrupted)
    with Path('/srv/cerebras-hardware/hardware.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);verify()
        assert not (root/'ACTIVE.json').exists()
        active=query('jobs','--my-jobs');assert all(j['status']['phase'] in TERMINAL for j in active['items'])
        config=json.loads((root/'experiment.json').read_text());assert config['full_resident'] and config['application_pes']==870000
        deadline=config['deadline_seconds'];assert isinstance(deadline,int) and 7200<=deadline<=10800
        reuse=json.loads((root/'reuse-artifact.json').read_text());assert reuse['compiler_succeeded_and_released']
        sram=json.loads((root/'sram.json').read_text());assert sram['passed'] and sram['application_pes']==870000
        checkpoint=json.loads(Path('/srv/qwen38-singlewse-hardware/model/COMPLETE.json').read_text())
        assert checkpoint['repository']=='Qwen/Qwen3.8-27B-FP8' and checkpoint['revision']=='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a' and len(checkpoint['files'])==74
        try:
            receipt=stage(root,'run','resident_run.py',deadline)
            candidate=json.loads((root/'CANDIDATE.json').read_text())
            assert candidate['candidate_execution_complete'] and candidate['normal_stop'] and candidate['physical'] and not candidate['full_model_accepted']
            verify();atomic_json(root/'COMPLETE.json',dict(scope='Complete resident physical candidate; frozen offline numerical acceptance pending',full_model=False,physical_execution=True,stages=[receipt],all_owned_jobs_released=True,compile_provenance=reuse))
            atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as error:
            atomic_json(root/'FAILURE.json',dict(error=type(error).__name__,message=str(error)));raise

if __name__=='__main__':main()
