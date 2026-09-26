"""One bounded original full-model attempt; preserve failed evidence and release jobs."""
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
        if (root/'ACTIVE.json').exists():raise ValueError('Reconcile existing attempt')
        if any(j['status']['phase'] not in TERMINAL for j in query('jobs','--my-jobs')['items']):
            raise ValueError('Another account job is active')
        try:
            config=json.loads((root/'experiment.json').read_text());reused=None
            if config.get('recover_artifact'):
                reused=config['recover_artifact'];audit=reused['audit']
                if not audit['jobs'] or audit['error'] or audit['cleanup_errors']:
                    raise ValueError('Original compiler resource audit failed')
                if not all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs']):
                    raise ValueError('Original compiler job was not successfully released')
                # The original host archive-bound failure stays in its receipt.
                # Revalidation writes the actual SRAM/source gate in this attempt.
                receipts=[]
            elif config.get('reuse_compiled'):
                reused=json.loads((root/'reused-compile.json').read_text())
                audit=reused['audit'];sram=json.loads((root/'sram.json').read_text())
                if audit['exit_code']!=0 or audit['error'] or audit['cleanup_errors'] or not audit['jobs']:
                    raise ValueError('Reused compiler audit failed')
                if not all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in audit['jobs']):
                    raise ValueError('Reused compiler job was not successfully released')
                if not sram['passed'] or sram['physical_application_pes']!=870000:
                    raise ValueError('Reused full-model SRAM gate failed')
                receipts=[]
            else:receipts=[stage(root,'compile','compile_hw.py',2400,profile='full-model')]
            admission=json.loads((root/'PRE_RUN_ADMISSION.json').read_text())
            if not all(admission[k] for k in ('passed','full_geometry_compiled','every_application_elf_sram_passed','embedded_sources_identical')):
                raise ValueError('Actual full geometry/ELF gates must pass before device allocation')
            receipts.append(stage(root,'run','run_hw.py',1800,profile='full-model'))
            result=json.loads((root/'result.json').read_text())
            if result.get('physical_capture_complete'):
                if not all(result[k] for k in ('normal_stop','physical','full_model','all_weights_retained')):
                    raise ValueError('Physical capture did not complete cleanly')
                verify()
                atomic_json(root/'PHYSICAL_CAPTURE_COMPLETE.json',dict(
                  model=result['model'],revision=result['revision'],tokens=result['tokens'],
                  stages=receipts,reused_compile=reused,all_owned_jobs_released=True,
                  numerical_qualification_pending=result['numerical_qualification_pending'],
                  strict_original_reference_passed=result['strict_original_reference_passed']))
                if result['numerical_qualification_pending']:
                    atomic_json(root/'NUMERICAL_REVIEW_REQUIRED.json',dict(
                      reason='Original strict per-layer comparisons did not all pass; actual-input operator captures preserved',
                      physical_capture_complete=True,full_model_accepted=False))
                    atomic_json(root/'ACTIVE.json',dict(phase='awaiting_numerical_review',active_jobs=[]))
                    return
            if not all(result[k] for k in ('passed','normal_stop','physical','full_model','all_weights_retained')):
                raise ValueError('Full-model physical acceptance failed')
            verify();atomic_json(root/'COMPLETE.json',dict(scope=result['scope'],full_model=True,
              model=result['model'],revision=result['revision'],tokens=result['tokens'],
              stages=receipts,reused_compile=reused,all_owned_jobs_released=True,west_to_east=True))
            atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as error:
            atomic_json(root/'FAILURE.json',dict(error=type(error).__name__,message=str(error)));raise

if __name__=='__main__':main()
