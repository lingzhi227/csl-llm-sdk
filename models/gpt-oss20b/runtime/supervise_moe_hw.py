"""One full MoE-layer diagnostic attempt; all owned jobs released on any outcome."""
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
            config=json.loads((root/'experiment.json').read_text())
            reused=None
            if config.get('reuse_compiled'):
                reused=json.loads((root/'reused-compile.json').read_text())
                if not json.loads((root/'sram.json').read_text())['passed']:raise ValueError('Reused SRAM gate failed')
                if not all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in reused['audit']['jobs']):
                    raise ValueError('Reused compiler job was not successfully released')
                receipts=[]
            else:receipts=[stage(root,'compile','compile_hw.py',900)]
            receipts.append(stage(root,'run','run_hw.py',900))
            result=json.loads((root/'result.json').read_text())
            config=json.loads((root/'experiment.json').read_text())
            scope_key='full_decoder_layer' if config.get('full_decoder_layer') else 'full_moe_layer' if config.get('full_moe_layer') else 'full_attention'
            if not all(result[k] for k in ('passed','normal_stop','physical',scope_key,'all_weights_retained')):
                raise ValueError('Whole MoE physical acceptance failed')
            verify();atomic_json(root/'COMPLETE.json',dict(scope=result['scope'],full_model=False,
              **{scope_key:True},stages=receipts,reused_compile=reused,all_owned_jobs_released=True,west_to_east=True))
            atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
        except BaseException as error:
            atomic_json(root/'FAILURE.json',dict(error=type(error).__name__,message=str(error)));raise

if __name__=='__main__':main()
