"""One fresh, bounded microkernel compile/runtime pair with correlated cleanup."""
import fcntl
import json
from pathlib import Path
import signal
from runtime.lifecycle import stage, query, TERMINAL
from runtime.store import atomic_json
from source_gate import verify


def interrupted(number, frame):
    raise RuntimeError("Supervisor signal " + str(number))


def main():
    root = Path.cwd()
    signal.signal(signal.SIGINT, interrupted)
    signal.signal(signal.SIGTERM, interrupted)
    # Use the same site resource lock as the prior Qwen hardware work.
    with Path("/srv/cerebras-hardware/hardware.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        verify()
        if (root / "ACTIVE.json").exists():
            raise ValueError("Existing attempt requires reconciliation")
        active = query("jobs", "--my-jobs")
        if any(j["status"]["phase"] not in TERMINAL for j in active["items"]):
            raise ValueError("Another account job is active")
        try:
            config=json.loads((root/'experiment.json').read_text()) if (root/'experiment.json').exists() else {}
            full_matrix=config.get('full_matrix',False)
            if full_matrix and (config.get('application_pes')!=2600 or config.get('matrix_shape')!=[17408,5120]):
                raise ValueError('Unexpected full matrix profile')
            if (root/'reuse-artifact.json').exists():
                import hashlib
                from check_sram import check
                reuse=json.loads((root/'reuse-artifact.json').read_text())
                assert full_matrix and reuse['compiler_succeeded_and_released']
                assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csl')}==reuse['csl_hashes']
                assert json.loads((root/'artifact.json').read_text())['sha256']==reuse['artifact_sha256']
                assert check(root)['application_pes']==2600
                receipts=[dict(stage='compile_reused',provenance=reuse)]
            else:
                receipts = [stage(root, "compile", "compile_hw.py", 900 if full_matrix else 600)]
            receipts.append(stage(root, "run", "run_hw.py", 600 if full_matrix else 300))
            result = json.loads((root / "result.json").read_text())
            if not (result["passed"] and result["normal_stop"] and result["physical"]):
                raise ValueError("Physical numerical acceptance failed")
            verify()
            atomic_json(root / "COMPLETE.json", dict(scope=result["scope"], full_model=False,
                        full_expert=result.get('full_expert',False),
                        stages=receipts, all_owned_jobs_released=True,
                        west_to_east=result.get("west_to_east",False)))
            atomic_json(root / "ACTIVE.json", dict(phase="complete", active_jobs=[]))
        except BaseException as error:
            atomic_json(root / "FAILURE.json", dict(error=type(error).__name__, message=str(error)))
            raise


if __name__ == "__main__":
    main()
