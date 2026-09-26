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
            receipts = [stage(root, "compile", "compile_hw.py", 600)]
            receipts.append(stage(root, "run", "run_hw.py", 300))
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
