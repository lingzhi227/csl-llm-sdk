"""One fresh, bounded microkernel compile/runtime pair with correlated cleanup."""
import fcntl
import json
from pathlib import Path
import signal
from runtime.lifecycle import stage, query, TERMINAL
from runtime.store import atomic_json
from source_gate import verify
from backend import complete_mlp_profile, message_limit


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
            complete_mlp=complete_mlp_profile(config)
            if config.get('complete_original_mlp') and not complete_mlp:
                raise ValueError('Unexpected complete MLP profile')
            message_limit(config)
            if full_matrix and (config.get('application_pes')!=2600 or config.get('matrix_shape')!=[17408,5120]):
                raise ValueError('Unexpected full matrix profile')
            if complete_mlp and (root/'reuse-compiler-output.json').exists():
                from backend import file_sha256
                from check_sram import check
                reuse=json.loads((root/'reuse-compiler-output.json').read_text())
                original=json.loads((root/'reused-compile-audit.json').read_text())
                admission=json.loads((root/'artifact-admission.json').read_text())
                assert original['stage']=='compile' and original['exit_code']==1 and original['error'] is None and not original['cleanup_errors']
                assert original['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] and not j['cancelled'] for j in original['jobs'])
                assert admission['passed'] and admission['physical_job_submitted'] is False
                assert admission['source_manifest_sha256']==file_sha256(root/'source-manifest.json')
                assert {p.name:file_sha256(p) for p in root.glob('*.csl')}==reuse['csl_hashes']
                artifact=json.loads((root/'artifact.json').read_text())
                assert artifact['sha256']==reuse['artifact_sha256']==file_sha256(Path(artifact['artifact']))
                assert check(root,offset=(67,1))['application_pes']==11388
                receipts=[dict(stage='compile_reused_after_host_admission',compiler_service=original,artifact_admission=admission)]
            elif (root/'reuse-artifact.json').exists():
                import hashlib
                from check_sram import check
                reuse=json.loads((root/'reuse-artifact.json').read_text())
                assert config.get('compiled_component_reuse') is True and not full_matrix
                assert reuse['compiler_succeeded_and_released']
                audit=json.loads((root/'reused-compile-audit.json').read_text())
                assert audit['stage']=='compile' and audit['exit_code']==0 and not audit['cleanup_errors'] and audit['error'] is None
                assert audit['jobs'] and all(j['phase']=='SUCCEEDED' and j['released'] for j in audit['jobs'])
                assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csl')}==reuse['csl_hashes']
                artifact=json.loads((root/'artifact.json').read_text())
                assert artifact['sha256']==reuse['artifact_sha256']
                assert hashlib.sha256(Path(artifact['artifact']).read_bytes()).hexdigest()==reuse['artifact_sha256']
                assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (root/'out/bin').glob('*.elf')}==reuse['elf_hashes']
                assert check(root)['application_pes']==config['application_pes']==reuse['application_pes']
                receipts=[dict(stage='compile_reused',provenance=reuse)]
            else:
                receipts = [stage(root, "compile", "compile_hw.py", 900 if full_matrix or complete_mlp else 600)]
            receipts.append(stage(root, "run", "run_hw.py", 900 if complete_mlp else (600 if full_matrix else 300)))
            result = json.loads((root / "result.json").read_text())
            if not (result["passed"] and result["normal_stop"] and result["physical"]):
                raise ValueError("Physical numerical acceptance failed")
            verify()
            scope = result['scope'] if 'scope' in result else config['scope']
            atomic_json(root / "COMPLETE.json", dict(scope=scope, full_model=False,
                        full_expert=result.get('full_expert',False),
                        stages=receipts, all_owned_jobs_released=True,
                        west_to_east=result.get("west_to_east",False)))
            atomic_json(root / "ACTIVE.json", dict(phase="complete", active_jobs=[]))
        except BaseException as error:
            atomic_json(root / "FAILURE.json", dict(error=type(error).__name__, message=str(error)))
            raise


if __name__ == "__main__":
    main()
