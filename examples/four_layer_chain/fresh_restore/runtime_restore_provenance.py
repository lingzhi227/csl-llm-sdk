"""Bind the fresh checkpoint run to original failures and successful finalization."""
from pathlib import Path
import hashlib
import importlib.util
import json


def restore_source(root, inputs, pinned, require):
    require(inputs['mode'] == 'restore', 'Only a distinct fresh restored context')
    path = root / 'restore-input.json'
    require(not path.is_symlink() and not path.stat().st_mode & 0o222 and 0 < path.stat().st_size <= 128 << 10,
            'Immutable bounded actual restore input')
    raw = path.read_bytes()
    r = json.loads(raw)
    baseline, completed = Path(r['baseline_root']), Path(r['baseline_completion_root'])
    require(r['schema'] == 3 and str(baseline) == inputs['baseline_root'] and
            str(completed) == inputs['baseline_completion_root'] and len({root, baseline, completed}) == 3,
            'Distinct exact original capture, completion and restore roots')
    base_names = {'runtime-manifest.json', 'capture.json', 'checkpoint-pos0.json', 'runtime-lifecycle.json',
                  'run-audit.json', 'DISPATCH-EXIT.json', 'COMPLETE.json', 'audit-supervisor.json',
                  'AUDIT-DISPATCH-EXIT.json', 'numeric-layer0.json'}
    completion_names = {'source-manifest.json', 'inputs.json', 'admission.json', 'numeric-audit.json',
                        'audit-supervisor.json', 'AUDIT-DISPATCH-EXIT.json'}
    require(set(r['files']) == base_names and set(r['completion_files']) == completion_names,
            'Complete exact baseline and completion provenance')
    values = {name: json.loads(pinned(baseline / name, pin, 32 << 20)) for name, pin in r['files'].items()}
    finalized = {name: json.loads(pinned(completed / name, pin, 8 << 20)) for name, pin in r['completion_files'].items()}
    require(r['completion_files']['source-manifest.json'] == inputs['baseline_completion_manifest'],
            'Exact independently accepted completion source')
    manifest = finalized['source-manifest.json']
    require(manifest['candidate'] == completed.name, 'Exact completion candidate')
    for name, pin in manifest['files'].items():
        require(Path(name).name == name and not (completed / name).stat().st_mode & 0o222,
                'Frozen completion source closure')
        pinned(completed / name, pin, 1 << 20)
    # This frozen stdlib-only gate validates original/resumed sources and each
    # family report. It does not invoke finalize.py or stream prepared arrays.
    spec = importlib.util.spec_from_file_location('qualified_chain_completion_gate', completed / 'finalizer_gate.py')
    gate = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gate)
    data, profiles, _ = gate.verify()
    proof = data['inputs']
    require(proof['baseline_root'] == str(baseline) and proof['prepared'] == inputs['prepared'],
            'Same complete original preparation lineage')
    for name, pin in proof['baseline_pins'].items():
        if name in r['files']:
            require(r['files'][name] == pin, 'Same original physical/failed-audit/Layer0 identity')
    numeric = finalized['numeric-audit.json']
    require(numeric['status'] == 'all_four_original_layer_operator_gates_passed' and
            numeric['mode'] == 'baseline_completion_only_audit' and numeric['baseline_root'] == str(baseline) and
            numeric['resumed_audit_root'] == proof['resumed_root'] and numeric['baseline_pins'] == proof['baseline_pins'] and
            numeric['resumed_pins'] == proof['resumed_pins'] and numeric['layers'] == data['reports'] and
            numeric['preparation_integrity_complete'] and numeric['source_end_gate_passed'] and
            numeric['execution_parameter_files'] == 648 and numeric['execution_parameter_bytes'] == 3081215872 and
            numeric['historical_audit_failures_preserved'] and numeric['missing_resumed_aggregate_preserved'] and
            numeric['quantitative_bounds_unchanged'] and numeric['actual_inputs_conditioned'] and
            numeric['preprocess_domain_version'] == 'chain-exp-domain-v2' and
            numeric['numerical_operators_reexecuted'] is False and numeric['full_model'] is False,
            'Successful completion preserves all four family reports and both historical failures')
    require(set(numeric['preparation']) == {'0', '1', '2', '3'}, 'Four complete preparation identity receipts')
    for key, receipt in numeric['preparation'].items():
        require(receipt['manifest'] == inputs['prepared'][key]['pin'] and
                receipt['root'] == inputs['prepared'][key]['root'] and receipt['all_full_hashes_exact'] and
                receipt['files'] == [199, 198, 198, 53][int(key)], 'Exact preparation integrity receipts')
    audit, end = finalized['audit-supervisor.json'], finalized['AUDIT-DISPATCH-EXIT.json']
    require(end['guard_reaped'] and end['guard_exit_code'] == 0 and audit['child_reaped'] and audit['exit_code'] == 0 and
            not audit['error'] and not audit['cleanup_error'] and not audit['remaining_group_members'] and
            audit['profile'] == profiles['audit'] and audit['SDK'] is False and finalized['admission.json'] == data['admission'],
            'Completion CPU owner independently admitted, successful and reaped')
    for pid in [end['guard_pid'], end['ssh_waiter_pid'], audit['child_pid']]:
        require(not Path('/proc', str(pid)).exists(), 'No completed audit owner exists before fresh restore')
    capture, life = values['capture.json'], values['runtime-lifecycle.json']
    require(capture['mode'] == 'baseline' and capture['completed'] == [1, 2, 3] and capture['normal_stop'] and
            capture['artifact_sha256'] == inputs['artifact_pin']['sha256'] and
            capture['prepared_pins'] == {k: v['pin'] for k, v in inputs['prepared'].items()},
            'Identical artifact, own weights and completed baseline')
    require(life['normal_stop'] and not life['primary_error'] and not life['stop_error'] and
            values['COMPLETE.json']['all_owned_jobs_released'], 'Normally stopped released physical baseline')
    checkpoint = capture['checkpoint']
    require({k: checkpoint[k] for k in ['bytes', 'sha256']} == r['files']['checkpoint-pos0.json'],
            'Actual complete original device checkpoint identity')
    return dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()), baseline, capture, checkpoint
