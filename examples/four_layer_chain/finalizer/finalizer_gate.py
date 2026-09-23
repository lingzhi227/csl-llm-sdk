"""Stdlib-only exact provenance, family report, and source closure validation."""
from pathlib import Path
import hashlib
import json
import os
import stat

ROOT = Path(__file__).resolve().parent
BASE = ROOT.parent / 'layer-chain-hw-run-004'
RESUMED = ROOT.parent / 'layer-chain-resumed-audit001'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stamp(path, readonly=False):
    info = path.lstat()
    require(stat.S_ISREG(info.st_mode) and not path.is_symlink(), 'Regular non-symlink file: ' + str(path))
    require(not readonly or not info.st_mode & 0o222, 'Readonly source/parameter: ' + str(path))
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns, info.st_mode)


def checked(path, pin, readonly=False, decode=False):
    before = stamp(path, readonly)
    require(0 <= pin['bytes'] <= 32 << 20 and before[2] == pin['bytes'], 'Bounded exact file size')
    digest = hashlib.sha256()
    chunks = [] if decode else None
    with path.open('rb') as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
            if decode:
                chunks.append(chunk)
    require(stamp(path, readonly) == before and digest.hexdigest() == pin['sha256'], 'Stable exact SHA256: ' + str(path))
    return json.loads(b''.join(chunks)) if decode else before


def local_json(name):
    path = ROOT / name
    require(stamp(path, True)[2] <= 128 << 10, 'Small readonly finalizer metadata')
    return json.loads(path.read_bytes())


def closure(root, name, pin):
    manifest = checked(root / name, pin, True, True)
    require(isinstance(manifest['files'], dict) and len(manifest['files']) <= 128, 'Bounded source inventory')
    for relative, item in manifest['files'].items():
        path = Path(relative)
        require(not path.is_absolute() and '..' not in path.parts, 'Relative frozen source')
        checked(root / path, item, True)
    return manifest


def expected_admission(manifest_raw, profile, inputs):
    return dict(candidate=ROOT.name, kind='completion_only_audit',
                source_manifest_sha256=hashlib.sha256(manifest_raw).hexdigest(), profile=profile,
                execution_parameter_files=648, execution_parameter_bytes=3081215872,
                completed_layers_reused=[0, 1, 2, 3], original_failures_preserved=True,
                original_sessions_reaped=True, controller_resource_release_confirmed=True,
                SDK=False, mathematical_reexecution=False, automatic_retry=False)


def report_summary(layer, report, pin, root):
    require(report['original_layer'] == layer and report['layer'] == layer and report['positions'] == 2,
            'Original layer and finite position coverage')
    require(report['full_model'] is False and report['source_propagated_whole_layer_enclosure'] is False,
            'Preserved numerical claim limits')
    if layer < 3:
        require(report['status'] == 'all_operator_gates_passed' and report['serials'] == 3 and
                report['all_conditional_matrix_rows'] == 12114432 and report['exact_FMA_samples'] == 94644 and
                report['exact_sample_FMA_slots'] == 9085824 and report['full128_recurrent_numeric_exact'] is True and
                report['original_projection_and_neural_handoffs'] is True and report['nominal_original_outputs_compared'] is True,
                'Qualified linear family report')
    else:
        require(report['status'] == 'passed' and report['serial_executions'] == 3 and
                report['local_conditional_rows'] == 11741184 and report['exact_FMA_samples'] == 91728 and
                report['exact_FMA_multiply_slots'] == 8805888 and report['all_operator_gates'] is True and
                report['device_reset_replay_exact'] is True and report['original_whole_layer_nominal_compared'] is True,
                'Qualified attention family report')
    results = report['results']
    require([(r['serial'], r['position']) for r in results] == [(1, 0), (2, 1), (3, 0)], 'Exact baseline/replay sequence')
    if layer < 3:
        require(results[0]['actual_hidden_sha256'] == results[2]['actual_hidden_sha256'], 'Linear reset output exact')
    if layer > 0:
        require(report['input_source'] == 'preceding_original_layer_actual_device_hidden', 'Actual causal input evidence')
        require(report['quantitative_bounds_unchanged'] is True, 'Unchanged operator error bounds')
    mismatch = 'nominal_BF16_mismatches' if layer < 3 else 'nominal_original_bf16_mismatches'
    return dict(layer=layer, root=str(root), path=f'numeric-layer{layer}.json', **pin,
                status=report['status'], reused_without_execution=True,
                nominal_BF16_mismatches=[r[mismatch] for r in results],
                nominal_max_absolute_error=[r['nominal_max_absolute_error'] for r in results],
                reset_replay_exact=True)


def verify(kind='audit'):
    require(kind == 'audit' and ROOT.name == 'layer-chain-audit-finalizer001' and ROOT.stat().st_dev == 51,
            'Distinct exact ALCF completion-only owner')
    raw = (ROOT / 'source-manifest.json').read_bytes()
    require(len(raw) <= 128 << 10, 'Bounded finalizer manifest')
    manifest = json.loads(raw)
    closure(ROOT, 'source-manifest.json', dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()))
    inputs, profile = local_json('inputs.json'), local_json('profile.json')
    admission = expected_admission(raw, profile, inputs)
    require(local_json('admission.json') == admission, 'Exact controller completion-only admission')
    require(inputs['baseline_root'] == str(BASE) and inputs['resumed_root'] == str(RESUMED), 'Exact predecessor roots')
    closure(BASE, 'runtime-manifest.json', inputs['baseline_pins']['runtime-manifest.json'])
    closure(RESUMED, 'source-manifest.json', inputs['resumed_pins']['source-manifest.json'])
    values = {}
    for label, root in [('baseline', BASE), ('resumed', RESUMED)]:
        values[label] = {}
        for name, pin in inputs[label + '_pins'].items():
            value = checked(root / name, pin, decode=name.endswith('.json'))
            if name.endswith('.json'):
                values[label][name] = value
    old, recent = values['baseline'], values['resumed']
    run, dispatch = old['run-audit.json'], old['DISPATCH-EXIT.json']
    require(dispatch['guard_reaped'] and dispatch['guard_exit_code'] == 0 and run['child_reaped'] and
            run['exit_code'] == 0 and not run['primary_error'] and not run['cleanup_errors'] and
            not run['remaining_group_members'] and not run['uncorrelated_new_jobs'] and len(run['jobs']) == 1 and
            all(j['phase'] == 'SUCCEEDED' and j['released'] and not j['cancelled'] for j in run['jobs']),
            'Successful physical baseline is released')
    require(old['capture.json']['completed'] == [1, 2, 3] and old['capture.json']['normal_stop'], 'Complete physical capture')
    pids = [dispatch['guard_pid'], dispatch['ssh_waiter_pid'], run['child_pid']]
    for source in [old, recent]:
        guard, end = source['audit-supervisor.json'], source['AUDIT-DISPATCH-EXIT.json']
        require(end['guard_reaped'] and end['guard_exit_code'] == 1 and guard['child_reaped'] and
                guard['exit_code'] == 1 and not guard['remaining_group_members'] and not guard['cleanup_error'],
                'Both historical audit failures remain failed and reaped')
        pids += [end['guard_pid'], end['ssh_waiter_pid'], guard['child_pid']]
    require(all(not Path('/proc', str(pid)).exists() for pid in pids), 'Prior owners absent')
    require(not (RESUMED / 'numeric-audit.json').exists(), 'Missing old final aggregate preserved')
    require(recent['inputs.json']['baseline_pins'] == {k: inputs['baseline_pins'][k] for k in recent['inputs.json']['baseline_pins']},
            'Resumed audit exact baseline provenance')
    require(recent['admission.json'] == json.loads((RESUMED / 'admission.template.json').read_bytes()),
            'Exact prior controller admission')
    original_inputs = json.loads((BASE / 'runtime-inputs.json').read_bytes())
    require(inputs['prepared'] == original_inputs['prepared'], 'Same four actual execution preparations')
    integrity = json.loads((RESUMED / 'ORIGINAL-RAW-INTEGRITY-REVIEW.json').read_bytes())
    require(integrity['capture_sha256'] == inputs['baseline_pins']['capture.json']['sha256'] and
            all(integrity[k] for k in ['all_raw_sha256_exact', 'all_stat_stable', 'raw_directory_exact', 'every_durable_receipt_exact_once']),
            'Independent full raw/journal integrity provenance')
    reports = []
    for layer in range(4):
        source, root = (old, BASE) if layer == 0 else (recent, RESUMED)
        name = f'numeric-layer{layer}.json'
        pin = inputs['baseline_pins' if layer == 0 else 'resumed_pins'][name]
        reports.append(report_summary(layer, source[name], pin, root))
    return dict(inputs=inputs, reports=reports, admission=admission), dict(audit=profile), None


def release_gate(profile):
    data, profiles, _ = verify()
    require(profile == profiles['audit'], 'Exact finalizer resource profile')
    return data['admission']


def atomic_json(path, value):
    raw = (json.dumps(value, separators=(',', ':')) + '\n').encode()
    require(len(raw) <= 8 << 20 and not path.exists(), 'Bounded new immutable output only')
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('xb') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.chmod(0o444)
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)
