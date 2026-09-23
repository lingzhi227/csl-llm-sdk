"""Close immutable preparation identities once without rerunning numerical work."""
from pathlib import Path
import hashlib
import time
from finalizer_gate import ROOT, BASE, RESUMED, atomic_json, checked, require, stamp, verify


def main():
    started = time.monotonic()
    data, _, _ = verify()
    require(not (ROOT / 'numeric-audit.json').exists(), 'No repeated completion')
    receipts, stamps = {}, {}
    total = 0
    for key, value in data['inputs']['prepared'].items():
        layer = int(key)
        root = Path(value['root'])
        require(root.parent == ROOT.parent and not root.is_symlink() and root.stat().st_dev == 51, 'Original parameter root')
        path = root / 'prepared.json'
        manifest = checked(path, value['pin'], True, True)
        stamps[str(path)] = stamp(path, True)
        require(manifest['all_original_bits_checked'] is True and manifest['application'] == [29 if layer == 3 else 30, 1160], 'Accepted preparation family')
        require(len(manifest['files']) == [199, 198, 198, 55][layer], 'Exact original array inventory')
        excluded = {'reference-input.npy', 'reference-output.npy'} if layer == 3 else set()
        require(excluded <= manifest['files'].keys(), 'Explicit Layer3 nominal exclusions exist')
        if layer in (1, 2):
            require(not {'reference-input.npy', 'reference-output.npy'} & manifest['files'].keys(), 'No intermediate nominal input')
        files = {}
        for name, pin in sorted(manifest['files'].items()):
            if name in excluded:
                continue
            require(Path(name).name == name, 'Flat prepared array path')
            before = checked(root / name, pin, True)
            stamps[str(root / name)] = before
            files[name] = dict(bytes=pin['bytes'], sha256=pin['sha256'])
            total += pin['bytes']
        receipts[key] = dict(layer=layer, root=str(root), manifest=value['pin'], execution_files=files,
                             files=len(files), disk_bytes_read=sum(p['bytes'] for p in files.values()),
                             excluded_nominal_files=sorted(excluded), all_full_hashes_exact=True,
                             original_shard_reads=0, parameter_consumption_reexecuted=False)
    require(sum(r['files'] for r in receipts.values()) == 648 and total == 3081215872, 'Complete exact execution inventory')
    end, _, _ = verify()
    require(end == data, 'Final original source and evidence gate closure')
    for path, before in stamps.items():
        require(stamp(Path(path), True) == before, 'Parameters/manifests stable over finalization')
    result = dict(status='all_four_original_layer_operator_gates_passed', mode='baseline_completion_only_audit',
                  layers=data['reports'], baseline_root=str(BASE), resumed_audit_root=str(RESUMED),
                  baseline_pins=data['inputs']['baseline_pins'], resumed_pins=data['inputs']['resumed_pins'],
                  preparation=receipts, execution_parameter_files=648, execution_parameter_bytes=total,
                  preparation_integrity_complete=True, source_end_gate_passed=True,
                  preparation_contract='All execution file SHA256 identities and readonly stamps, including runtime configuration arrays; not a claim that this process consumed parameters in numerical operators.',
                  inherited_pre_layer0_gates='Frozen original source-order proof plus independently recorded full raw/journal integrity review; detailed original return objects were not saved and were not rerun.',
                  reused_math_reports=[0, 1, 2, 3], historical_audit_failures_preserved=True,
                  missing_resumed_aggregate_preserved=True, preprocess_domain_version='chain-exp-domain-v2',
                  quantitative_bounds_unchanged=True, actual_inputs_conditioned=True,
                  whole_chain_propagated_enclosure=False, nominal_prefill_vs_recurrent_path_distinction=True,
                  restored_position1_qualified=False, mathematical_audit_passed=True, actual_restore_qualified=False,
                  total_seconds=time.monotonic() - started, SDK_imported=False, CPU_forward=False,
                  numerical_operators_reexecuted=False, full_model=False, controller_acceptance=False)
    atomic_json(ROOT / 'numeric-audit.json', result)


if __name__ == '__main__':
    main()
