"""Separate released-owner arithmetic and complete checkpoint restoration audit."""
from pathlib import Path
import gc
import hashlib
import io
import json
import time
from checkpoint import Restore
from runtime_gate import ROOT, verify, pinned, restore_source
from runtime_boundary import require
from runtime_evidence import RawEvidence, Evidence, AuditPrepared, numeric_inputs
from runtime_prepared import Prepared
from runtime_families import load
from runtime_journal import journal
from runtime_control import audit as audit_control
from runtime_timing import atomic_json


def main():
    import numpy as np
    started = time.monotonic()
    inputs, profiles, _ = verify('audit')
    from runtime_audit_guard import release_gate
    release_gate(profiles['audit'])
    require(not (ROOT/'numeric-audit.json').exists(), 'One offline attempt; preserve any previous result')
    captured = json.loads((ROOT/'capture.json').read_bytes())
    require(captured['mode'] == inputs['mode'], 'Exact admitted capture mode')
    host, lowered = [json.loads((ROOT/name).read_bytes()) for name in ('host-plan.json', 'lowered.json')]
    prepared, families = Prepared(inputs['prepared']), load()
    pins = {k: v['pin'] for k, v in inputs['prepared'].items()}
    require(captured['artifact_sha256'] == inputs['artifact_pin']['sha256'] and captured['prepared_pins'] == pins,
            'Actual raw capture of the admitted artifact and each original layer')
    checked_journal = journal(ROOT, captured, host, prepared)
    raw = RawEvidence(ROOT, captured)
    require({p.name for p in (ROOT/'raw').iterdir()} == captured['raw_files'].keys(), 'Every raw file and no unknown evidence')
    for name in captured['raw_files']:
        raw.raw(name)
    baseline_root = baseline = restore = None
    if inputs['mode'] == 'restore':
        _, baseline_root, baseline, checkpoint = restore_source(inputs)
        restore = Restore(baseline_root, checkpoint, host, lowered, inputs['artifact_pin']['sha256'], pins)
    checked_control = audit_control(raw, host, lowered, families, prepared, restore)
    if inputs['mode'] == 'baseline':
        from qualified_linear_math import load as load_linear
        from qualified_attention_math import load as load_attention
        from linear_numeric import audit as audit_linear
        from attention_numeric import audit as audit_attention
        layer_reports = []
        for layer in range(4):
            nominal_pin = inputs['nominal_outputs'][str(layer)]
            encoded = pinned(Path(inputs['reference_root'])/f'step0-layer{layer}-output.npz', nominal_pin, 65536)
            with np.load(io.BytesIO(encoded), allow_pickle=False) as archive:
                require(archive.files == ['hidden'] and archive['hidden'].shape == (1, 5, 5120) and
                        archive['hidden'].dtype == np.dtype('<u2'), 'Original official five-token chunk-prefill nominal boundary')
                nominal = archive['hidden'][0, :2].copy()
            plan, copies = numeric_inputs(families, lowered, host, layer)
            evidence = Evidence(ROOT, captured, host, lowered, layer)
            audit, math = (audit_linear, load_linear()) if layer < 3 else (audit_attention, load_attention())
            result = audit(evidence, AuditPrepared(prepared, evidence), plan, copies, math, nominal)
            result.update(original_layer=layer, physical_x_offset=plan['physical_x_offset'], nominal_reference=nominal_pin,
                          input_source='original_embedding' if layer == 0 else 'preceding_original_layer_actual_device_hidden',
                          nominal_path='official_five_token_chunk_prefill; actual device runs sequential recurrent positions')
            name = f'numeric-layer{layer}.json'
            atomic_json(ROOT/name, result)
            content = (ROOT/name).read_bytes()
            layer_reports.append(dict(layer=layer, path=name, bytes=len(content), sha256=hashlib.sha256(content).hexdigest(),
                                      status=result['status'], full_model=False))
            del evidence, result, math
            gc.collect()
        result = dict(status='all_four_original_layer_operator_gates_passed', layers=layer_reports,
                      actual_inputs_conditioned=True, whole_chain_propagated_enclosure=False,
                      nominal_prefill_vs_recurrent_path_distinction=True, restored_position1_qualified=False)
    else:
        from restore_compare import compare
        comparison = compare(baseline_root, baseline, ROOT, captured, host, lowered)
        result = dict(status='fresh_restored_position1_all_semantic_banks_bit_exact', comparison=comparison,
                      baseline_operator_audit_pinned=True, separate_restored_arithmetic_rerun=False)
    preparation = prepared.verify_unchanged()
    verify('audit')
    release_gate(profiles['audit'])
    result.update(journal=checked_journal, control=checked_control, preparation=preparation,
                  capture_raw_bytes_checked=raw.read_bytes, total_seconds=time.monotonic()-started,
                  mode=inputs['mode'], full_model=False, SDK_imported=False, CPU_forward=False,
                  controller_acceptance=False)
    atomic_json(ROOT/'numeric-audit.json', result)


if __name__ == '__main__':
    main()
