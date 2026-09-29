"""Verify the complete bridge stage against P48's admitted original bank map."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_frontend_stage import verified_frozen


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(attempt, census_path, output):
    if not re.fullmatch(r'layer-mlp-compile-\d{3}', attempt) or output.exists():
        raise ValueError('New frozen full-stage audit required')
    folder = ROOT/'evidence'/attempt
    base = ROOT/'evidence/layer-mlp-compile-029'
    def read(root, name):
        manifest = json.loads((root/'source-manifest.json').read_text())['files']
        return verified_frozen(root/'source'/name, manifest[name])
    def profiles(root):
        result = {}
        for c in json.loads(read(root, 'profiles.json'))['profiles']:
            for pe in c['pes']:
                assert tuple(pe) not in result
                result[tuple(pe)] = c
        return result
    complete = json.loads((folder/'COMPLETE.json').read_text())
    assert complete['passed'] and complete['sram_passed'] and not complete['executed']
    census = json.loads(census_path.read_text())
    assert census['passed'] and census['attempt'] == attempt
    assert census['source_manifest_sha256'] == sha((folder/'source-manifest.json').read_bytes())
    before, after = profiles(base), profiles(folder)
    assert set(before) == set(after) == {(x, y) for x in range(63, 141) for y in range(146)}
    immutable = ['gdn-bank-placement.json', 'gdn-compact-mlp.json', 'worker-setups.json',
                 'previous-worker-setups.json', 'mixer-setups.json', 'resident-gdn-routes.json']
    banks = {}
    for name in immutable:
        raw = read(folder, name)
        assert raw == read(base, name), name
        banks[name] = sha(raw)
    plan = json.loads(read(folder, 'resident-gdn-bridge-routes.json'))
    bridges = {tuple(p['bridge']) for p in plan['plans']}
    fronts = {tuple(p['frontend']) for p in plan['plans']}
    workers = {tuple(pe) for p in plan['plans'] for k in ('source_workers', 'child_workers') for pe in p[k]}
    bridge_names = {'bridge_group', 'bridge_frames', 'bridge_workers', 'bridge_input',
                    'bridge_output', 'bridge_return_input', 'bridge_return_output',
                    'bridge_forward_queue', 'bridge_reverse_queue'}
    checked_sources = set()
    for pe, old in before.items():
        new = after[pe]
        if pe in bridges:
            assert new['source'] == 'bridge_' + old['source']
            assert set(new['parameters']) - set(old['parameters']) == bridge_names
        elif pe in fronts:
            assert new['source'] == 'weighted_' + old['source']
        else:
            assert new['source'] == old['source']
        allowed_colors = {'gdn_input_color', 'gdn_output_color'} if pe in workers else (
            {'gdn_send_color', 'gdn_return_color'} if pe in fronts else set())
        old_params = {k: v for k, v in old['parameters'].items() if k not in allowed_colors}
        new_params = {k: v for k, v in new['parameters'].items()
                      if k not in allowed_colors and not (pe in bridges and k in bridge_names)}
        assert old_params == new_params, pe
        pair = old['source'], new['source']
        if pair not in checked_sources:
            a, b = read(base, old['source']), read(folder, new['source'])
            assert b.startswith(a) if pe in bridges | fronts else b == a
            checked_sources.add(pair)
    # The tested program body, as well as its placement/parameters, must be the
    # newly admitted version. Old compile030 cannot qualify the retirement fix.
    resource = ROOT/'evidence/layer-backend-compile-043'
    assert read(folder, 'bridge_compact_layer_projection.csl') == read(resource, 'bridge_compact_layer_projection.csl')
    baseline_proof = ROOT/'evidence/layer-frontend-reference-004/bank-remap-proof.json'
    if not baseline_proof.exists():
        raise ValueError('Original bank proof is missing')
    report = dict(passed=True, attempt=attempt, application_pes=len(after),
                  original_banks_bytes=388094976, original_bank_maps_and_setups=banks,
                  original_parameters_except_selected_transport_colors_unchanged=True,
                  original_source_bodies_retained=True, tested_corrected_bridge_body_admitted=True,
                  bridges=sorted(map(list, bridges)), weighted_frontends=sorted(map(list, fronts)),
                  source_manifest_sha256=sha((folder/'source-manifest.json').read_bytes()),
                  census_receipt_sha256=sha(census_path.read_bytes()),
                  reference_attempt='layer-frontend-reference-004', reference_proof_sha256=sha(baseline_proof.read_bytes()),
                  physical=False, neural_execution=False,
                  scope='Unchanged exact original bank maps, native parameters and original code bodies. '
                        'Reference004 applies to these identical bank descriptors; no new payload readback or neural execution.')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('--census', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.attempt, args.census, args.output)
