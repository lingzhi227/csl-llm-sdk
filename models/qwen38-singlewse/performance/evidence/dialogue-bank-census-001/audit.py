"""Read-only comparison of two complete, frozen workstation ELF censuses."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

ROOT = Path('/srv/model-storage/qwen38-singlewse/runs')
before = ROOT/'layer-mlp-compile-018'
after = ROOT/'layer-mlp-compile-021'


def read(root, name):
    return json.loads((root/name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def census(root):
    data = read(root, 'sram.json')
    assert data['passed'] and data['application'] == [78, 146]
    ox, oy = data['offset']
    found = {}
    for record in data['records']:
        assert record['passed'] and record['stack_allowance_bytes'] == 4096
        total = record['low_section_end']+4096
        assert total <= 48128
        for x, y, w, h in record['rectangles']:
            for yy in range(y, y+h):
                for xx in range(x, x+w):
                    pe = (xx-ox+63, yy-oy)
                    assert 63 <= pe[0] < 141 and 0 <= pe[1] < 146 and pe not in found
                    found[pe] = (record['source'], total)
    assert len(found) == 11388 == data['application_pes']
    return found


def profiles(root):
    found = {}
    for profile in read(root, 'profiles.json')['profiles']:
        for pe in profile['pes']:
            key = tuple(pe)
            assert key not in found
            found[key] = (profile['source'], profile['parameters'])
    assert len(found) == 11388
    return found


for root in [before, after]:
    assert read(root, 'COMPLETE.json')['sram_passed']
    for name, digest in read(root, 'source-manifest.json')['files'].items():
        assert sha(root/name) == digest, (str(root), name)
neural = [p.name for p in before.glob('*.csl') if p.name != 'layout.csl']
assert all((before/name).read_bytes() == (after/name).read_bytes() for name in neural)
same_metadata = ['mixer-setups.json', 'worker-setups.json', 'device-network.json',
                 'network-binding.json', 'joint-bank-placement.json', 'joint-stage.json']
assert all((before/name).read_bytes() == (after/name).read_bytes() for name in same_metadata)
assert (before/'layout.csl').read_bytes().split(b' const route_0=')[1] == (after/'layout.csl').read_bytes().split(b' const route_0=')[1]
old, new = census(before), census(after)
op, np = profiles(before), profiles(after)
assert old.keys() == new.keys() == op.keys() == np.keys()
roles = defaultdict(list)
gains = Counter()
cells = []
for pe in sorted(old):
    role, prior = old[pe]
    assert new[pe][0] == role == op[pe][0] == np[pe][0]
    oldp, newp = op[pe][1], np[pe][1]
    assert {k: v for k, v in oldp.items() if k != 'bank_words'} == {k: v for k, v in newp.items() if k != 'bank_words'}
    bank_gain = 4*(oldp.get('bank_words', 0)-newp.get('bank_words', 0))
    assert bank_gain >= 0
    total = new[pe][1]
    margin = 48128-total
    roles[role].append(margin)
    gains[prior-total] += 1
    cells.append([*pe, role, margin, prior-total, bank_gain])
assert sum(c[5] for c in cells) == 3207168
report = dict(passed=True, physical=False, neural_execution=False, complete_model_speed=False,
              source_attempt=before.name, candidate_attempt=after.name, application_pes=len(cells),
              unchanged_neural_csl_files=len(neural), unchanged_descriptors_and_routes=True,
              original_weights_and_work_slots_unchanged=True, bank_bytes_reclaimed=sum(c[5] for c in cells),
              actual_sram_bytes_reclaimed=sum(c[4] for c in cells), minimum_margin=min(c[3] for c in cells),
              sram_delta_to_pe_count=dict(sorted(gains.items())),
              roles={r: dict(pes=len(v), minimum_margin=min(v), maximum_margin=max(v),
                     free_pe_counts={str(n): sum(m >= n for m in v) for n in [512, 1024, 2048, 4096, 8192]}) for r, v in roles.items()},
              cell_fields=['x', 'y', 'role', 'actual_margin_bytes', 'actual_sram_gain_bytes', 'bank_gain_bytes'],
              cells=cells,
              identities={r.name: {n: sha(r/n) for n in ['source-manifest.json', 'sram.json']} for r in [before, after]},
              scope='Actual complete-program storage only. Margins include the declared4096B stack. Additional consumer code, scratch and routes require new composition/compile/runtime admission. No GDN fusion or multi-turn execution is qualified.')
print(json.dumps(report, separators=(',', ':')))
