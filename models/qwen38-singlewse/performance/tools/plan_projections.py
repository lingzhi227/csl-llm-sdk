"""Generate compact complete shared-input projection and reduction-profile plans."""
import argparse
import json
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from spatial.projections import build
from spatial.forest import document
parser = argparse.ArgumentParser(); parser.add_argument('--atlas', type=Path, required=True); parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
if args.output.exists():
    raise ValueError('Frozen output exists')
bundles = build(ROOT.parent, args.atlas); atlas = json.loads(args.atlas.read_text())
forests = document(atlas, bundles)
args.output.mkdir()
for name, data in [('bundles.json', bundles), ('forests.json', forests)]:
    with (args.output / name).open('x') as f:
        json.dump(data, f, indent=2); f.write('\n')
print(json.dumps(bundles['metrics']))
