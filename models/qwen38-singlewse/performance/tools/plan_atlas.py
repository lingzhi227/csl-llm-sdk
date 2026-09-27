"""Build complete original model ownership without reading checkpoint payloads."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.atlas import AtlasPolicy,build
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
r=build(ROOT.parent,AtlasPolicy())
with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(dict(metrics=r['metrics'],bank_profiles=r['bank_profiles'],quant_storage=r['quant_storage'])))
