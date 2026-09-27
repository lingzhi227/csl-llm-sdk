import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.model import load_original
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=load_original(ROOT.parent)
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result['metrics']))
