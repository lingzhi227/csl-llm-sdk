"""Emit compact full-model storage overlay and paired column-input descriptors."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.columnar import build
p=argparse.ArgumentParser();p.add_argument('--atlas',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
result=build(a.atlas)
with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(dict(metrics=result['metrics'],classes=result['classes'],profiles=[{k:v for k,v in r.items() if k in ['k_blocks','max_horizontal_stream_words','max_column_stream_words','unused_application_colors']} for r in result['input_profiles']])))
