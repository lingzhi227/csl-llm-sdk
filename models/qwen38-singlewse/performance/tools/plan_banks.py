import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.banks import BankPolicy,build
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--rows',type=int,default=2);p.add_argument('--reserved-pes',type=int,default=16384);a=p.parse_args()
r=build(ROOT.parent,BankPolicy(rows_per_tile=a.rows,reserved_actor_pes=a.reserved_pes))
with a.output.open('x') as f:json.dump(r,f,indent=2);f.write('\n')
print(json.dumps(r['metrics']))
