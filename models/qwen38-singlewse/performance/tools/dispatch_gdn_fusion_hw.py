"""Dispatch one fresh payload-verified physical connected frontend/recurrent qualification."""
import argparse,json,re,shlex,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser();parser.add_argument('attempt');args=parser.parse_args()
if not re.fullmatch('[0-9]{3}',args.attempt):raise ValueError('Attempt')
name='gdn-fusion-hw-'+args.attempt;out=ROOT/'evidence'/name
if not json.loads((out/'payload-staging.json').read_text())['payload_verified']:raise ValueError('Fixture transfer not admitted')
if (out/'dispatch.json').exists():raise ValueError('Previously dispatched attempt')
root='/srv/qwen38-singlewse-hardware/'+name
script='root_name='+repr(root)+'\n'+'''import json,os,subprocess
from pathlib import Path
root=Path(root_name);os.chdir(root)
from source_gate import verify
verify()
assert not (root/'SUPERVISOR.json').exists()
with (root/'supervisor.log').open('x') as out:
 proc=subprocess.Popen(['/opt/cerebras/venv/bin/python','-u','supervise.py'],
  cwd=root,stdout=out,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
receipt=dict(pid=proc.pid,root=str(root))
with (root/'SUPERVISOR.json').open('x') as out:
 json.dump(receipt,out);out.write('\\n');out.flush();os.fsync(out.fileno())
print(json.dumps(receipt))
'''
result=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
receipt=json.loads(result.stdout)
with (out/'dispatch.json').open('x') as f:json.dump(receipt,f,indent=2);f.write('\n')
print(json.dumps(receipt))
