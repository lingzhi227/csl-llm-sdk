"""Start exactly one new, staged micro-experiment through the existing ALCF session."""
import argparse,json,re,shlex,subprocess
from pathlib import Path

p=argparse.ArgumentParser();p.add_argument('name');args=p.parse_args()
if not re.fullmatch(r'[a-z][a-z0-9-]+-hw-[0-9]{3}',args.name):raise ValueError('Invalid experiment name')
root='/srv/gpt-oss20b-hardware/'+args.name
script='root_name='+repr(root)+'\n'+'''
import json,os,subprocess
from pathlib import Path
root=Path(root_name)
os.chdir(root)
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
r=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],
                  capture_output=True,text=True,check=True,timeout=30)
receipt=json.loads(r.stdout)
out=Path(__file__).resolve().parents[1]/'evidence'/args.name
assert out.is_dir()
(out/'dispatch.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
