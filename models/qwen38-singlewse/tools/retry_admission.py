"""Retry only a pre-job shared-lock/account rejection, preserving its receipts."""
import argparse,json,re,shlex,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('name');p.add_argument('attempt');a=p.parse_args()
assert re.fullmatch(r'[a-z][a-z0-9-]+-hw-[0-9]{3}',a.name)
assert re.fullmatch(r'[0-9]{3}',a.attempt)
script='root_name='+repr('/srv/qwen38-singlewse-hardware/'+a.name)+'\nattempt='+repr(a.attempt)+'\n'+'''
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(root_name);os.chdir(root)
from source_gate import verify
verify()
assert not any((root/n).exists() for n in ['ACTIVE.json','compile.jobs','run.jobs','compile-audit.json','run-audit.json','artifact.json','FAILURE.json','COMPLETE.json'])
prior=(root/'supervisor.log').read_text()
assert 'BlockingIOError: [Errno 11]' in prior or 'Another account job is active' in prior
pid=json.loads((root/'SUPERVISOR.json').read_text())['pid']
proc=Path('/proc')/str(pid)
if (proc/'cmdline').exists():assert b'supervise.py' not in (proc/'cmdline').read_bytes()
label='admission-retry-'+attempt
assert not (root/(label+'.json')).exists()
with (root/(label+'.log')).open('x') as log:
 child=subprocess.Popen(['/opt/cerebras/venv/bin/python','-u','supervise.py'],cwd=root,
   stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
r=dict(pid=child.pid,root=str(root),prior_rejection_sha256=hashlib.sha256(prior.encode()).hexdigest(),
       no_prior_physical_job=True,source_unchanged=True)
with (root/(label+'.json')).open('x') as out:json.dump(r,out);out.write(chr(10));out.flush();os.fsync(out.fileno())
print(json.dumps(r))
'''
r=subprocess.run(['sh','/path/to/alcf-session.sh','host','python3 -c '+shlex.quote(script)],capture_output=True,text=True,check=True,timeout=30)
out=Path(__file__).resolve().parents[1]/'evidence'/a.name
(out/('admission-retry-'+a.attempt+'.json')).write_text(r.stdout);print(r.stdout)
