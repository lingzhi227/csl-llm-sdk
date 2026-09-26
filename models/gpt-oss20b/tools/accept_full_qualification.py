"""Bind a passed independent CPU qualification to released physical evidence.

Does not change strict-reference result.json or previous failures. Creates an
explicitly numerically-qualified COMPLETE receipt after rechecking hashes.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SESSION = ['sh', '/path/to/alcf-session.sh', 'host']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name'); p.add_argument('--qualification-run', required=True)
    a = p.parse_args()
    assert re.fullmatch(r'full-model-hw-[0-9]{3}', a.name)
    assert re.fullmatch(r'full-qualification-[0-9]{3}', a.qualification_run)
    path = '/srv/model-storage/gpt-oss20b/runs/'+a.qualification_run+'/QUALIFICATION.json'
    raw = subprocess.run(['ssh','workstation','cat '+shlex.quote(path)],
                         capture_output=True, check=True, timeout=30).stdout
    assert len(raw) < 2<<20
    report = json.loads(raw)
    assert report['passed'] and report['physical_tokens'] == [13225,11,5922]
    assert len(report['layers']) == 48 and all(r['passed'] for r in report['layers'])
    assert report['head']['passed']
    contract = json.loads((ROOT/'evidence'/a.name/'numerical-contract.json').read_text())
    assert not contract['capture_observed']
    for key, name in [('contract_sha256','docs/NUMERICAL-QUALIFICATION.md'),
                      ('verifier_sha256','reference/qualify_full_capture.py')]:
        digest = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
        assert report[key] == digest
        assert contract['files'][name] == digest
    remote = '/srv/gpt-oss20b-hardware/'+a.name
    code = '''import datetime,hashlib,json,pathlib,sys
root=pathlib.Path(REMOTE);sys.path.insert(0,str(root))
from source_gate import verify
from runtime.lifecycle import query,TERMINAL
from runtime.store import atomic_json
verify()
r=json.loads(sys.stdin.buffer.read())
assert r['passed'] and not (root/'COMPLETE.json').exists()
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
for name,value in r['physical_evidence'].items():
 assert pathlib.Path(name).name==name
 assert digest(root/name)==value, name
capture=json.loads((root/'PHYSICAL_CAPTURE_COMPLETE.json').read_text())
result=json.loads((root/'result.json').read_text())
assert capture['all_owned_jobs_released'] and result['normal_stop'] and result['all_weights_retained']
systems=query('systems')['items']
for stage in capture['stages']:
 assert stage['exit_code']==0 and not stage['cleanup_errors'] and not stage['error']
 for job in stage['jobs']:
  jid=job['id'];current=query('job',jid)
  assert current['status']['phase']=='SUCCEEDED' and job['released']
  assert not any(jid in {str(v).split('/')[-1] for v in [s.get('jobId',''),*s.get('jobIds',[])]} for s in systems)
atomic_json(root/'QUALIFICATION.json',r)
complete=dict(scope=r['scope'],full_model=True,model=result['model'],revision=result['revision'],
 tokens=capture['tokens'],west_to_east=True,all_owned_jobs_released=True,
 numerical_qualification_passed=True,strict_original_reference_passed=result['strict_original_reference_passed'],
 qualification_sha256=digest(root/'QUALIFICATION.json'),stages=capture['stages'],
 acceptance_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
atomic_json(root/'COMPLETE.json',complete)
atomic_json(root/'ACTIVE.json',dict(phase='complete',active_jobs=[]))
print(json.dumps(complete))
'''.replace('REMOTE',repr(remote))
    command = 'cd '+shlex.quote(remote)+' && /opt/cerebras/venv/bin/python -c '+shlex.quote(code)
    completed = subprocess.run(SESSION+[command],input=raw,capture_output=True,check=True,timeout=60)
    print(completed.stdout.decode())


if __name__ == '__main__': main()
