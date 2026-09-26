"""Read compact progress from an owned attempt; never allocate or cancel jobs."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("name")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-z][a-z0-9-]+-hw-[0-9]{3}", args.name):
        raise ValueError("Invalid attempt name")
    remote = "/srv/gpt-oss20b-hardware/" + args.name
    script = "root_name=" + repr(remote) + "\n" + '''
import json,re,subprocess,time
from pathlib import Path
root=Path(root_name)
def small(name):
    path=root/name
    if not path.exists():return None
    if path.stat().st_size>2<<20:raise ValueError('Receipt exceeds bound: '+name)
    return json.loads(path.read_text())
result=dict(time_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),root=str(root),
            active=small('ACTIVE.json'),failure=small('FAILURE.json'),
            pre_run_admission=small('PRE_RUN_ADMISSION.json'),jobs=[])
complete=small('COMPLETE.json')
if complete:
    result['complete']={k:complete[k] for k in ('scope','full_model','all_owned_jobs_released') if k in complete}
for stage in ('compile','run'):
    path=root/(stage+'.jobs')
    if path.exists():
        if path.stat().st_size>65536:raise ValueError('Job capture exceeds bound')
        for job_id in sorted(set(re.findall(r'(?m)^(wsjob-[a-z0-9]+)$',path.read_text()))):
            job=json.loads(subprocess.check_output(['csctl','get','job',job_id,'-o','json'],text=True,timeout=15))
            result['jobs'].append(dict(stage=stage,id=job_id,phase=job['status']['phase'],
              execution_time=job['status']['executionTime'],completion_time=job['status']['completionTime'],
              errors=job['status']['errorReplicas']))
    audit=small(stage+'-audit.json')
    if audit:result[stage+'_audit']=audit
for name in ('supervisor.log','compile.log','run.log'):
    path=root/name
    if path.exists():
        with path.open('rb') as stream:
            stream.seek(max(0,path.stat().st_size-4000))
            result[name+'_tail']=stream.read(4000).decode(errors='replace')
print(json.dumps(result))
'''
    proc = subprocess.run(
        ["sh", "/path/to/alcf-session.sh", "host", "python3 -c " + shlex.quote(script)],
        capture_output=True, text=True, check=True, timeout=50,
    )
    result = json.loads(proc.stdout)
    output = Path(__file__).resolve().parents[1] / "evidence" / args.name
    if output.is_dir():
        (output / "latest-progress.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
