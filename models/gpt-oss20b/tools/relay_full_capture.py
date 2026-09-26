"""Relay released physical captures ALCF -> workstation without local payload files."""
import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
SESSION = ['sh', '/path/to/alcf-session.sh', 'host']
NAMES = ['PHYSICAL_CAPTURE_COMPLETE.json', 'result.json', 'source-manifest.json',
         'PRE_RUN_ADMISSION.json', 'weight-load.json', 'weight-retention.json',
         'observations.json', 'actual-1.npz', 'actual-2.npz']


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('name'); a = p.parse_args()
    assert re.fullmatch(r'full-model-hw-[0-9]{3}', a.name)
    source = '/srv/gpt-oss20b-hardware/'+a.name
    target = '/srv/model-storage/gpt-oss20b/runs/'+a.name+'-capture'
    code = '''import json,pathlib
p=pathlib.Path(SOURCE)
r=json.loads((p/'PHYSICAL_CAPTURE_COMPLETE.json').read_text())
assert r['all_owned_jobs_released']
assert all((p/n).is_file() and (p/n).stat().st_size < 64<<20 for n in NAMES)
print(json.dumps(dict(tokens=r['tokens'],all_owned_jobs_released=True)))
'''.replace('SOURCE', repr(source)).replace('NAMES', repr(NAMES))
    status = subprocess.run(SESSION+['python3 -c '+shlex.quote(code)],
                            capture_output=True, text=True, check=True, timeout=30)
    reader = subprocess.Popen(SESSION+['tar -cf - -C '+shlex.quote(source)+' '+shlex.join(NAMES)],
                              stdout=subprocess.PIPE)
    try:
        subprocess.run(['ssh', 'workstation', 'mkdir '+shlex.quote(target)+
                        ' && tar -xf - -C '+shlex.quote(target)],
                       stdin=reader.stdout, check=True, timeout=60)
    finally:
        reader.stdout.close()
        if reader.wait(timeout=15): raise RuntimeError('ALCF capture archive transfer failed')
    receipt = dict(source=source, workstation=target, files=NAMES,
                   local_payload_files_created=False, physical=json.loads(status.stdout))
    (ROOT/'evidence'/a.name/'capture-relay.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__ == '__main__': main()
