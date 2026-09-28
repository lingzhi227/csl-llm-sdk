"""Dispatch a bounded read-only original layer0/1 initialization audit."""
import argparse,hashlib,io,json,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def main():
    p=argparse.ArgumentParser();p.add_argument('attempt');a=p.parse_args()
    if len(a.attempt)!=3 or not a.attempt.isdigit():raise ValueError('Attempt')
    name='layer-weight-audit-'+a.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    remote='/srv/model-storage/qwen38-singlewse/runs/'+name
    active=subprocess.run(['ssh','workstation','systemctl --user list-units --type=service --state=active,activating,deactivating --no-legend qwen38-single-*'],capture_output=True,text=True,check=True,timeout=20)
    if active.stdout.strip():raise ValueError('Active workstation owner: '+active.stdout)
    mapping={n:n for n in ['configs/tensors.json','configs/hub.json','runtime/weights.py','runtime/source_gate.py',
                          'performance/runtime/layer_weights.py','performance/spatial/layer_schedule.py']}
    mapping.update({'execute.py':'performance/runtime/audit_layer_weights.py',
                    'selected-layer0-1.json':'performance/evidence/layer-native-schedule-001/selected-layer0-1.json'})
    files={dest:(ROOT/source).read_bytes() for dest,source in mapping.items()}
    files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    payload=io.BytesIO()
    with tarfile.open(fileobj=payload,mode='w') as archive:
        for n,b in files.items():
            item=tarfile.TarInfo(n);item.size=len(b);archive.addfile(item,io.BytesIO(b))
    subprocess.run(['ssh','workstation','mkdir '+shlex.quote(remote)+' && tar -xf - -C '+shlex.quote(remote)],input=payload.getvalue(),check=True,timeout=30)
    unit='qwen38-single-'+name
    cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=512M','--property=MemorySwapMax=0','--property=TasksMax=16',
         '--property=CPUQuota=100%','--property=AllowedCPUs=6','--property=RuntimeMaxSec=180','--property=TimeoutStopSec=5',
         '--property=KillMode=control-group','--property=LimitFSIZE=8388608','--property=LimitCORE=0','--working-directory='+remote,
         '--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1','--setenv=PYTHONDONTWRITEBYTECODE=1',
         '/usr/bin/taskset','--cpu-list','6','/usr/bin/flock','-n','/srv/cerebras-workstation/heavy.lock','/usr/bin/python3','execute.py']
    subprocess.run(['ssh','workstation',shlex.join(cmd)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json'])
    (out/'dispatch.json').write_text(json.dumps(dict(remote=remote,unit=unit,physical=False),indent=2)+'\n')
    print(json.dumps(dict(remote=remote,unit=unit,physical=False)))


if __name__=='__main__':main()
