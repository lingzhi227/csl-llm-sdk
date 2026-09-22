"""Independently reread durable actual arrays; no SDK or device interaction."""
from pathlib import Path
import hashlib,json
import numpy as np
from capture_store import SPECS
from check_capture import check
from check_fault_capture import inspect_faults

def validate(root,require_complete=True):
    root=Path(root);result=json.loads((root/'result.json').read_bytes())
    captures=result['captures'];assert len(captures)==result['copies']<=15
    labels=[];arrays={};host_bytes=0
    for receipt in captures:
        path=Path(receipt['path']);label=path.stem
        assert path==Path('evidence')/(label+'.npz') and label in SPECS and label not in labels
        labels.append(label);q=root/path;raw=q.read_bytes()
        assert not q.is_symlink() and len(raw)==receipt['bytes']<=131072 and hashlib.sha256(raw).hexdigest()==receipt['sha256']
        with np.load(q,allow_pickle=False) as archive:
            assert archive.files==['words'];value=archive['words']
            assert value.dtype==np.uint32 and value.shape==(SPECS[label][0]*SPECS[label][1],)
            arrays[label]=value.reshape(SPECS[label]).tolist();host_bytes+=value.nbytes
        assert receipt['shape']==list(SPECS[label])
    assert host_bytes==result['host_slot_bytes']<=283776
    assert sum(v['bytes'] for v in captures)<=1048576
    polls=[name for name in labels if name.startswith('progress-')]
    assert polls==[f'progress-{i:02d}' for i in range(len(polls))] and len(polls)<=8
    if not require_complete:return dict(prefix_valid=True,copies=len(captures),host_bytes=host_bytes)
    assert result['status']=='passed' and result['normal_stop'] and result['hardware'] and result['launches']==2
    assert 1<=len(polls)<=8 and result['copies']==len(polls)+7
    assert labels==['initial',*polls,'origin','peer','producers','final-status','first-fault','first-fault-final']
    initial=arrays['initial']
    for y in range(2):
        for x in range(178):
            role=4 if y==1 else 1 if x==0 else 2 if x==2 else 3 if 40<=x<176 else 0
            assert initial[y*178+x]==[1,role,0,x]
    summary=check(arrays['final-status'],arrays['origin'][0],arrays['peer'][0],arrays['producers'])
    fault=inspect_faults(arrays['first-fault'],arrays['first-fault-final'])
    assert summary==result['summary'] and fault==result['fault_report']
    assert fault['capture_stable'] and not fault['latched_faults'] and not fault['incomplete_columns']
    return dict(passed=True,copies=len(captures),host_bytes=host_bytes,strict650_and_zero_faults=True)
