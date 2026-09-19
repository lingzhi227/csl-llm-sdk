"""Re-read saved actual arrays with the original strict012/015 parent checks."""
from pathlib import Path
import hashlib,json
ROOT=Path(__file__).resolve().parent

def captured_result(case_root=None):
    import numpy as np
    from check_lifecycle import initialized,check_protocol,stable,check_overlap
    case_root=ROOT/'cases/lifecycle' if case_root is None else Path(case_root)
    directory=case_root/'evidence';path=case_root/'result.json'
    assert path.stat().st_size<=65536 and not path.is_symlink()
    result=json.loads(path.read_bytes());receipts=result['captures']
    assert result['hardware'] is True and result['PEs']==3
    assert result['packets']==9 and result['application_words']==200 and result['original_neural_epochs']==0
    assert len(receipts)==result['copies']<=17 and result['launches']<=11 and result['H2D']==0
    records={};total=0;host_bytes=0
    from capture_store import SPECS
    allowed=SPECS
    for item in receipts:
        relative=Path(item['path']);assert not relative.is_absolute() and '..' not in relative.parts
        q=case_root/relative;assert not any(v.is_symlink() for v in (q,*q.parents))
        assert q.parent==directory and q.suffix=='.npz' and q.stem not in records
        assert q.stem in allowed and item['shape']==list(allowed[q.stem])
        raw=q.read_bytes();assert len(raw)==item['bytes']<=8192
        assert hashlib.sha256(raw).hexdigest()==item['sha256'];total+=len(raw)
        with np.load(q,allow_pickle=False) as archive:
            assert archive.files==['words'];words=archive['words']
            shape=allowed[q.stem];assert words.dtype==np.uint32 and words.shape==(shape[0]*shape[1],)
            host_bytes+=words.nbytes;records[q.stem]=words.reshape(shape).tolist()
    assert total<=1048576 and host_bytes==result['host_bytes']<=18120
    assert {q.name for q in directory.iterdir()}=={Path(r['path']).name for r in receipts}|{'journal.jsonl'}
    assert {q.name for q in case_root.iterdir()}=={'evidence','result.json','capture-progress.json'}
    progress=json.loads((case_root/'capture-progress.json').read_bytes());assert progress['captures']==receipts
    assert result['status']=='passed' and result['normal_stop'] is True
    assert set(result['checks'])=={'protocol','stability','overlap'}
    assert all(row['passed'] is True for row in result['checks'].values())
    polls=[name for name in records if name.startswith('state-')]
    assert polls==['state-'+str(i) for i in range(len(polls))] and 1<=len(polls)<=8
    assert result['launches']==len(polls)+3 and len(receipts)==len(polls)+9
    initialized(records['initial-state'],records['routing'])
    check_protocol(records['final-state'],records['observed'],records['tx-proof'],records['rx-banks'],records['live-frames'],records['live-sources'],records['row-archive'])
    stable(records[polls[-1]],records['final-state']);check_overlap(records['final-state'])
    return result
