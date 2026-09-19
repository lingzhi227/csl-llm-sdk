"""Run one admitted child and retain protocol and artifact outcomes separately."""
from pathlib import Path
import hashlib,json,os,resource,signal,subprocess,sys,time
ROOT=Path(__file__).resolve().parent

def atomic(name,value):
    raw=(json.dumps(value,indent=2)+'\n').encode();assert len(raw)<=1048576
    temporary=ROOT/(name+'.tmp')
    with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    temporary.replace(ROOT/name)

def captured_result(case_root=None):
    import numpy as np
    from check_lifecycle import initialized,check_protocol,stable,check_overlap
    case_root=ROOT/'cases/lifecycle' if case_root is None else Path(case_root)
    directory=case_root/'evidence';path=case_root/'result.json'
    assert path.stat().st_size<=65536 and not path.is_symlink()
    result=json.loads(path.read_bytes());receipts=result['captures']
    assert result['hardware'] is False and result['PEs']==3
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

def main():
    from runtime_binding import verify_runtime_binding
    from runtime_source_gate import verify,PROFILE
    verify();binding=verify_runtime_binding()
    assert sorted(os.sched_getaffinity(0))==[0]
    assert PROFILE['seconds']==60 and PROFILE['case_seconds']==45 and PROFILE['idle_seconds']==30 and PROFILE['cases']==1
    assert not (ROOT/'suite-result.json').exists() and not (ROOT/'cases').exists()
    resource.setrlimit(resource.RLIMIT_AS,(4<<30,4<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    began=time.monotonic();child=None;result=None;failure=None;artifact=None;artifact_error=None;trace=None;trace_error=None
    try:
        work=ROOT/'simulator-work/lifecycle';work.mkdir(parents=True,exist_ok=False)
        with (ROOT/'case-0.log').open('xb') as stream:
            started=time.monotonic();timed_out=False
            proc=subprocess.Popen([sys.executable,'-B',str(ROOT/'driver.py')],
                cwd=work,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
            atomic('active-case.json',dict(case=0,pid=proc.pid,case_seconds=45))
            try:proc.wait(timeout=max(0,45-(time.monotonic()-started)))
            except subprocess.TimeoutExpired:
                timed_out=True;os.killpg(proc.pid,signal.SIGTERM)
                try:proc.wait(timeout=1)
                except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait(timeout=1)
            finally:
                child=dict(case=0,pid=proc.pid,returncode=proc.poll(),reaped=proc.poll() is not None,
                           timed_out=timed_out,seconds=time.monotonic()-started)
                atomic('case-exit-0.json',child)
        assert not timed_out,'One child exceeded45s; no retry'
        assert child['returncode']==0,'Actual driver failed; preserve result and raw captures'
        result=captured_result()
    except BaseException as exc:failure=dict(type=type(exc).__name__,message=str(exc)[:512])
    finally:
        # The child is reaped before reading any retained simulator output.
        try:
            from trace_inventory import inventory
            trace=inventory(ROOT/'simulator-work/lifecycle')
            trace['observation_phase']='after_driver_reap_before_unit_cleanup'
            trace['whole_unit_quiescence_verified']=False
            atomic('trace-inventory.json',trace)
            if trace['violations']:trace_error=dict(type='TraceBoundViolation',message=';'.join(trace['violations'])[:512])
        except BaseException as exc:trace_error=dict(type=type(exc).__name__,message=str(exc)[:512])
        # Preserve generated observations even after native abort or trace-cap failure.
        try:
            verify();artifact=verify_runtime_binding(post_runtime=True)
            atomic('artifact-after-case-0.json',artifact)
        except BaseException as exc:artifact_error=dict(type=type(exc).__name__,message=str(exc)[:512])
    passed=failure is None and artifact_error is None and trace_error is None
    atomic('suite-result.json',dict(passed=passed,cases=[] if result is None else [result],
        children=[] if child is None else [child],error=failure,artifact_error=artifact_error,
        artifact_observation=artifact,trace_inventory='trace-inventory.json' if trace is not None else None,trace_error=trace_error,compiled_binding=binding,seconds=time.monotonic()-began,
        qualification='traced_bidirectional_qualification_pending_independent_review' if passed else 'failed',
        compile_invocations=0,hardware_invocations=0,original_neural_epochs=0,
        scope='Full original ninepacket200word lifecycle under tracing; no untraced012 repair, physical136 or model qualification'))
    if not passed:raise RuntimeError(failure or artifact_error or trace_error)

if __name__=='__main__':main()
