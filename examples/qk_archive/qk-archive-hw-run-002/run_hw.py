"""Two original-kernel reset generations on real CS3 with immediate durable raw captures."""
import json
import hashlib
import os
from pathlib import Path
import time
import numpy as np
from source_gate import ROOT,PROFILE,verify
from compiled_binding import preflight
from hw00.job_capture import install
from hw00.store import atomic_json
import logging
from capture_store import save_capture, save_progress
from check_capture import check_epoch, check_negative, require

START = time.monotonic()
SEQUENCE = 0


def event(kind, **fields):
    global SEQUENCE
    require(SEQUENCE < 96, 'Finite journal')
    raw = (json.dumps(dict(sequence=SEQUENCE, seconds=time.monotonic()-START, kind=kind, **fields))+'\n').encode()
    with (ROOT/'journal.jsonl').open('ab') as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    SEQUENCE += 1


def main():
    require(sorted(os.sched_getaffinity(0)) == [0], 'CPU0 required')
    verify()
    artifact,binding=preflight()
    install(os.environ['HW00_JOB_CAPTURE'])
    logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder
    require(not any((ROOT/name).exists() for name in ('captures','journal.jsonl','result.json','progress.json')),
            'One fresh runtime attempt')
    class Runtime:
        def __init__(self):
            self.closed=False
            self.runtime=SdkRuntime(str(artifact),simulator=False,disable_version_check=True,
                resource_cpu=PROFILE['worker_cpu_millicores'],resource_mem=PROFILE['worker_memory_bytes'])
            self.runtime.__enter__()
        def __getattr__(self,name):
            return getattr(self.runtime,name)
        def stop(self):
            if not self.closed:
                self.runtime.__exit__(None,None,None)
                self.closed=True

    event('load_enter',adapter='physical SdkRuntime context entry')
    runner = Runtime()
    event('load_exit',physical_context_active=True)
    event('run_exit',started_by_context_entry=True)
    result = dict(epochs=[], scope='Physical CS3 synthetic two-reset alias/archive fixture; no original model inference')
    capture_files = {}
    counts = dict(copies=0, launches=0, host_slot_bytes=0)
    error = None
    stopped = False
    previous = None
    try:
        symbols = {name: runner.get_id(name) for name in ('evidence','archive','reference','workspace','output')}

        def copy(label, symbol, x, width, count, half=False):
            value = np.zeros(width*count, np.uint32)
            event('copy_enter', label=label, symbol=symbol, box=[x,0,width,1], count=count,
                  host_bytes=int(value.nbytes), half=half)
            runner.memcpy_d2h(value, symbols[symbol], x, 0, width, 1, count,
                data_type=MemcpyDataType.MEMCPY_16BIT if half else MemcpyDataType.MEMCPY_32BIT,
                streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            answer = value.reshape(width,count).tolist()
            capture_files[label] = save_capture(ROOT,label,answer)
            counts['copies'] += 1
            counts['host_slot_bytes'] += int(value.nbytes)
            event('copy_exit',label=label)
            save_progress(ROOT,capture_files,result['epochs'],counts,SEQUENCE)
            return answer if width>1 else answer[0]

        def launch(name):
            event('launch_enter',name=name)
            runner.launch(name, nonblock=False)
            counts['launches'] += 1
            event('launch_exit',name=name)

        launch('initialize')
        for epoch in (1,2):
            # Prior generation's every raw capture and completed progress are fsynced first.
            launch('prepare')
            launch('compute')
            evidence = copy(f'epoch{epoch}_evidence','evidence',0,4,64)
            archive = copy(f'epoch{epoch}_archive','archive',1,1,960)
            reference = copy(f'epoch{epoch}_reference','reference',3,1,960)
            workspace = copy(f'epoch{epoch}_workspace','workspace',0,1,1280)
            output = copy(f'epoch{epoch}_output','output',0,1,256,True)
            baseline = copy(f'epoch{epoch}_baseline_output','output',3,1,256,True)
            result['epochs'].append(check_epoch(evidence,archive,reference,workspace,output,baseline,epoch))
            if previous is not None:
                require(archive[704:] != previous[0][704:] and output != previous[1],
                        'Changed reset input changes normalized QK and final output')
            previous = (archive,output)
            save_progress(ROOT,capture_files,result['epochs'],counts,SEQUENCE)
        launch('negative_checks')
        result['negative'] = check_negative(copy('negative_evidence','evidence',0,4,64),evidence)
        require(counts == dict(copies=13,launches=6,host_slot_bytes=32768), 'Exact operation/copy contract')
    except BaseException as exc:
        error = exc
        event('failure',message=str(exc)[:512])
    finally:
        event('stop_enter')
        try:
            runner.stop()
            stopped = True
            event('stop_exit')
        except BaseException as exc:
            if error is None:
                error = exc
        result.update(status='passed' if error is None else 'failed',normal_stop=stopped,
                      seconds=time.monotonic()-START,journal_events=SEQUENCE,**counts,
                      message=None if error is None else str(error)[:512],capture_files=capture_files)
        raw = (json.dumps(result,indent=2)+'\n').encode()
        require(len(raw)<=65536,'Bounded summary')
        with (ROOT/'result.json').open('xb') as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    if error is not None:
        raise error
    atomic_json(ROOT/'physical.json',dict(simulator=False,runtime_context_exited=runner.closed,
        artifact_sha256=binding['artifact']['sha256'],application=[4,1],full_model=False,
        scope='Actual original-kernel alias/archive equivalence on a synthetic four-PE fixture'))
    verify()


if __name__ == '__main__':
    main()
