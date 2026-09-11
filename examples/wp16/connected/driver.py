"""Exact connected three-generation driver; scoped admission required for execution."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time
import numpy as np
from checks import Checks, item_key, require
from qwen38.wp16_reuse_layout import ARRAYS, FLOAT32, SCOPE, budget, sequence
from qwen38.wp16_reuse_protocol import identity, scope, descriptor, COMPLETION_MASKS

PREPARATION_SHA='3a3f204e32a89db8f60a01c9d373e4b364f61e010cabbb1ea7de2eaba58dceb0'
SOURCE_MANIFEST_SHA='15775313a35edf96db27d33cef155ce5814becdc360002be251e0630a6c444fb'
ORACLE_SHA='1e915d5c9bca30f6a3a8e9d5529201b17620b0c6c3587ba5b7d0835ff9886516'

def sha(raw):return hashlib.sha256(raw).hexdigest()


def load_prepared(work):
    from qwen38.wp16_reuse_source import validate_arrays
    folder=work/'prepared';raw=(folder/'preparation.json').read_bytes()
    require(len(raw)<=65536 and sha(raw)==PREPARATION_SHA,'Exact independently accepted connected source receipt')
    receipt=json.loads(raw)
    require(receipt['status']=='connected_source_prepared' and receipt['source_manifest_sha256']==SOURCE_MANIFEST_SHA and receipt['sensitivity']['passed'],'Qualified connected source and sensitivity')
    require(receipt['generation_input_contraction']==[[1,0,1],[2,1,2],[3,3,3]] and
        receipt['projection_input_range']==[0,112] and receipt['channel_range']==receipt['output_range']==[0,128],
        'Exact connected input/contraction/range identities')
    require(receipt['source_uses_CPU_or_SDK_observations'] is False and receipt['original_model_rerun'] is False,'Independent connected source')
    require(set(receipt['files'])=={'weights/gate_proj.bf16','weights/up_proj.bf16','weights/down_proj.bf16','hidden.bf16','oracle.npz'},'Five exact connected prepared inputs')
    verified={}
    for name,expected in receipt['files'].items():
        raw=(folder/name).read_bytes()
        require(len(raw)<=131072 and len(raw)==expected['bytes'] and sha(raw)==expected['sha256'],'Exact prepared file identity: '+name)
        verified[name]=raw
    require(sha(verified['oracle.npz'])==ORACLE_SHA,'Exact accepted connected oracle')
    weights={role:np.frombuffer(verified['weights/'+role+'.bf16'],dtype='<u2').reshape(128,128 if role=='down_proj' else 112)
        for role in ('gate_proj','up_proj','down_proj')}
    hidden=np.frombuffer(verified['hidden.bf16'],dtype='<u2').reshape(3,112)
    with np.load(io.BytesIO(verified['oracle.npz']),allow_pickle=False) as archive:
        require(set(archive.files)==set(receipt['arrays']),'Exact connected source array keys')
        oracle={name:archive[name] for name in archive.files}
    validate_arrays(oracle)
    for name,array in oracle.items():
        info=receipt['arrays'][name]
        require(list(array.shape)==info['shape'] and str(array.dtype)==info['dtype'] and
            array.nbytes==info['bytes'] and sha(array.tobytes())==info['sha256'],'Exact source array shape/dtype/payload')
    return weights,hidden,oracle


def upload_values(item,weights,hidden):
    name,pe,phase,tile=(item[k] for k in ('name','pe','phase','tile'));g=item['generation'];count=item['count']
    if name=='control':
        triple=(0,0xffffffff,0) if g==0 else identity(g)
        total=112 if pe<2 else 128;width=112 if pe<2 else 64;offset=0 if tile is None else tile*width
        return np.array([*triple,total,0 if tile is None else tile,offset,0 if tile is None else width,0],dtype=np.uint32)
    if name=='generation_control':return np.array([*scope(g),g-1,1,0],dtype=np.uint32)
    if name=='handoff_control':return np.array(descriptor(g,int(phase[-1])),dtype=np.uint32)
    if name=='release_control':return np.array([*scope(g),COMPLETION_MASKS[pe],g,0],dtype=np.uint32)
    if name=='weights_storage':
        require(pe in (0,1,3) and tile is not None,'Original selected numeric weight buffers only')
        role=('gate_proj','up_proj','','down_proj')[pe];width=112 if pe<2 else 64;offset=tile*width
        require((pe<2 and g==0 and tile==0) or (pe==3 and g in (1,2,3) and tile in (0,1)),'Exact resident/down weight scope')
        result=np.zeros(count,dtype=np.uint32);result[0],result[-1]=0xa55a,0x5aa5
        result[1:-1].reshape(width,128)[:]=weights[role][:,offset:offset+width].T
        return result
    if name=='input_storage':
        require(pe<2 and tile==0 and g in (1,2,3),'No host intermediate consumer input')
        result=np.zeros(count,dtype=np.float32);result[0],result[-1]=77.5,-88.25
        result.view(np.uint32)[1:-1]=hidden[g-1].astype(np.uint32)<<16
        return result
    raise ValueError('Undeclared connected upload: '+name)

class Evidence:
    """Append bounded journal; save each readback group once, including failed gates."""
    def __init__(self, root):
        self.root = root
        root.mkdir()
        self.started = time.monotonic()
        self.records = 0
        self.journal_bytes = 0
        self.observation_bytes = 0
        self.group = None
        self.pending = {}
        self.files = {}

    def event(self, kind, **fields):
        require(self.records < 2000, 'Diagnostic journal record bound')
        raw = (json.dumps(dict(record=self.records, seconds=time.monotonic() - self.started,
                               kind=kind, **fields), sort_keys=True) + '\n').encode()
        require(len(raw) <= 4096, 'Diagnostic journal record size bound')
        require(self.journal_bytes + len(raw) <= 2097152, 'Diagnostic total journal byte bound')
        with (self.root / 'journal.jsonl').open('ab') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.records += 1
        self.journal_bytes += len(raw)

    def write(self, name, raw, limit):
        require(len(raw) <= limit, 'Bounded evidence file: ' + name)
        with (self.root / name).open('xb') as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        self.files[name] = dict(bytes=len(raw), sha256=sha(raw))

    def observation(self, item, array):
        group = (item['generation'], item['input_index'], item['generation'], item['phase'], item['tile'])
        if group != self.group:
            self.flush()
            self.group = group
        tag = item_key(item)
        require(tag not in self.pending, 'Unique readback group key')
        self.pending[tag] = array

    def flush(self):
        if not self.pending:
            return
        buffer = io.BytesIO()
        np.savez(buffer, **self.pending)
        raw = buffer.getvalue()
        self.observation_bytes += len(raw)
        require(self.observation_bytes <= 2097152, 'All private observation archive budget')
        generation, input_index, contraction, phase, tile = self.group
        name = 'g'+str(generation)+'_i'+str(input_index)+'_c'+str(contraction)+'_'+phase+('_tile'+str(tile) if tile is not None else '')+'.npz'
        self.write(name, raw, 1048576)
        self.pending = {}

    def finish(self, report):
        report.update(evidence_files=self.files, journal_records=self.records,
                      journal_bytes=self.journal_bytes,
                      private_observation_archive_bytes=self.observation_bytes)
        self.write('result.json', (json.dumps(report, indent=2) + '\n').encode(), 1048576)


def flush_and_stop(evidence, runner, report, primary):
    """Evidence failures cannot bypass stop or replace the original failure."""
    def failure(name, exc):
        nonlocal primary
        report[name] = dict(error_type=type(exc).__name__, message=str(exc))
        if primary is None:
            primary = exc

    def event(kind, **details):
        try:
            evidence.event(kind, **details)
        except BaseException as exc:
            failure('cleanup_journal_error', exc)

    try:
        evidence.flush()
    except BaseException as exc:
        failure('pre_stop_flush_error', exc)
    try:
        before = dict(report, status='checks_passed_awaiting_stop' if primary is None else 'failed_awaiting_stop',
                      evidence_files=evidence.files, journal_records=evidence.records)
        evidence.write('before-stop.json', (json.dumps(before, indent=2) + '\n').encode(), 1048576)
    except BaseException as exc:
        failure('pre_stop_summary_error', exc)
    stopped = False
    if runner is not None:
        event('stop_enter')
        try:
            started = time.monotonic()
            runner.stop()
            seconds = time.monotonic() - started
            stopped = True
        except BaseException as exc:
            failure('stop_error', exc)
        else:
            event('stop_exit', api_seconds=seconds)
    return primary, stopped


def main():
    from qwen38.wp16_reuse_sdk_admission import PROFILE, verify
    from qwen38.wp16_affinity import current
    work = Path.cwd()
    digest = verify(work)
    current(PROFILE['cpu_affinity'])
    require(os.environ.get('WP16_SDK_STAGE') == 'simulate', 'Scoped simulator entry required')
    require((work / 'runtime-admission-simulate.json').is_file() and
            (work / 'container-admission-simulate.json').is_file(), 'Outer and inner pre-import admission required')
    operations = sequence()
    require(json.loads((work / 'physical-sequence.json').read_bytes()) == operations, 'Frozen exact transaction schedule')
    require(json.loads((work / 'resource-plan.json').read_bytes()) == budget(), 'Frozen physical ledger')
    weights, hidden, oracle = load_prepared(work)
    checks = Checks(oracle, weights)
    evidence = Evidence(work / 'device-evidence')
    runner = None
    primary = None
    stopped = False
    counts = dict(copy_calls=0, launch_calls=0, h2d=0, d2h=0, host_bytes=0, native_bytes=0)
    report = dict(status='failed', scope=SCOPE, source_manifest_sha256=digest,
                  original_input_indices=[0,1,3], generation_input_contraction=[[1,0,1],[2,1,2],[3,3,3]], full17408_down_output=False,
                  full_model_or_original_MLP_forward=False, physical_WSE=False,
                  resident_weight_bitwise_equality_continuous_coverage=False,
                  weight_evidence='Resident projection payload equality before generation1/after generation3 only; current per-generation guards/source checks; complete down tile and final retention reads')

    def call(label, action, **metadata):
        evidence.event(label + '_enter', **metadata)
        started = time.monotonic()
        result = action()
        seconds = time.monotonic() - started
        evidence.event(label + '_exit', api_seconds=seconds, **metadata)
        return result

    try:
        evidence.event('sdk_import_enter', affinity=current())
        from cerebras.sdk.runtime.sdkruntimepybind import (
            SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform)
        evidence.event('sdk_import_exit', affinity=current())
        evidence.event('construct_enter')
        constructed = time.monotonic()
        runner = SdkRuntime('out', get_platform(None,
            SimfabConfig(suppress_trace=True, num_threads=1, dump_core=False), SdkTarget.WSE3))
        evidence.event('construct_exit', api_seconds=time.monotonic()-constructed)
        symbols = {name: call('get_id', lambda name=name: runner.get_id(name), name=name) for name in ARRAYS}
        require(len(set(symbols.values())) == len(ARRAYS), 'Distinct exported symbol IDs')
        report['symbols'] = {name: int(value) for name, value in symbols.items()}
        call('load', runner.load)
        call('run', runner.run)
        for item in operations:
            checks.before_operation(item)
            metadata = dict(operation=item,generation=item['generation'],input_index=item['input_index'],contraction_id=item['generation'])
            if item['kind'] == 'launch':
                call('launch', lambda: runner.launch(item['name'], nonblock=False), **metadata)
                counts['launch_calls'] += 1
                checks.after_operation(item)
                continue
            name, pe, count, direction = (item[k] for k in ('name', 'pe', 'count', 'direction'))
            dtype = np.float32 if name in FLOAT32 else np.uint32
            values = upload_values(item, weights, hidden) if direction == 'h2d' else np.zeros(count, dtype=dtype)
            require(values.dtype == dtype and values.shape == (count,) and values.flags.c_contiguous and
                    values.nbytes == item['physical_host_bytes'] and values.nbytes <= 65536, 'Exact physical host buffer')
            if direction == 'h2d':
                checks.upload(item, values)
            metadata.update(symbol_id=int(symbols[name]), dtype=str(values.dtype), shape=list(values.shape),
                            input_sha256=sha(values.tobytes()) if direction == 'h2d' else None)
            evidence.event('copy_enter', **metadata)
            params = dict(data_type=MemcpyDataType.MEMCPY_16BIT if item['bits'] == 16 else MemcpyDataType.MEMCPY_32BIT,
                          streaming=False, order=MemcpyOrder.ROW_MAJOR, nonblock=False)
            if direction == 'h2d':
                started = time.monotonic()
                runner.memcpy_h2d(symbols[name], values, pe, 0, 1, 1, count, **params)
                api_seconds = time.monotonic() - started
            else:
                started = time.monotonic()
                runner.memcpy_d2h(values, symbols[name], pe, 0, 1, 1, count, **params)
                api_seconds = time.monotonic() - started
                evidence.observation(item, values)
            evidence.event('copy_exit', api_seconds=api_seconds, output_sha256=sha(values.tobytes()), **metadata)
            counts['copy_calls'] += 1; counts[direction] += 1
            counts['host_bytes'] += item['physical_host_bytes']; counts['native_bytes'] += item['native_device_bytes']
            if direction == 'd2h':
                # MEMCPY16 defines the low16 payload in each32-bit host slot.
                # Preserve raw slots above; upper bits are not device BF16 data.
                payload = values & np.uint32(65535) if item['bits'] == 16 else values
                checks.receive(item, payload)
                boundary = (name=='tile_guard_snapshot' and ((item['phase']=='projection_tile_readback' and pe==1) or (item['phase']=='down_tile_readback' and pe==3))) or (item['phase']=='reset_observation' and name=='mlp_product' and pe==3) or (item['phase'] in ('initialize_epoch0','retention_after_down','released_join') and name=='lifecycle_state' and pe==3) or (item['phase']=='resident_weights_before_generation1' and pe==1)
                if boundary:
                    # Persist the complete tile group before any following H2D.
                    evidence.flush()
            checks.after_operation(item)
        require(counts == dict(copy_calls=647, launch_calls=49, h2d=99, d2h=548,
                               host_bytes=978168, native_bytes=539652), 'Complete physical transaction accounting')
        report['checks'] = checks.finish()
        evidence.event('all_numerical_protocol_retention_release_checks_passed')
    except BaseException as exc:
        primary = exc
        report.update(error_type=type(exc).__name__, message=str(exc))
        try:
            evidence.event('failure', error_type=type(exc).__name__, message=str(exc)[:1024])
        except BaseException as journal_error:
            report['failure_journal_error'] = str(journal_error)
    finally:
        report['physical_counts'] = counts
        primary, stopped = flush_and_stop(evidence, runner, report, primary)
        try:
            report['observed_affinity_after_sdk'] = current()
        except BaseException as exc:
            report['final_affinity_error'] = str(exc)
            if primary is None:
                primary = exc
        report.update(normal_stop=stopped, seconds=time.monotonic() - evidence.started)
        if primary is None and stopped:
            report['status'] = 'connected_reuse_passed'
        try:
            evidence.finish(report)
        except BaseException as exc:
            if primary is None:
                primary = exc
    if primary is not None:
        raise primary
    require(stopped, 'Normal SDK runtime stop required')


if __name__ == '__main__':
    main()
