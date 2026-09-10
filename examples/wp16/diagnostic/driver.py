"""Exact four-PE diagnostic schedule; execute only through its scoped SDK entry."""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

from checks import Checks, key, require
from qwen38.wp16_diag_layout import ARRAYS, FLOAT32, SCOPE, budget, sequence
from qwen38.wp16_diag_protocol import descriptor

PREPARATION_SHA = '7049783b915990a96b74deeb7d1356d6db1f65ae7bf53cc888cfe27d10ef898d'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_prepared(work):
    folder = work / 'prepared'
    raw = (folder / 'preparation.json').read_bytes()
    require(sha(raw) == PREPARATION_SHA, 'Exact accepted preparation receipt')
    receipt = json.loads(raw)
    require(receipt['status'] == 'partial_source_prepared' and receipt['scope'] == SCOPE,
            'Prepared diagnostic scope')
    require(receipt['source_uses_original_CPU_observations'] is False and
            receipt['full17408_down_output_used'] is False, 'Independent partial source')
    verified = {}
    for name, expected in receipt['files'].items():
        raw = (folder / name).read_bytes()
        require(len(raw) == expected['bytes'] and sha(raw) == expected['sha256'], 'Prepared file identity: ' + name)
        verified[name] = raw
    weights = {role: np.frombuffer(verified['weights/' + role + '.bf16'], dtype='<u2').reshape(128, 128 if role == 'down_proj' else 5120)
               for role in ('gate_proj', 'up_proj', 'down_proj')}
    hidden = np.frombuffer(verified['hidden.bf16'], dtype='<u2').reshape(4, 5120)[0]
    with np.load(io.BytesIO(verified['oracle.npz']), allow_pickle=False) as archive:
        require(set(archive.files) == set(receipt['arrays']), 'Exact partial oracle array names')
        oracle = {name: archive[name] for name in archive.files}
    for name, array in oracle.items():
        spec = receipt['arrays'][name]
        require(array.shape == tuple(spec['shape']) and str(array.dtype) == spec['dtype'] and
                array.nbytes == spec['bytes'], 'Prepared oracle schema: ' + name)
    require(sum(a.nbytes for a in oracle.values()) == 37888, 'Prepared oracle allocation bound')
    return weights, hidden, oracle


def load_prefixes(work):
    """Only a separately accepted, frozen prefix receipt and exact three arrays."""
    from qwen38.wp16_sdk_admission import PROFILE
    folder = work / 'prefix-source'
    raw = (folder / 'preparation.json').read_bytes()
    require(len(raw) <= 65536 and sha(raw) == PROFILE['prefix_preparation_sha256'], 'Exact accepted all-prefix receipt')
    receipt = json.loads(raw)
    require(receipt['status'] == 'prefix_source_prepared' and
            receipt['source_manifest_sha256'] == PROFILE['prefix_source_manifest_sha256'] and
            receipt['source_uses_CPU_or_SDK_observations'] is False, 'Independent all-prefix source')
    spec = receipt['files']['prefixes.npz']
    raw = (folder / 'prefixes.npz').read_bytes()
    require(len(raw) <= 262144 and len(raw) == spec['bytes'] and sha(raw) == spec['sha256'] == PROFILE['prefix_archive_sha256'], 'Admitted all-prefix archive')
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        require(set(archive.files) == {'gate_prefix_fp32', 'up_prefix_fp32', 'prefix_columns'}, 'Three all-prefix arrays')
        arrays = {name: archive[name] for name in archive.files}
    for name, array in arrays.items():
        spec = receipt['arrays'][name]
        require(list(array.shape) == spec['shape'] and str(array.dtype) == spec['dtype'] and
                array.nbytes == spec['bytes'] and sha(array.tobytes()) == spec['sha256'], 'Frozen prefix array schema/hash')
    for stage in ('gate', 'up'):
        values = arrays[stage + '_prefix_fp32']
        require(values.dtype == np.float64 and values.shape == (46,128,2) and
                np.isfinite(values).all() and np.all(values[:,:,0] <= values[:,:,1]), 'Finite ordered all-prefix source')
    columns = arrays['prefix_columns']
    require(columns.dtype == np.uint32 and columns.shape == (46,) and
            columns.tolist() == [min(112*(i+1),5120) for i in range(46)] and
            sum(a.nbytes for a in arrays.values()) == 188600, 'Exact all-prefix columns and byte budget')
    return arrays


def upload_values(item, weights, hidden):
    """Only original selected weights, original hidden and exact control metadata."""
    name, pe, phase, tile = (item[k] for k in ('name', 'pe', 'phase', 'tile'))
    count = item['count']
    if name == 'control':
        total = 5120 if pe < 2 else 128
        width = 112 if pe < 2 else 64
        offset = tile * width if tile is not None else 0
        return np.array([1, 0, 1, total, tile if tile is not None else 0, offset,
                         min(width, total - offset) if tile is not None else 0, 0], dtype=np.uint32)
    if name == 'handoff_control':
        return np.array(descriptor(int(phase[-1])), dtype=np.uint32)
    if name == 'release_control':
        return np.array([1, 0, 1, 1], dtype=np.uint32)
    if name == 'weights_storage':
        require(pe in (0, 1, 3) and tile is not None, 'Only selected original projection/down weights')
        role = ('gate_proj', 'up_proj', '', 'down_proj')[pe]
        width, total = (112, 5120) if pe < 2 else (64, 128)
        offset = tile * width
        valid = min(width, total - offset)
        result = np.full(count, 0x3e80, dtype=np.uint32)
        result[0], result[-1] = 0xa55a, 0x5aa5
        result[1:-1].reshape(width, 128)[:valid] = weights[role][:, offset:offset + valid].T
        return result
    if name == 'input_storage':
        require(pe < 2 and tile is not None, 'No host numerical consumer input')
        offset = tile * 112
        valid = min(112, 5120 - offset)
        result = np.full(count, 7., dtype=np.float32)
        result[0], result[-1] = 77.5, -88.25
        result.view(np.uint32)[1:1 + valid] = hidden[offset:offset + valid].astype(np.uint32) << 16
        return result
    raise ValueError('Undeclared upload: ' + name)


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
        group = (item['phase'], item['tile'])
        if group != self.group:
            self.flush()
            self.group = group
        tag = key(item['phase'], item['name'], item['pe'], item['tile'])
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
        phase, tile = self.group
        name = phase + ('_tile' + str(tile) if tile is not None else '') + '.npz'
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
    from qwen38.wp16_sdk_admission import PROFILE, verify
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
    prefixes = load_prefixes(work)
    checks = Checks(oracle, weights, prefixes)
    evidence = Evidence(work / 'device-evidence')
    runner = None
    primary = None
    stopped = False
    counts = dict(copy_calls=0, launch_calls=0, h2d=0, d2h=0, host_bytes=0, native_bytes=0)
    report = dict(status='failed', scope=SCOPE, source_manifest_sha256=digest,
                  original_dense_input_index=0, full17408_down_output=False,
                  full_model_or_original_MLP_forward=False, physical_WSE=False,
                  middle_projection_full_weight_bitwise_D2H_coverage=False,
                  weight_evidence='Host upload hashes, per-tile device guards/source checks; full device payload equality at first/last projections, both down tiles and final retention only')

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
            metadata = dict(operation=item)
            if item['kind'] == 'launch':
                call('launch', lambda: runner.launch(item['name'], nonblock=False), **metadata)
                counts['launch_calls'] += 1
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
                if name == 'tile_guard_snapshot' and pe in (1, 3):
                    # Persist the complete tile group before any following H2D.
                    evidence.flush()
        require(counts == dict(copy_calls=785, launch_calls=61, h2d=303, d2h=482,
                               host_bytes=5959676, native_bytes=3061668), 'Complete physical transaction accounting')
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
            report['status'] = 'diagnostic_passed'
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
