"""All original mixer contractions on the full resident compiled layer stage.

The host supplies synthetic boundary operands, then reads actual device roots.
It does not compute neural intermediates. Explicit root consumption is diagnostic
qualification, not the final fused serving schedule or a model TPS measurement.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from backend import runtime, file_sha256, complete_mixer_profile
from device_client import Encoder, Client, batches
from mixer_qualification import validate, root_rows, canonical_bf16, SHAPES
from run_layer_mlp import BankFile, row_runs
from progress import Progress


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--physical', action='store_true')
    physical = parser.parse_args().physical
    config = json.loads(Path('experiment.json').read_text())
    if not complete_mixer_profile(config):
        raise ValueError('Complete original mixer profile required')
    for name, expected in config['payload_hashes'].items():
        if file_sha256(Path(name)) != expected:
            raise ValueError('Frozen original payload identity: ' + name)
    meta = json.loads(Path('fixture.json').read_text())
    if meta['criterion'] != config['acceptance'] or meta['cases_count'] != 4:
        raise ValueError('Frozen independent acceptance changed')
    data = dict(np.load('fixture.npz', allow_pickle=False))
    load = lambda name: json.loads(Path(name).read_text())
    plan = validate(load('joint-stage.json'), load('device-network.json'), load('mixer-setups.json'),
                    load('profiles.json'), load('bank-index.json'))
    banks = BankFile('banks.npy')
    if banks.nbytes != 396953216:
        raise ValueError('Original bank extent')
    progress = Progress(True)
    captured, observations = {}, []
    last_progress = [0.]
    started = time.monotonic()

    def pulse(phase, generation=None, force=False):
        now = time.monotonic()
        if force or now - last_progress[0] >= 2:
            progress.mark(phase, generation)
            last_progress[0] = now

    def save():
        np.savez('actual.npz', **captured)

    try:
        with runtime(physical) as (runner, dtype, order):
            loaded = time.monotonic()
            pulse('initialize', force=True)
            encoder = Encoder(3554)
            client = Client(runner, dtype, order, plan['gateway'])
            bank_id = runner.get_id('bank')
            opts = dict(streaming=False, order=order.ROW_MAJOR, nonblock=False, data_type=dtype.MEMCPY_32BIT)

            def sdk_bank(run, write):
                first, count = run[0], run[0]['words']
                if [r['offset'] for r in run] != list(range(first['offset'], first['offset'] + count*len(run), count)):
                    raise ValueError('SDK bank run is not contiguous')
                expected = banks[first['offset']:first['offset'] + count*len(run)]
                x, y = first['pe'][0]-plan['origin'][0], first['pe'][1]-plan['origin'][1]
                if write:
                    runner.memcpy_h2d(bank_id, expected, x, y, len(run), 1, count, **opts)
                else:
                    actual = np.empty(count*len(run), np.uint32)
                    runner.memcpy_d2h(actual, bank_id, x, y, len(run), 1, count, **opts)
                    np.testing.assert_array_equal(actual, expected)
                pulse('sdk_bank_upload' if write else 'sdk_bank_retention')

            for run in row_runs(plan['sdk_banks'], lambda r: r['words']):
                sdk_bank(run, True)

            def initialization():
                for r in plan['device_banks']:
                    tag = plan['targets'][tuple(r['pe'])]
                    for first in range(0, r['words'], 256):
                        n = min(256, r['words']-first)
                        yield encoder.command(tag, 0, 0, first, n, payload=banks[r['offset']+first:r['offset']+first+n])
                    yield encoder.command(tag, 0, 1, 0, 40, payload=plan['descriptors'][tuple(r['pe'])])
            for batch in batches(initialization()):
                client.stream(batch)
                pulse('device_bank_upload')
            initialized = time.monotonic()
            print(json.dumps(dict(phase='initialized',bank_bytes=banks.nbytes,seconds=initialized-loaded,commands=client.completed)),flush=True)
            pending = []

            def read(tag, segment, first, count):
                frame = encoder.command(tag, 1, segment, first, count)
                client.stream([*pending, frame]); pending.clear()
                return client.read_reply()

            for case in meta['cases']:
                ci = case['case']
                invocation = 0x70000001 + ci
                for mi, shape in enumerate(SHAPES):
                    generation = ci*5 + mi + 1
                    pulse('begin_projection', generation, True)
                    phase_started = time.monotonic()
                    frames = (encoder.command(tag,5,invocation,generation,mi) for tag in range(3554))
                    for batch in batches(frames):
                        client.stream(batch); pulse('begin_projection',generation)
                    if mi in (0,4):
                        parts = shape[1]//128
                        packed = np.asarray(data[f'input_{ci}_{parts}'],dtype='<u2').view('<u4')
                        frames = [encoder.operand(invocation,k,parts,packed[k*64:(k+1)*64]) for k in range(parts)]
                        for batch in batches(frames):
                            client.stream(batch); pulse('projection_operands',generation)
                    actual = np.zeros(shape[0],np.uint16)
                    seen = np.zeros(shape[0],np.uint8)
                    captured[f'output_{ci}_{mi}'] = actual
                    captured[f'output_valid_{ci}_{mi}'] = seen
                    for ri,(root,row) in enumerate(root_rows(plan['roots'][mi])):
                        deadline = time.monotonic()+10
                        while True:
                            status = read(root['tag'],3,0,4)
                            if status[2] == 1:
                                np.testing.assert_array_equal(status[:3],[generation,row,1]); break
                            if time.monotonic() > deadline:
                                raise TimeoutError('Root readiness '+str((generation,root['tag'],row,status.tolist())))
                        packet = read(root['tag'],4,0,5)
                        np.testing.assert_array_equal(packet[[0,1,2,4]],[generation,row,root['rows'],mi])
                        values = np.asarray([packet[3]],dtype='<u4').view('<u2')[:root['rows']]
                        if np.any(seen[row:row+root['rows']]):raise ValueError('Duplicate root output')
                        actual[row:row+root['rows']] = values; seen[row:row+root['rows']] = 1
                        pending.append(encoder.command(root['tag'],6,generation,row))
                        pulse('capture_projection',generation)
                        if ri%256 == 0:save()
                    if not np.all(seen == 1):raise ValueError('Missing original output rows')
                    save()
                    np.testing.assert_array_equal(canonical_bf16(actual),canonical_bf16(data[f'bf16_{ci}_{mi}']))
                    audit = np.zeros((3554,6),np.uint32);captured[f'audit_{ci}_{mi}'] = audit
                    for tag in range(3554):
                        count = plan['iterations'][mi][tag]
                        audit[tag] = read(tag,2,0,6)
                        keys = shape[1]//128
                        expected = [generation,count,count if tag%keys < keys-1 else 0,count,int(count>0 and mi in (0,4)),1]
                        np.testing.assert_array_equal(audit[tag],expected)
                        pulse('projection_drain_audit',generation)
                    save()
                    observation = dict(case=ci,matrix=mi,outputs=shape[0],all_original_bf16_exact_except_zero_sign=True,
                                       all3554_counters_exact=True,host_diagnostic_seconds=time.monotonic()-phase_started)
                    observations.append(observation);print(json.dumps(observation),flush=True)
            status_all = np.zeros((3554,4),np.uint32);captured['final_status']=status_all
            for tag in range(3554):
                status_all[tag]=read(tag,3,0,4)
                np.testing.assert_array_equal(status_all[tag,2:],[0,20]);pulse('final_leases')
            for r in plan['device_banks']:
                tag=plan['targets'][tuple(r['pe'])]
                for first in range(0,r['words'],256):
                    n=min(256,r['words']-first)
                    np.testing.assert_array_equal(read(tag,0,first,n),banks[r['offset']+first:r['offset']+first+n])
                    pulse('device_bank_retention')
                np.testing.assert_array_equal(read(tag,1,0,40),plan['descriptors'][tuple(r['pe'])])
            for run in row_runs(plan['sdk_banks'], lambda r:r['words']):sdk_bank(run,False)
            save()
            result=dict(passed=True,physical=physical,normal_stop=True,scope=config['scope'],complete_original_mixer_projections=True,
                        complete_neural_stage=False,full_model_speed_target_achieved=False,original_output_values=86400,
                        original_bank_bytes_retained=banks.nbytes,all_internal_pes=3554,observations=observations,
                        loading_seconds=loaded-started,initialization_seconds=initialized-loaded,
                        total_host_diagnostic_seconds=time.monotonic()-started,commands=client.completed)
        Path('result.json').write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result),flush=True)
    finally:
        save()


if __name__ == '__main__':main()
