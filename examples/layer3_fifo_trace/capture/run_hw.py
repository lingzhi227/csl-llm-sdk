"""Bind accepted compilation and every original parameter before device allocation."""
import json
import logging
import os
from pathlib import Path
from source_gate import ROOT, PROFILE, verify
from hw00.job_capture import install
from hw00.store import atomic_json
from layer3_capture import capture, require, hash_file


def record_gate(root, entries):
    for name, item in entries.items():
        path = root/name
        require(not Path(name).is_absolute() and '..' not in Path(name).parts and not path.is_symlink()
            and path.stat().st_size == item['bytes'] and hash_file(path) == item['sha256'], 'Exact input identity: '+name)


def preflight(inputs):
    compiled = Path(inputs['compile_root']); record_gate(compiled, inputs['compile_receipts'])
    frozen = json.loads((compiled/'manifest.json').read_bytes()); record_gate(compiled, frozen['files'])
    complete = json.loads((compiled/'COMPLETE.json').read_bytes())
    sram = json.loads((compiled/'sram.json').read_bytes())
    placement = json.loads((compiled/'placement.json').read_bytes())
    require(complete['all_owned_jobs_released'] and not complete['runtime_created'] and sram['passed']
        and placement['passed'] and placement['PEs'] == 33750 and placement['matrix_PEs'] == 30576,
        'Accepted full original-layer actual SRAM and disjoint placement')
    require(len(sram['application_files']) == inputs['application_programs'] == len(placement['files'])
        and 1 <= inputs['application_programs'] <= 640
        and all(x['passed'] and x['low_section_end']+x['stack_allowance_bytes'] <= 48128
                for x in sram['application_files'].values())
        and placement['snapshot_rectangle'] == [748, 0, 2, 25]
        and placement['snapshot_words'] == 160 and placement['host_snapshot_bytes'] == 32000
        and placement['actual_read_locations'] == 50 and placement['inactive_trace_sinks'] == 19
        and placement['trace_sources'] == placement['trace_sinks'] == 6
        and placement['global_sink'] == [748, 0] and placement['head_sink_count'] == 24
        and placement['diagnostic_RMS_alias_bytes'] == 4096
        and not placement['retained_QK_record_audit_valid'],
        'Accepted actual all-head diagnostic geometry and SRAM')
    for row in placement['files']:
        header = row['packet_header_array']; payload = row['packet_payload_array']
        require(header['bytes'] == 4 and payload['bytes'] == 124
            and header['address'] % 4 == 0 and payload['address'] % 4 == 0
            and max(header['address'], payload['address']) >= min(header['address']+4, payload['address']+124),
            'Accepted independent packet header and payload arrays')
    for name in ['fifo-trace-contract.json']:
        require(hash_file(ROOT/name) == frozen['files'][name]['sha256'], 'Exact compiled trace source contract')
    require(hash_file(ROOT/'layer3-plan.json') == frozen['files']['layer3-plan.json']['sha256'], 'Exact compiled graph geometry')
    artifact = Path(inputs['artifact'])
    require(artifact.parent == compiled and not artifact.is_symlink() and artifact.stat().st_size == inputs['artifact_bytes']
        and hash_file(artifact) == inputs['artifact_sha256'], 'Exact compiled artifact')
    prep = Path(inputs['preparation_candidate']); record_gate(prep, inputs['preparation_receipts'])
    require(hash_file(ROOT/'prepared.json') == inputs['preparation_receipts']['prepared.json']['sha256'], 'Prepared receipt association')
    require(json.loads((prep/'supervisor.json').read_bytes())['status'] == 'passed', 'Preparation passed')
    prepared = Path(inputs['prepared_root']); receipt = json.loads((prep/'prepared.json').read_bytes())
    require(receipt['original_tensor_count'] == 11 and receipt['matrix_uploads'] == 95
        and receipt['packed_matrix_bytes'] == 751435776 and receipt['positions'] == list(range(8)), 'Complete original-layer packing')
    record_gate(prepared, receipt['files'])
    for name in receipt['files']:
        path = prepared/name
        require(path.stat().st_dev == 51 and path.stat().st_mode & 0o222 == 0, 'Accepted readonly native prepared array')
    return artifact


def main():
    verify(); inputs = json.loads((ROOT/'inputs.json').read_bytes()); artifact = preflight(inputs)
    install(os.environ['HW00_JOB_CAPTURE']); logging.basicConfig(level=logging.INFO)
    from cerebras.sdk.client import SdkRuntime
    from cerebras.appliance.pb.sdk.sdk_common_pb2 import MemcpyDataType, MemcpyOrder

    class Runtime:
        def __init__(self):
            self.closed = False
            self.runtime = SdkRuntime(str(artifact), simulator=False, disable_version_check=True,
                resource_cpu=PROFILE['worker_cpu_millicores'], resource_mem=PROFILE['worker_memory_bytes'])
            self.runtime.__enter__()

        def __getattr__(self, name):
            return getattr(self.runtime, name)

        def stop(self):
            if not self.closed:
                self.runtime.__exit__(None, None, None); self.closed = True

    runner = None
    try:
        runner = Runtime(); result = capture(runner, MemcpyDataType, MemcpyOrder, ROOT)
    finally:
        if runner is not None:
            runner.stop()
    require(result['status'] == 'captured' and result['normal_stop'] and result['copies'] in PROFILE['copies']
        and result['launches'] == PROFILE['launches'], 'Exact bounded physical observation capture')
    atomic_json(ROOT/'physical.json', dict(simulator=False, artifact_sha256=inputs['artifact_sha256'],
        runtime_context_exited=runner.closed, application=[750, 45], matrix_PEs=30576,
        original_layer=3, mathematical_audit_passed=False, diagnostic_only=True, complete_epochs=result['complete_epochs'], full_model=False))
    verify()


if __name__ == '__main__':
    main()
