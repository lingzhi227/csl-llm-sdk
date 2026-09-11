"""Unexecuted portable host configuration template.

The accepted run uses the separately recorded frozen source hash. Local machine
paths/stamps are removed here. Configure WORK/IMAGE/IMAGE_STAMP, freeze and admit
a separate candidate before execution; placeholder defaults refuse execution.
"""
import hashlib
import json
from pathlib import Path
import stat

from qwen38.elf_inventory import admit_wse3_sram, inventory

CANDIDATE = 'wp16-connected-sdk-003'
WORK = Path('/path/to/qwen38-cache') / CANDIDATE
IMAGE = Path('/path/to/cerebras-sdk-2.10.1.sif')
IMAGE_SHA = 'fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d'
IMAGE_STAMP = dict(st_size=0, st_dev=0, st_ino=0, st_mtime_ns=0, st_ctime_ns=0)
GENERATED = {
    'out/generated/MEMCPY_XY_ROUTES.elf': dict(bytes=888, sha256='824710fe484d28737ad9edea40c8685cbb1e7295784cfb9c721f4ebdaf000f5c'),
    'out/generated/coord.elf': dict(bytes=1344, sha256='04036570767d0d057ee3241caf97d49f87d901f8588fef7fe51b849a2e0bb16e'),
    'out/generated/default.elf': dict(bytes=52720, sha256='96865af0a1682028d57bf677b784546a150cf0f63da1df315a3c80110949cf35'),
}
PROFILE = dict(package='WP16',candidate=CANDIDATE,candidate_limit=1,
    scope='connected-fourPE-resident128x112-three-generations-partial-down128',pes=4,input_indices=[0,1,3],
    generation_input_contraction=[[1,0,1],[2,1,2],[3,3,3]],
    projection_rows=[0,128],projection_columns=[0,112],down_rows=[0,128],down_columns=[0,128],
    compile_seconds=300,planned_simulation_seconds=400,simulation_seconds=420,
    memory_bytes_by_stage={'compile':1073741824,'simulate':536870912},swap_bytes=0,cpu_affinity=[0],
    cpu_limit_method='single_logical_cpu_affinity_not_exclusive_or_bandwidth_quota',worker_threads=1,tasks_max=128,
    ram_reserve_bytes=8589934592,ssd_reserve_bytes=34359738368,ssd_cache_bytes=21474836480,ssd_run_bytes=16777216,
    file_bytes=2097152,log_bytes=1048576,core_bytes=0,source_bytes=4194304,compiled_bytes=4194304,
    private_observation_archive_bytes=2097152,affinity_journal_bytes=2097152,device_journal_bytes=2097152,device_journal_records=2000,
    selected_original_weight_bytes=90112,new_weight_acquisition_bytes=0,source_rerun=False,
    preparation_receipt_sha256='3a3f204e32a89db8f60a01c9d373e4b364f61e010cabbb1ea7de2eaba58dceb0',
    oracle_sha256='1e915d5c9bca30f6a3a8e9d5529201b17620b0c6c3587ba5b7d0835ff9886516',
    connected_source_manifest_sha256='15775313a35edf96db27d33cef155ce5814becdc360002be251e0630a6c444fb',
    full_weight_D2H_calls=13,full_weight_D2H_host_bytes=524392,
    projection_weight_H2D=2,projection_weight_endpoint_D2H=4,down_weight_H2D=6,down_weight_D2H=9,
    per_generation_reset_reads=56,per_generation_retention_reads=31,per_generation_release_reads=16,
    generation_bound_observation_keys=True,per_generation_weight_guard_and_source_checks=True,
    continuous_resident_weight_bitwise_equality_proven=False,ordinary_SRAM_bytes=49152,stack_allowance_bytes=4096,
    sdk_image=str(IMAGE),sdk_image_sha256_from_existing_installation_receipt=IMAGE_SHA,sdk_image_rehashed=False,
    original_model_rerun=False,Torch_import=False,GPU=False,download_or_install=False,full17408_down_output=False,
    network_scope='SDK local IPC only; no model HTTP or other remote network requests',
    physical_operations=696,copy_calls=647,launch_calls=49,host_bytes=978168,native_bytes=539652,
    raw_D2H_bytes=660392,maximum_observation_archives=39,normal_stop_required=True,automatic_retry=False)
PROFILE['expected_runtime_generated_files'] = GENERATED
PROFILE['runtime_generated_identity_assumed_proven_for_new_application'] = False
PROFILE['log_limit_method'] = 'sampled_stop_at_1MiB_with_2MiB_hard_per_file_ceiling'
STEPS = dict(steps=[dict(name=name, seconds=seconds,
    argv=['/usr/local/bin/python3', 'entry_sdk.py', name])
    for name, seconds in (('compile', 300), ('simulate', 420))])
COMPILE_ARGS = ['layout.csl', '--arch=wse3', '--fabric-dims=11,3', '--fabric-offsets=4,1',
                '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
    PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage_limits(stage):
    require(stage in ('compile','simulate'),'Exact bounded SDK stage')
    return {'memory.max':str(PROFILE['memory_bytes_by_stage'][stage]),'memory.swap.max':'0','pids.max':'128'}


def exact(left, right):
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def safe_file(root, name, limit):
    relative = Path(name)
    require(not relative.is_absolute() and '..' not in relative.parts and str(relative) == name, 'Safe relative SDK file')
    path = root / relative
    require(not any(p.is_symlink() for p in (path, *path.parents)), 'No SDK candidate symlinks')
    info = path.stat()
    require(stat.S_ISREG(info.st_mode) and info.st_size <= limit, 'Bounded regular SDK file')
    return path


def digest_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        while chunk := stream.read(1048576):
            digest.update(chunk)
    return digest.hexdigest()


def validate(admission, manifest_sha256):
    expected = dict(package='WP16', candidate=CANDIDATE, candidate_limit=1,
        sdk_execution_authorized=True, manifest_sha256=manifest_sha256,
        profile=PROFILE, steps=STEPS, original_model_rerun_authorized=False,
        new_weight_preparation_authorized=False)
    require(exact(admission, expected), 'Explicit exact single SDK candidate admission required')


def verify(work):
    require(work == WORK and not any(p.is_symlink() for p in (work, *work.parents)), 'Exact SDK candidate directory')
    manifest_path = safe_file(work, 'source-manifest.json', 262144)
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    digest = hashlib.sha256(raw).hexdigest()
    require(manifest.get('package') == 'WP16' and manifest.get('candidate') == CANDIDATE, 'SDK freeze identity')
    require(25 <= len(manifest['files']) <= 96, 'Bounded SDK source inventory')
    total = 0
    for name, expected in manifest['files'].items():
        path = safe_file(work, name, 2097152)
        total += path.stat().st_size
        require(digest_file(path) == expected, 'Frozen SDK source changed: ' + name)
    require(total <= PROFILE['source_bytes'], 'All frozen SDK sources and selected input byte budget')
    require(exact(json.loads((work / 'profile.json').read_bytes()), PROFILE), 'Exact SDK profile')
    require(exact(json.loads((work / 'steps.json').read_bytes()), STEPS), 'Exact serial SDK steps')
    admission = json.loads(safe_file(work, 'controller-admission.json', 262144).read_bytes())
    validate(admission, digest)
    return digest


def image_stamp():
    require(not any(p.is_symlink() for p in (IMAGE, *IMAGE.parents)) and IMAGE.is_file(), 'Qualified existing SDK SIF')
    info = IMAGE.stat()
    stamp = {name: getattr(info, name) for name in IMAGE_STAMP}
    require(stamp == IMAGE_STAMP, 'SDK installation file identity changed; separate review needed')
    return dict(current_stat=stamp, sha256_from_existing_installation_receipt=IMAGE_SHA, rehashed=False)


def application_elf_names(files):
    """SDK edge support remains inventoried, outside the application's direct bin."""
    expected = {'out/bin/out_' + str(pe) + '_0.elf' for pe in range(4)}
    actual = {name for name in files if Path(name).parent == Path('out/bin') and Path(name).suffix == '.elf'}
    require(actual == expected, 'Exactly four actual compute PE ELFs required')
    return sorted(expected)


def compiled_state(work):
    """Bounded identity inventory and actual ELF gate for each of the four compute PEs."""
    files = {}
    total = 0
    output = work / 'out'
    require(output.is_dir() and not output.is_symlink(), 'Compiled output directory required')
    for path in sorted(output.rglob('*')):
        require(not path.is_symlink(), 'No compiled output symlinks')
        if path.is_dir():
            continue
        name = str(path.relative_to(work))
        path = safe_file(work, name, PROFILE['file_bytes'])
        total += path.stat().st_size
        require(len(files) < 512 and total <= PROFILE['compiled_bytes'], 'Compiled file inventory budget')
        files[name] = dict(bytes=path.stat().st_size, sha256=digest_file(path))
    application = application_elf_names(files)
    footprints = {name: inventory(safe_file(work, name, 2097152).read_bytes()) for name in application}
    gates = {name: admit_wse3_sram(value, 4096, 49152) for name, value in footprints.items()}
    return dict(files=files, compiled_bytes=total, footprints=footprints, pes=gates,
                passed=all(gate['passed'] for gate in gates.values()))


def compare_compiled_states(actual, frozen, *, after_runtime=False):
    require(not any(name.startswith('out/generated/') for name in frozen['files']), 'Fresh compile must precede runtime-generated files')
    require(all(actual['files'].get(name) == spec for name,spec in frozen['files'].items()), 'All original compiled files unchanged')
    extras = {name:spec for name,spec in actual['files'].items() if name not in frozen['files']}
    require(after_runtime or not extras, 'No runtime-generated extras before simulator')
    require(all(GENERATED.get(name) == spec for name,spec in extras.items()), 'Unknown or changed runtime auxiliary artifact')
    require(sum(v['bytes'] for v in extras.values()) <= 54952, 'Bounded runtime auxiliary bytes')
    for name in ('footprints', 'pes', 'passed'):
        require(exact(actual[name], frozen[name]), 'Unchanged application ELF footprint/SRAM gates')
    require(actual['passed'] and actual['compiled_bytes'] == frozen['compiled_bytes'] + sum(v['bytes'] for v in extras.values()), 'Compiled byte reconciliation and SRAM pass')
    return extras


def verify_compiled(work, *, after_runtime=False):
    actual = compiled_state(work)
    frozen = json.loads(safe_file(work, 'compiled-admission.json', 1048576).read_bytes())
    compare_compiled_states(actual, frozen, after_runtime=after_runtime)
    return actual
