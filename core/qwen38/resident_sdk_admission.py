"""Unexecuted portable configuration template for reproducing the resident fragment.

Configure paths/image identity and freeze a fresh candidate before execution.
The accepted historical SDK002 run retains its original hashes and failed guard;
these placeholder defaults refuse execution. Geometry-specific auxiliary
identities come from the independent post-run qualification for11x4 only.
"""
import hashlib
import json
from pathlib import Path
import stat

from qwen38.elf_inventory import admit_wse3_sram, inventory

CANDIDATE = 'wp16-resident-sdk-002'
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
    scope='resident-eightPE-192to128to128-three-generations',pes=8,input_indices=[0,1,3],
    generation_input_contraction=[[1,0,1],[2,1,2],[3,3,3]],
    projection_rows=[0,128],projection_columns=[0,192],down_rows=[0,128],down_columns=[0,128],
    compile_seconds=0,planned_simulation_seconds=400,simulation_seconds=420,
    memory_bytes_by_stage={'simulate':536870912},swap_bytes=0,cpu_affinity=[0],
    cpu_limit_method='single_logical_cpu_affinity_not_exclusive_or_bandwidth_quota',worker_threads=1,tasks_max=128,
    ram_reserve_bytes=8589934592,ssd_reserve_bytes=34359738368,ssd_cache_bytes=21474836480,ssd_run_bytes=8388608,
    file_bytes=2097152,log_bytes=1048576,core_bytes=0,source_bytes=2097152,compiled_bytes=4194304,
    private_observation_archive_bytes=2097152,affinity_journal_bytes=2097152,device_journal_bytes=1048576,device_journal_records=512,
    selected_original_weight_bytes=131072,new_weight_acquisition_bytes=0,source_rerun=False,
    preparation_receipt_sha256='1ac5572cfa0938e33af78f2ee6ca58306740eea3478550f5d2a4e0bb594d8ef3',
    oracle_sha256='2e109cd55accd83c3a6ac147aa2fae8ee0f27ed2043d5b71ce4a4a23649dd96e',
    resident_source_manifest_sha256='6fb78fa5d06c6d36fd2be5e32ab5f1979c8d3c80763c803e401480c0c7588485',
    initial_weight_H2D_calls=6,steady_token_weight_H2D_calls=0,final_weight_D2H_calls=6,
    per_generation_readbacks=35,host_stage_launches_per_input=1,host_neural_intermediate_copies=0,
    generation_bound_observation_keys=True,per_generation_weight_guard_and_source_checks=True,
    continuous_resident_weight_bitwise_equality_proven=False,physical_SRAM_bytes=49152,
    ordinary_SRAM_max_bytes=44032,stack_allowance_bytes=4096,minimum_spare_bytes=1024,
    sdk_image=str(IMAGE),sdk_image_sha256_from_existing_installation_receipt=IMAGE_SHA,sdk_image_rehashed=False,
    original_model_rerun=False,Torch_import=False,GPU=False,download_or_install=False,full_model=False,
    network_scope='SDK local IPC only; no model HTTP or remote physical endpoint requests',
    physical_operations=125,copy_calls=121,launch_calls=4,host_bytes=581280,native_bytes=310232,
    raw_D2H_bytes=316760,maximum_observation_archives=5,normal_stop_required=True,automatic_retry=False)
PROFILE['expected_runtime_generated_files'] = GENERATED
PROFILE['runtime_generated_identity_assumed_proven_for_new_application'] = False
PROFILE['unknown_runtime_auxiliary_requires_preserved_failure_and_independent_review_not_retry'] = True
PROFILE['log_limit_method'] = 'sampled_stop_at_1MiB_with_2MiB_hard_per_file_ceiling'
PROFILE['compiled_origin'] = dict(candidate='wp16-resident-sdk-001',
    source_manifest_sha256='dbc7ada6414d6f0d78bcc702bc3286fab95c35c7103890b6dcab3cf2ed2c6cc5',
    compiled_files=36,compiled_bytes=777441,
    compiled_state_sha256='65968df3874c250da1d184a866890a95787cd3743ecb1762d668d88130b83392',
    parent_compiler_returncode=0,parent_host_gate_passed=False,parent_simulator_started=False,
    recompile=False,copy_compiled_within_workstation=True,parent_outputs_modified=False)
PROFILE['actual_SDK_ELF_rectangle_gate_before_simulator_construction'] = True
PROFILE['host_neural_intermediate_copies_semantics'] = 'zero neural compute/feedback dependency; 112 diagnostic D2H copies remain budgeted'
STEPS = dict(steps=[dict(name=name, seconds=seconds,
    argv=['/usr/local/bin/python3', '-B', 'entry_sdk.py', name])
    for name, seconds in (('simulate', 420),)])
COMPILE_ARGS = ['layout.csl', '--arch=wse3', '--fabric-dims=11,4', '--fabric-offsets=4,1',
                '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1', '--dump-dsr-alloc-graph']
THREAD_ENV = dict(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1',
    NUMEXPR_NUM_THREADS='1', OMP_DYNAMIC='FALSE', MKL_DYNAMIC='FALSE',
    PYTHONDONTWRITEBYTECODE='1', CUDA_VISIBLE_DEVICES='')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stage_limits(stage):
    require(stage == 'simulate','Only simulation continuation is admitted; no compilation')
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
    expected = {f'out/bin/out_{rank}_0.elf' for rank in range(8)}
    actual = {name for name in files if Path(name).parent == Path('out/bin') and Path(name).suffix == '.elf'}
    require(actual == expected, 'Exactly eight actual compute PE ELFs required')
    return sorted(expected)


def compiled_state(work):
    """Bounded identity inventory and actual ELF gate for each of the eight compute PEs."""
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
    gates = {name: admit_wse3_sram(value, 4096, 48128) for name, value in footprints.items()}
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
    frozen_bytes = safe_file(work, 'compiled-admission.json', 1048576).read_bytes()
    require(hashlib.sha256(frozen_bytes).hexdigest() == PROFILE['compiled_origin']['compiled_state_sha256'],
            'Exact preserved SDK001 compiled inventory')
    frozen = json.loads(frozen_bytes)
    compare_compiled_states(actual, frozen, after_runtime=after_runtime)
    return actual
