"""Check inherited CPU0 and explicit clean-container environment before SDK import."""
import json
import os
from pathlib import Path
import runpy
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from qwen38.wp16_affinity import current
from qwen38.wp16_sdk_admission import THREAD_ENV, WORK, require, verify, verify_compiled


def write(name, value):
    with (WORK / name).open('x') as stream:
        json.dump(value, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())


def main():
    require(len(sys.argv) == 2 and sys.argv[1] in ('compile', 'simulate'), 'Exact inner SDK stage')
    stage = sys.argv[1]
    require(Path.cwd() == WORK and os.environ.get('WP16_SDK_STAGE') == stage, 'Exact clean-container stage')
    digest = verify(WORK)
    require(all(os.environ.get(k) == v for k, v in THREAD_ENV.items()) and os.environ.get('TMPDIR') == '/tmp',
            'Explicit one-worker environment survived container isolation')
    outer = json.loads((WORK / ('runtime-admission-' + stage + '.json')).read_bytes())
    require(outer['source_manifest_sha256'] == digest and outer['stage'] == stage and
            outer['unit'] == os.environ.get('WP16_SDK_UNIT') and outer['affinity'] == [0] and
            outer['actual_limits'] == {'memory.max': '1073741824', 'memory.swap.max': '0', 'pids.max': '128'},
            'Matching host actual-resource admission')
    if stage == 'simulate':
        verify_compiled(WORK)
    write('container-admission-' + stage + '.json', dict(stage=stage, unit=outer['unit'],
        source_manifest_sha256=digest, affinity=current(), thread_environment=THREAD_ENV,
        sdk_import_has_not_started=True, actual_cgroup_verified_by='host entry and outer live-task observer'))
    status = 'failed'
    script = 'compile_sdk.py' if stage == 'compile' else 'driver.py'
    try:
        sys.argv = [script]
        runpy.run_path(script, run_name='__main__')
        status = 'passed'
    finally:
        write('container-exit-' + stage + '.json', dict(stage=stage, status=status,
            unit=outer['unit'], source_manifest_sha256=digest, affinity=current()))


if __name__ == '__main__':
    main()
