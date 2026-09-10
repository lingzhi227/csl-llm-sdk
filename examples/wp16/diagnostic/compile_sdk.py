"""One fixed compile, bounded output inventory, every actual PE SRAM admission."""
import json
import os
from pathlib import Path
import subprocess

from qwen38.wp16_affinity import current
from qwen38.wp16_sdk_admission import COMPILE_ARGS, WORK, compiled_state, require, verify


def main():
    require(Path.cwd() == WORK and os.environ.get('WP16_SDK_STAGE') == 'compile', 'Scoped compiler stage')
    verify(WORK)
    current()
    require(not (WORK / 'out').exists(), 'Compiler candidate already attempted')
    result = subprocess.run(['sdk_debug_shell', 'compile', *COMPILE_ARGS])
    require(result.returncode == 0, 'SDK compiler failed with status ' + str(result.returncode))
    current()
    verify(WORK)
    compiled = compiled_state(WORK)
    with (WORK / 'compiled-admission.json').open('x') as stream:
        json.dump(compiled, stream, indent=2); stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    require(compiled['passed'], 'Actual PE SRAM end plus4096-byte stack exceeds49152')


if __name__ == '__main__':
    main()
