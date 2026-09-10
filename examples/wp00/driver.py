"""Seven raw BF16 patterns, one PE. This is a native memcpy ABI test only."""
from array import array
from dataclasses import replace
import json
from pathlib import Path
import sys
import time
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / 'core'))
from qwen38.codecs import encode, decode
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform

root = Path(__file__).resolve().parent
phases = []
def record(phase):
    phases.append({'phase':phase, 'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases, indent=2)+'\n')

record('construct_enter')
runner = SdkRuntime('out', get_platform(None, SimfabConfig(suppress_trace=True, num_threads=1, dump_core=False), SdkTarget.WSE3))
record('construct_done')
source, destination = runner.get_id('input'), runner.get_id('output')
for name in ('load', 'run'):
    record(name+'_enter'); getattr(runner, name)(); record(name+'_done')
bits = array('H', [123, 0, 0x8000, 0x3f80, 0xbf80, 0x7f80, 0xff80, 0x7fc1])
frame = encode(memoryview(bits)[1:], 'memcpy16_containers')
host_input = np.frombuffer(frame.payload, dtype='<u4').copy()
host_output = np.full(7, 0xdeadbeef, dtype=np.uint32)
options = dict(streaming=False, data_type=MemcpyDataType.MEMCPY_16BIT,
               order=MemcpyOrder.ROW_MAJOR, nonblock=False)
record('h2d_enter')
runner.memcpy_h2d(source, host_input, 0, 0, 1, 1, frame.sdk_count, **options)
record('h2d_done'); record('device_copy_enter')
runner.launch('copy_bits', nonblock=False)
record('device_copy_done'); record('d2h_enter')
runner.memcpy_d2h(host_output, destination, 0, 0, 1, 1, frame.sdk_count, **options)
record('d2h_done')
actual = decode(replace(frame, payload=host_output.astype('<u4', copy=False).tobytes()), frame.codec)
expected = list(memoryview(bits)[1:])
(root/'bits.json').write_text(json.dumps({'expected':expected, 'actual':actual,
    'raw_d2h_containers':host_output.tolist(), 'bit_equal':actual==expected,
    'logical_bytes':frame.logical_bytes, 'host_bytes':frame.host_bytes,
    'sdk_count':frame.sdk_count, 'sdk_unit':frame.sdk_unit}, indent=2)+'\n')
record('stop_enter'); runner.stop(); record('stop_done')
if actual != expected:
    raise RuntimeError('Device bit mismatch')
record('python_complete')
print('PASS: seven BF16 bit patterns; native memcpy16; normal SDK stop', flush=True)
