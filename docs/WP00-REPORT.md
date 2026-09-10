# WP00: resource and transfer foundation

September 10, 2026. Independent Qwen3.8 project, limited to resource management,
host wire codecs and a tiny SDK transfer control. No model weights were downloaded.
No GPU reference, neural kernel suite or physical Cerebras experiment was run.

## Actual checks

Standard-library tests exercise BF16 special bit patterns (positive/negative zero,
normal values, infinities, NaN payload), odd element count and offset, empty-input
rejection, wrong dtype/stride, corrupt metadata/padding, exact RAM/disk boundaries,
cache entry bounds/symlinks and lock contention. Low resource conditions are injected;
no memory or disk exhaustion was performed. The initial 12 tests passed on macOS
and Linux; later launcher/preparation fixtures are included in the final test suite.

Reproduction: `python3 -m unittest discover -s tests -v`.
The preparation helper only builds a fresh small bundle and hashes it; it never
starts an experiment or overwrites an attempted run.

## SDK evidence

| Item | Observation |
|---|---|
| SDK | 2.10.1; existing SIF, not copied or downloaded |
| Image SHA-256 | `fff17e81c61dcb6012bdee2941a6fdc570f5c8604967530e7b7108651258193d` from existing installation receipt; no new full-image rehash |
| Container Python | 3.11.8, `/python/python-x86_64/bin/python` |
| Native supervisor Python | 3.13.11 |
| Device geometry | 1×1 application, 8×3 fabric, offset 4,1; WSE-3 |
| Compiler/simulator concurrency | One each, serial; compiler parallelism 1; simulator threads 1 |
| Compiler invocation | Vendor `sdk_debug_shell compile`, standard native memcpy, one channel |
| Host transport | MEMCPY_16BIT, 7 logical elements, 14 logical bytes, 28 host-container bytes |
| Expected and actual low halfwords | `0000 8000 3f80 bf80 7f80 ff80 7fc1` |
| Completion | construct/load/run/H2D/device-copy/D2H/stop recorded; Python complete; both services exit 0 |
| Compile outer wall time | 4.083871 s |
| Simulation outer wall time | 2.058936 s |
| Observed cgroup memory maxima | Compile 451,870,720 bytes; simulate 175,562,752 bytes |
| Allocated run storage | Failed attempt 76 KiB; successful attempt 328 KiB at observation |

The memory values are maxima observed from live cgroup counters/peak files during
one-second polling. The cgroup disappeared at service exit; a final lifetime peak
was not available. Treat these observations as lower bounds, not exact peak RSS.
Wall times include container/service overhead and are not wafer performance results.

Attempt 001 exited 255 before compiler execution: the installed launcher bound the
host TMPDIR to the same not-yet-created container path before mounting its parent.
Attempt 002 was separately approved and froze 11 source files. Its parent work bind
precedes `tmp:/tmp`, container TMPDIR is explicitly `/tmp`, and the vendor compiler
entry remains unchanged. CSL and runtime driver bytes match attempt 001. No repeated
unchanged SDK attempt occurred. The failed service's MainPID=0 and empty ControlGroup
were checked before clearing that specific failed unit; all project units were gone
after the successful run. Cleanup is reported separately from normal execution.

## Guard scope and limitations

MemoryMax=21,474,836,480 bytes and MemorySwapMax=0 were read back from the actual
services. Admission requires room for that ceiling plus 8 GiB MemAvailable reserve.
Runtime checks retain at least 8 GiB available RAM, 32 GiB disk free, and cap the
project cache at 20 GiB. Existing system swap use is not counted as task swap.
The runner serializes heavy work with flock plus host/PID/start-tick/boot identity;
leftover project units block a new launch even after a supervisor crash.

One-second disk/RAM sampling is not a disk quota or an atomic machine-wide memory
reservation. A 64 MiB per-file limit, 10,000 cache-entry bound, TasksMax=128,
CPUQuota=400%, cgroup tree cleanup, 300-second per-step RuntimeMaxSec and outer
watchdog bound this microexperiment. No stress test proved every forced-termination
path, no SDK phase-specific deadlines/reset protocol is implemented, and arbitrary
other users' workloads are outside this task lock. A conservative guard refusal is
a failure to admit, not permission to delete unrelated files or raise limits.

Packed-u32 and raw-u32 formats have CPU tests only. Native memcpy success does not
qualify packed streams, BF16 arithmetic, repeated-request reset, full model semantics,
GPU reference, physical links or one/two/three-wafer inference. Those remain later
controller-assigned packages.
