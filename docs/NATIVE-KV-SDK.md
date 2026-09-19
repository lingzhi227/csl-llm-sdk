# Native KV multicast and bounded Q retention: accepted SDK fixture

Three normal operations, including a reset, completed in the SDK simulator.
Independent review checked all raw payloads, canaries, source and receiver
ownership, deliberately delayed streams, exactly-once synthetic consumers,
normal stop and resource release. This is transport qualification with synthetic
data. Complete original neural layers and full-model CSL generations remain zero.

## What runs on device

The 9×5 layout has 45 PEs: six receiver heads, four native KV sources, 24 Q
packet roots, one origin, six observer sinks and four idle PEs. The heads use
the original-sized QK and attention input arrays. Four independent native
128-halfword streams write directly to their K-low, K-high, V-low and V-high
slices. The Q packet path retains four 128-halfword rows per head and preserves
Q and the raw gate before those arrays are reused. There is no neural arithmetic
or model weight in this fixture.

Each source keeps its actual buffer until its source completion callback.
Each receiver joins four distinct completion tasks with Q retention and receipt
source completion before its synthetic consumer executes once outside packet
receive ownership. Source completion releases local source memory; it does not
prove that every network queue has drained. Native payloads have no identity
header: fixed routes, invocation identity, local retirement and the prior command
barrier establish their ownership. Recovery after an aborted transfer requires
reinitialization, not an assumption that a new begin call flushes the fabric.

The three operations use positions 0, 1, then position 0 with a new reset epoch.
A different receiver and stream is held in each operation until Q preservation
and the other three native callbacks complete. In operation 2, a native source
also remains live while a real packet receive releases it. Observer traffic and
the actual SDK command service coexist with those transfers. Every head checks
all retained values and guard words before publishing readiness.

## Independent checks and resource evidence

| Check | Accepted result |
| --- | --- |
| Actual compiled programs and placement | 45 programs, all 45 PEs |
| Maximum ordinary storage plus 4,096-byte stack allowance | 25,376 bytes, below 48,128 |
| Exact synthetic payload | 29,184 BF16 halfwords across three operations |
| Canary words | 192 halfwords |
| Actual invalid API calls rejected | 525, after the normal captures |
| Application packet messages | 451 |
| Host operations | 8 launches, 13 D2H copies, 209,664 host bytes |
| Saved evidence | 13 immutable raw captures, 226,723 bytes, 47 journal events |
| Guarded elapsed time and peak cgroup memory | 197.475793 seconds; 220,430,336 bytes |

The run used CPU 0, one simulation thread, 1 GiB hard memory, zero job swap and
128 processes. The runtime child had 300 seconds within a 330-second outer
deadline. File, log and candidate limits were 8 MiB, 2 MiB and 32 MiB. All owned
processes exited and the unit/cgroup were freshly confirmed absent. The 63
accepted compilation files were reused unchanged; three SDK runtime-generated
auxiliary files matched the known final names, sizes and hashes.

The negative phase calls the real shared lifecycle APIs after all three normal
captures. It checks rejected identity, duplicate and illegal ownership operations
without sending malformed fabric messages. Its poisoned state is not reused as
a fourth normal operation.

## Preserved failed attempts

| Attempt | Result and correction |
| --- | --- |
| SDK001 | Frontend rejected the identifier `packed`; no ELF or runtime. SDK002 renamed it to `pack_ok` in three shared credit files. |
| SDK002 | Compilation passed. The 90-second runtime child timed out during the second operation's readback; the first operation's four copies had taken about 64 seconds. In-memory raw captures were lost. The next candidate reused all compiled files, added durable incremental captures and used a separately reviewed runtime budget. |
| SIM001 | A strict output inventory rejected three normal SDK-generated auxiliary files after 4.73 seconds. All 63 original files were unchanged. SIM002 separated the pre-load, live-write and final exact-hash checks. |

Every failed attempt and its resource-release evidence remains preserved. None
was retried unchanged. Each correction had a new frozen source identity and
separate admission. No extra compiler run was used for either runtime continuation.

## Limits and next step

The current accepted physical layer-3 diagnostic remains at 20 of 24 attention
heads ready and zero complete neural epochs. This fixture does not prove that
the complete original head/root programs fit, that the original layer is
numerically correct, or that native transport improves inference latency.
Its local timestamps include setup and observers and are not synchronized
cross-PE clocks. A selected complete-program fit and a subsequent full graph
compile must precede original-layer hardware qualification.

[Exact source](../examples/native_kv_sdk) ·
[Acceptance summary and failure history](../evidence/native-kv-sdk/acceptance-summary.json) ·
[Raw result](../evidence/native-kv-sdk/result.json) ·
[Source mapping](../evidence/native-kv-sdk/source-map.json)
