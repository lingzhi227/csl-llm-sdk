# Physical reproduction of malformed READY messages

A 356-PE synthetic fixture reproduces the network-header value appearing inside
an application packet on physical WSE-3. It uses 136 concurrent producers and no
neural computation or original model parameters. The strict protocol check
fails. All 408 producer snapshots match their expected eight application words.
The root cause remains unresolved; complete original neural-layer epochs remain
**zero**.

## Workload and observations

The 178 x 2 application places its origin at (0,0), phase-2 peer at (2,0), and
136 producers at x40..175 on row 0. Independent vertical diagnostics drain to
idle row-1 sinks. A single GO starts all producers. Concurrently, the origin
requests eight 128-element synthetic rows, using the unchanged production
31/31/26-word row-fragment code. The peer keeps a pending request until the
previous source lease is released.

Every producer records its payload at enqueue, immediately before the SDK send,
and in its source-completion callback. Independent review compared all 136 x 3
records word by word against `[10,1,1,0,0,3,owner,0]`; there were no mismatches.
These are observations at three points, not continuous DMA-memory monitoring.

Origin records first show valid READY owners 0..16 and all three group-0 row
fragments. Its 21st record is a rejected eight-word READY packet:

```text
network header = 0x00400108
application words = [10, 1, 1, 0x00400108, 0, 10, 0, 1]
reason = 2 (identity fields invalid)
```

The network-header value has entered application word 3, which must be zero.
The peer's 13 records reach group-1 offset-0 enqueue and pre-send observations;
its completion was not observed. The complete capture contains 442 coherent
records: 408 producer, 21 origin and 13 peer. No torn record or sink-reported
error was found. Sender append overflow is not exported, so those sink checks
alone cannot prove that every possible sender record was retained.

The fixture does not identify the failing sender. It also does not distinguish
wire framing, transient buffer corruption, custom receive logic or compiler
resource allocation. Static array separation is not dynamic DSR or stack proof.

## Attempts and lifecycle

| Attempt | Actual result | Accepted scope |
| --- | --- | --- |
| SDK001 | Compiled 356 programs; driver exceeded 180 seconds during origin-record read | Six initial/status captures, 35,760 bytes and 23 journal entries; record payload and normal stop unavailable |
| Physical compile001 | Compiled all 356 PEs with exact six CSL source files | Maximum storage plus 4 KiB stack allowance 25,472 bytes under 48,128; artifact 2,011,258 bytes |
| Physical runtime001 | Strict record check failed; runtime context exited normally | 13 captures, 104,936 bytes, 34 journal entries; 13 copies, 101,504 host bytes and two launches returned |

The physical context entered in 192.991 seconds. Capture took 7.091 seconds,
including the 6.761-second context exit. The guard lasted 206.187 seconds;
maximum journal idle time was 5.022 seconds. Those are observed host timings,
not pure fabric latency. Independent release checks confirmed terminal jobs,
no active assignments and gone owned processes; the executor reaped the failed
runtime with exit code 1. Scheduler success means the runtime allocation ended,
not that the application protocol passed. No COMPLETE or successful physical
result was written.

SDK001's complete-looking status counts were never treated as correct payloads.
The SDK attempt was not rerun unchanged. Physical compile and runtime had
separate source manifests and single-attempt admissions. Earlier original-layer
failures, fit failures and diagnostic read timeouts remain preserved in the
[native graph report](NATIVE-LAYER3-GRAPH.md), [archive report](QK-ARCHIVE-PHYSICAL.md)
and [FIFO report](LAYER3-FIFO-TRACE.md).

## Next comparison and limits

A candidate combines the SDK-format network header and payload in
one leased 32-word transmit buffer and injects it in one asynchronous operation.
It preserves all 136 concurrent producers, the receiver, strict packet checker
and neural math. Its first compilation failed before producing an artifact: an interior array
element was passed as a single pointer where a many pointer was required. The
failed attempt took 41.576 seconds and was independently released. A corrected
source candidate adds the explicit pointer cast; no pass is claimed here. A favorable result
would still combine changes to buffering, operation boundaries, output extent
and compiler allocation; it would not uniquely establish an SDK defect.

Any selected transport must subsequently pass complete-graph fit, resource
ownership review and original-layer physical numerics. The full graph currently
has only 576 bytes of static margin. Three sequential stages, state checkpoints,
64-layer CSL text generation and measured token latency remain unfinished.

The repository includes exact device sources, the real record checker and host
capture components, provenance and small independently accepted summaries. Raw
arrays and compiled artifacts remain at the compute site; their hashes and
shapes are in [capture receipts](../evidence/ready-fanin/capture-receipts.json).
These summaries are not a substitute for independently re-reading raw arrays.

[Source](../examples/ready_fanin) ·
[Attempt evidence](../evidence/ready-fanin/attempts.json) ·
[Source correlation](../evidence/ready-fanin/source-correlation.json)
