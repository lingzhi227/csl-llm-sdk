# Full136 physical first-fault evidence

On September 22, 2026, physical RUN008 again fails the original strict
650-record criterion. Its new first-fault bank captures a complete, stable
failure before cleanup. Independent review accepts the evidence integrity,
normal context exit, resource release and backup. No transport repair, complete
neural-layer epoch or CSL text-generation result is claimed.

## What the capture establishes

All 408 complete producer source records are exact at enqueue, immediately
before send and source completion. The independent sinks retain 447 coherent
records: 408 producer records, 23 origin records and 16 peer records, with no
torn records. Origin's first 22 records contain 18 READYs, all three group 0
fragments and group 1 offset 0. Record 23 reports protocol error 12 at header
0x00400128, advertised length 8. Every one of the 13 captures shared with RUN004
has identical bytes, shape and SHA256, despite the new observation code.

Two 178-by-128 fault-bank reads are byte-identical. The sole completed fault is
at origin, with pre-failure RX phase 3 (waiting for a control tail), 23 headers,
23 completed bodies, 22 control-tail/native consumptions and one ordinary tail
completion. The unexpected-data-task counter is zero. Its failed RX first8 is
`[10, 1, 1, 4194600, 0, 10, 0, 1]`; the fourth word equals the routing header.
The following 23 RX words match the prior valid group 1 fragment and lie outside
this eight-word receive extent. They are retained buffer contents, not 23 more
words from the malformed READY. The value 3 in the tail scratch is ordinary data.

Origin's TX frame contains the previously completed REQUEST1. Both its frame
and tail completion counts are 2, TX phase is idle, and no source lease is held.
The zero source bank therefore means no qualified current source was copied;
it does not show a producer transmitting zeros. A routing-header value inside
the received body is consistent with mixed packet boundaries, but it does not
identify a sender, a router defect or a unique root cause.

## Observation and lifecycle limits

COMPILE010 covers all 356 actual application programs and PEs, with a maximum
29,792 bytes including a 4,096-byte stack allowance under the 48,128-byte ceiling.
All actual task bindings and function DSR signatures match accepted COMPILE006.
Independent instruction inspection verifies scalar copy helpers and the latch
store before the helpers. The final snapshot stores are word 31=2 then word 0=2,
after all copies. The compiler does not preserve literal source statement order.
These completion markers prove a finished scalar copy; they do not make values
updated by asynchronous DMA an atomic snapshot. Source observations at three
points likewise do not prove continuous immutability.

The run saves 15 arrays totaling 287,736 bytes, using 15 D2H calls carrying
283,776 bytes, two launches and no H2D. Capture including normal exit takes
6.920 seconds; the supervised stage takes 234.018 seconds including setup.
These are diagnostic lifecycle timings, not token latency. The original owner
exits 1 because the application check fails, while the scheduler job finishes
SUCCEEDED. Independent release confirms no system assignment and all three
observed native processes gone. All 71 selected source/receipt/capture files
are independently rehashed in durable backup. No raw arrays or ELF are included
in this public update.

The nine host-source files were separately qualified with synthetic
capture replies, including incomplete, faulted, unstable, failed-copy and
failed-stop cases. That host exercise is separate from physical acceptance.
The original 650 checker and all failure criteria remain unchanged; success
requires it plus two complete, identical and entirely zero fault banks.

The next source proposal checks immutable packet-prefix fields at the completed
body callback, before arming the tail receive. It is a discriminator under
separate review, not evidence of a repair. All 136 concurrency, numerical payloads and the
full graph's phase 2-before-phase 3 dependency remain required. A later pass after
added scalar work would not, by itself, identify a timing cause.

[Exact own source](../examples/ready_fanin/native_control/first_fault) |
[Accepted failure evidence](../evidence/ready-fanin/native-control/full136-first-fault.json) |
[Original full136 failure](NATIVE-CONTROL-PHYSICAL.md) |
[Separate small bidirectional pass](NATIVE-CONTROL-BIDIRECTIONAL.md)
