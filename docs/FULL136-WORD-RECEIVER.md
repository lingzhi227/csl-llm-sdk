# Full136 failure through the origin word receiver

Physical RUN010 on September 22, 2026 still fails the original strict650-record
criterion. The same malformed first eight words arrive through an origin
input-queue data-task path that stores each argument with scalar instructions.
There is no reachable origin bulk header/body/tail receive operation on this
selected path. Bulk origin body DMA is therefore not necessary for this new
malformed prefix. Sender/output behavior, earlier boundaries and routing remain
unresolved; no transport repair or original neural epoch is claimed.

## Actual comparison

RUN010 has440 coherent records:408 exact producer source records,22 origin
records and10 peer records. No record is torn. Its bad prefix is again
`[10, 1, 1, 4194600, 0, 10, 0, 1]`, with network header0x00400128 and length8.
It fails at origin record22 rather than RUN009 record23. Both failures report
error278/protocol22 in phase2, before committing that invalid body.

| Observation | RUN009 | RUN010 |
|---|---:|---:|
| Headers / bodies | 23 / 23 | 22 / 22 |
| Native control entries | 22 | 21 |
| Ordinary tail events | 0 | 0 |
| Coherent records | 447 | 440 |
| Exact producer source records | 408 | 408 |

The complete execution context is not identical. Origin ordering first differs
at record6. RUN010 RX words8–25 retain group0 offset46–63; words26–30 retain
group0 offset41–45 from the preceding fragment. Its TX array retains completed
REQUEST0 while a valid REQUEST1 source is queued and has not been issued.
RUN009 instead retains completed REQUEST1 with no current held source. The
same malformed prefix cannot justify claiming the same complete trace or
identifying a particular producer, router, or source-lifetime defect.

## Qualified path and limits

COMPILE012's actual origin data-task argument, index bound and scalar body
store were independently inspected. The native41 handler blocks picking,
requires the completed-body phase and zero low16 tail payload, then calls the
unchanged application receiver. Common failure blocks the queue before its
first-fault latch. Old DMA receive tasks remain bound but are unreachable from
the selected initialize/data/native-control path; arbitrary external task
activation is outside that claim.

All356 programs fit: maximum32,032 bytes including4096 stack allowance, below
the48,128-byte ceiling. The origin descriptor allocation changes and is reviewed
separately; the other355 function-level descriptor signatures match COMPILE011.
It would be incorrect to claim all356 mappings stayed identical. Fault helpers
remain scalar and complete with word31 then word0, which establishes a finished
copy, not atomicity across asynchronous updates. Producer snapshots still
observe source buffers at three points, not actual staged frames continuously.

The two178-by-128 first-fault arrays match. Capture including normal stop takes
6.630seconds; the supervised stage takes91.734seconds including setup. These
are diagnostic timings, not token latency. The host makes15 D2H calls totaling
283,776 host bytes and stores15 arrays totaling287,736 bytes, with two launches
and zero H2D. The application exits1 for strict failure despite scheduler
SUCCEEDED. Independent review verifies release and rehashes all71 backup files.

The original fixed-length SDK sender, combined-frame sender and managed-DMA
receiver already failed in earlier runs; those paths are not newly proposed
fixes. The next source-only discriminator checks actual sender staging at
three lease boundaries, while a separate explicit-channel design is under
review. Neither is a compiled or accepted transport change. All prior failure
history and the original650 payload/ordering checks remain intact. Complete
original CSL neural epochs and model generations remain zero.

[Exact changed source](../examples/ready_fanin/native_control/word_receiver) |
[Evidence](../evidence/ready-fanin/native-control/full136-word-receiver.json) |
[Previous body-boundary result](FULL136-BODY-BOUNDARY.md)
