# Full136 failure after actual sender-frame checks

Physical RUN011 on September22,2026 still fails the original650-record transport
criterion. All136 producers reach exact source-completion records after three
checks of their actual staged frame, held source and native-control tail.
No sender check reports fault23,24 or25. The origin nevertheless receives the
same malformed prefix `[10, 1, 1, 4194600, 0, 10, 0, 1]`. This narrows the
observed failure boundary but does not establish a repair or unique cause.

## What the actual instructions establish

COMPILE013's138 sender programs contain seven guard variants. Independent
instruction review verifies actual header, tail, source and staged-payload
loads at three points: before frame issue, after frame completion before tail
issue, and after tail completion before releasing the source. A failed guard
enters the existing first-fault path and bypasses the protected action. An
existing protocol error also returns before those actions. Tail comparison is
not constant-folded. New guards use no descriptors, microthreads or activation.

The compiler removes the source's greater-than31 length rejection under this
fixture's reachable invariants. Actual loops remain bounded to31 and check
index against length before each paired load. This does not qualify arbitrary
memory-corrupted-length rejection. Source pointers specialize to the actual
origin tx, peer rows.tx and producer tx allocations.

All356 programs fit: maximum33,056 bytes including4096 stack allowance, with
15,072 bytes below the48,128-byte ceiling. Function descriptor signatures match
COMPILE012, including that compile's separately qualified origin allocation.

## Physical result and comparison

RUN011 retains440 coherent records:408 exact producer records,22 origin
records and10 peer records. No record is torn. It fails with error278/protocol22
at origin sequence22, before accepting the invalid body. All136 producers'
completion records establish that each traversed its three guard checks.

The complete retained first-fault metadata,31 RX words,32 TX words and31 source
words match RUN010. All10 peer records also match. Earlier origin arrival order
differs at records2,3,9,10,12,13,14 and15, so the full origin trace is not equal.
The TX array retains completed REQUEST0 while a valid REQUEST1 source is queued
and unissued. A completed stable scalar copy is not an atomic DMA snapshot.

Capture including normal stop takes7.158seconds; the supervised stage takes
217.449seconds including setup. These are diagnostic timings, not token latency.
The host makes15 D2H calls totaling283,776 host bytes and stores15 arrays totaling
287,736 bytes, with two launches and zero H2D. The application exits1 for strict
failure despite scheduler SUCCEEDED. Independent review confirms release and
rehashes all73 backup files.

Three point checks do not observe every DMA read or injected wavelet. Source
and staging could share transient corruption between observations. Output
queues, routing/merge behavior and earlier packet-boundary handling remain
unresolved. Prior fixed-length and receiver variants already failed and are
preserved; another unchanged receiver retry is not the next step. Replacement
qualification uses explicit full-packet software relays and separate READY,
REQUEST and row channels with independent equivalent acceptance. It remains
separate from this failed diagnostic and from original neural inference.

All previous failure history and strict650 checks remain intact. Complete
original CSL neural epochs and model generations remain zero.

[Exact changed source](../examples/ready_fanin/native_control/sender_frame) |
[Evidence](../evidence/ready-fanin/native-control/full136-sender-frame.json) |
[Previous word-receiver result](FULL136-WORD-RECEIVER.md)
