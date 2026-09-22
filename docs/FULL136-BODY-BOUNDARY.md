# Full136 failure before current tail receive

Physical RUN009 on September 22, 2026 still fails the original strict 650-record
criterion. The added body-completion check detects the same malformed prefix
as RUN008 before the current packet's tail receive is armed. Independent review
accepts the boundary evidence, normal exit, resource release and durable backup.
This narrows the fault location; it is neither a transport repair nor a complete
original neural-layer epoch or CSL text-generation result.

## Comparable observations

The first 22 origin records and all 16 peer records match RUN008 exactly. All
408 complete producer source records remain correct. The 31-word retained RX
bank and 32-word completed REQUEST1 frame also match. The first failed packet
is again origin record 23, header 0x00400128, length 8, with received prefix
`[10, 1, 1, 4194600, 0, 10, 0, 1]`. RX words 8–30 retain the earlier group 1,
offset 0 fragment outside this packet's receive extent.

| Observation | RUN008 | RUN009 |
|---|---:|---:|
| First error / protocol reason | 268 / 12 | 278 / 22 |
| Receive phase before failure | 3: waiting for tail | 2: body completed |
| Headers / bodies completed | 23 / 23 | 23 / 23 |
| Native control tails completed | 22 | 22 |
| Ordinary data tail events | 1 | 0 |
| Tail scratch word | 3 | 0 |

These five metadata differences are the only changes in the selected fault
bank. Both complete 178-by-128 reads agree within RUN009. The 447 diagnostic
records are coherent with no torn records; the original checker rejects the
reported protocol error. A zero current-source bank has no valid source lease
and does not establish an actual zero-valued transmission.

## Meaning and limits

Actual COMPILE011 instructions place the origin and peer scalar prefix loads
and validity branches before the tail descriptor is armed. RUN009 takes the
new failure branch in phase 2. The current packet's later tail receive or native
callback therefore cannot alone account for a prefix already invalid there.
Earlier packet or tail handling, boundary misalignment, fabric contention or
merge behavior, body DMA/descriptor lifetime and sender lifetime remain open.
The header value inside the body does not uniquely identify a sender or defect.

All 356 actual programs fit, with a maximum 30,192 bytes including the 4,096-byte
stack allowance under the 48,128-byte ceiling. Independent review checks actual
task bindings, function DSR comparisons and scalar fault-copy closures. The
latch precedes copy helpers, and the final stores commit word 31 then word 0.
This proves a completed scalar copy, not an atomic view across DMA-updated
words. The three producer source observations also remain point observations.

The unchanged host capture saves 15 arrays totaling 287,736 bytes: 15 D2H calls,
283,776 host bytes, two launches and no H2D. Capture including normal exit takes
6.983 seconds; the supervised stage takes 204.182 seconds including setup.
These are diagnostic timings, not token latency. The owner exits 1 for strict
application failure, while the scheduler job finishes SUCCEEDED. Independent
review confirms no assigned system and all three observed processes gone;
all 71 selected files are independently rehashed in durable backup.

The next diagnostic proposal examines origin input-queue data-task delivery
while keeping 136 producer concurrency, strict checks and the full model's
phase 2 before phase 3 dependency. It is not an accepted runtime or a fix. A
pass after changing receive service or timing would not prove a unique cause.
RUN008 and all earlier failures remain preserved. Full protocol success still
requires all original 650 records and two complete, identical, zero fault banks.

[Exact source delta](../examples/ready_fanin/native_control/body_boundary) |
[Evidence and source map](../evidence/ready-fanin/native-control/full136-body-boundary.json) |
[Earlier RUN008 first-fault evidence](FULL136-FIRST-FAULT.md)
