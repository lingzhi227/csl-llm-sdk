# Original layer 3: finite FIFO trace and a new physical boundary

The accepted diagnostic combines a bounded SDK FIFO qualification, compilation
of the full original layer 3 graph, and one physical 32 KB observation. It records
20 of 24 attention READY heads, 549 of 576 expected Q/K/V packet observations, and 380
raw FIFO events. Complete original neural epochs remain zero. The next concrete
boundary is missing K/V fanout data at heads 17, 21, 22 and 23.

## Source and actual compiled storage

The application remains 750 × 45: 33750 PEs, including 30576 matrix PEs. All original
weights, inputs, mathematical operations and caches remain in the graph.
Six selected sources append event words to 64/128-word FIFOs. An independently
routed receiver retains a raw finite prefix; source code does not wait for a
per-event callback to publish its next event. The original global and 24-head
sideband publication points remain in place.

The 563 application programs include 532 matrix programs. All 23 embedded CSL files
match the frozen source. Actual ELF inspection checks every PE, separate 4-byte
packet headers and 124-byte payloads, all 8192-byte head KV caches and 5120-byte
attention workspaces, six real source FIFO arrays and 50 real 640-byte idle read
regions. Required live arrays are disjoint. Maximum ordinary section end plus
4096 stack allowance is 47920 bytes, leaving 208 below 48128. This is a static
admission result; peak dynamic stack use was not measured.

Three earlier complete-head trace probes exceeded the ceiling by 944, 448 and 128
bytes. Sharing the existing `trace_event` and `finish` functions with `noinline`
reduced code without changing their event fields or original completion conditions.
The first full-layout attempt failed on an unsupported three-argument range
syntax. Its source and failure remain preserved; compile 002 corrected only the
nine route-loop forms and scope note, preserving every route coordinate. The
successful compile took 187.269656 seconds under its original 600-second budget.

## FIFO SDK qualification

A separate 16-PE simulator fixture measured capacities 64 and 128. It checked exact
legal 62/106-token bursts and a device handshake in which 61/105 tokens reached the
sink before a release producer allowed the source's final marker. Eight D2 H
copies transfer 28992 host bytes after the relevant blocking launch; there are
three launches and 27 journal entries, with no H2 D. Packet/header/payload and the
original sideband buffers coexist and are inspected in actual ELF storage.

A previous host-triggered release design was revoked before execution: installed
SDK D2 H callbacks also run outside the selected rectangle, so a selected-read
assumption was invalid. The accepted device handshake avoids that assumption.
It proves the main task did not finish before the prefix arrived; it does not
prove continuous CPU stall or qualify an SDK synchronous-output stall.

## One original-weight physical observation

The unchanged first 396 host operations upload and verify every original
parameter, initialize, read initial state, upload the first original hidden input,
prepare, read prepared state, and launch compute. Only operation 397 changes:
one read of(748, 0, 2, 25), 160 u32 per PE, 8000 words/32000 bytes. At column 748 the
original 25 records occupy the first 32 words and the remaining 128 words are zero.
Column 749 contains six identified trace histories and 19 all-zero regions.

The read occurred 20.011086 seconds after compute launch returned, inside the
35-second window. All 25 original records are coherent and field-valid; all six
histories and every padding region pass independent raw-word checks. The fixed
sequence completed 394 copies, three launches and 799 journal entries, followed by
normal stop. The independently checked three NPZ files total 3272788 bytes and
remain on the execution filesystem. The 204677-byte journal also remains there.
Small diagnostic words and result ledgers are published for host-only replay.

| Source | Observed events | Last original progress |
|---|---:|---|
| K group 1 at(71, 1) |62|18 source packet completions, all 6 fanout advances |
| Q group 12 at(401, 2) |12|3 source packet completions, one fanout advance |
| Q group 64 at(401, 6) |12|3 source packet completions, one fanout advance |
| Head 0 |98|24 Q/K/V callbacks returned and header receive rearmed |
| Head 3 |98|24 Q/K/V callbacks returned and header receive rearmed |
| Head 16 |98|24 Q/K/V callbacks returned and header receive rearmed |

The sampled global frame phase is 1, waiting for the remaining attention READYs.
All 24 heads have four complete Q collectors. Counts below are packed-u16 pairs;
a complete 128-value collector has 64 pairs.

| Missing head | Observed packets | K counts | V counts | Current missing data |
|---|---:|---|---|---|
|17|21|64, 64|64, 0|V group 5 |
|21|20|23, 23|64, 64|K groups 6 and 7 after their first fragment |
|22|15|0, 0|46, 23|K groups 6 and 7 and remaining V6/V7 fragments |
|23|13|0, 0|0, 23|K groups 6 and 7, V6 and remaining V7 fragments |

The other 20 heads report READY source completion, zero QK/attention runtime error
codes and valid finite summaries. The source roots relevant to the remaining
boundary are V5(291, 9), K6(346, 1), K7(401, 1), V6(346, 9), V7(401, 9).
The prior DSD-header observation had 142 packets and zero READY heads; 519 of its
800 original observation words differ. Changed tracing, shared functions and
allocation/timing prevent attributing this improvement to one cause.

## Limits and next work

Each sink record is independently coherent, not globally simultaneous. A final
source-completion marker is not a delivery acknowledgement. Missing events do
not uniquely identify the current program counter or a permanent deadlock.
Source counters were not read; reaching the maximum sink count does not exclude
later source bound violations. The 1024-f32 QK RMS record aliases later attention
workspace, so the old retained-RMS audit is invalid. These observations establish
neither reference numerical equality nor full-layer/model completion.

The host stage took 251.999086 seconds, of which the capture function used 34.025294
seconds. Scheduler records span initialization, execution and shutdown. These
are not token latency, kernel timing or a verified billing charge. Independent
fresh checks confirmed a terminal successful job, no remaining system assignment
and all three owned processes gone.

The next source design compares moving traces to the incomplete K/V roots and
heads with deterministic device-only sender credits. The progress-oriented
proposal grants one root at a time and joins original source completion with a
recipient's full-row receipt before advancing fanout. Original neural READYs and
all root completions jointly gate the next frame. This is not yet implemented
or qualified. Any new protocol needs bounded synthetic ownership/order checks,
complete-head storage qualification and new full-layout admission. An applicable
normalized-BF16 enclosure audit or retained-RMS evidence plan must be fixed before
a future complete mathematical capture. No host neural feedback is proposed.

[Device and capture sources](../examples/layer3_fifo_trace) ·
[SDK fixture](../examples/fifo_trace_sdk) ·
[Compact evidence](../evidence/layer3-fifo-trace.json) ·
[Original idle words](../evidence/layer3-fifo-idle-words.json)
