# Native control under physical concurrent load

The native-control transport compiled for the complete 356-PE diagnostic fixture,
then failed strict checking on physical WSE-3. Focused two-PE simulator successes
remain valid within their recorded scope. They did not establish correctness
with 136 concurrent producers. Complete original neural-layer epochs remain zero.

## Accepted compilation

Compile 006 retained the formal 178 x 2 topology, 136 READY senders, eight requests,
24 row fragments and 650-record success condition. All six embedded CSL sources
matched the frozen source. The actual 356 programs fit with at most 27,872 bytes
including the 4,096-byte stack allowance, leaving 20,256 bytes under the 48,128-byte
limit. The compiled archive was 2,570,943 bytes with SHA256
b9f38e0921852ae69508ca6022ec1a5667091a2d463ab9a47db3a987aaa3f2c5.

Independent inspection decoded 609,352 instructions over all 356 programs. Origin
RX uses register pair 1, frame output pair 2, synchronous copy pair 3, tail output
DDS5/S1DS4, GO DDS4, diagnostics pair 7 and SDK pair 0. Tail S1DS4 and GO DDS4
are different register families; active RX and GO are disjoint. Peer/producers
use tail pair 4. Native data 6, local 10 and control 41 bindings and the 0x00290000
control payload were checked. Static allocation and fit do not measure all
dynamic interleavings or stack high-water use.

The first independent array audit expected whole-array symbols. The compiler
had scalarized protocol and consumption counters; the corrected audit required
complete indices, exact extents and contiguous storage. No compilation was
repeated. The actual compile executor exited 0 and all owned resources were
independently released before the separately admitted physical run.

## Physical failure and exact comparison

Run 004 used the original 300-second outer limit, 30-second journal-idle limit and
90-second capture/normal-exit limit after context entry. It performed 13 D2H
copies totaling 101,504 host bytes, two launches and no H2D. All 13 raw files,
104,936 bytes, were saved. The runtime stopped normally and the executor exited 1.
Guarded duration was 211.120 seconds: context entry accounted for 202.798 seconds,
and capture including stop took 7.287 seconds. This was an executed protocol
failure, not a setup-only timeout. A fresh audit found the job terminal, no
system assignment and all three owned processes gone.

The independent raw audit found 447 coherent records and no torn columns.
All 408 complete sixteen-word producer records were exact. The origin had 23
records and the peer 16. Origin records 1..22 delivered valid READY owners 0..17,
the three group 0 fragments and the first 31-word group 1 fragment. Record 23 was
an error event with header 0x00400128, length 8 and error 268=256+12. Its terminal
words were [268,12,4,3,0,0,0,0]: the control scratch held ordinary value 3.
Phase 4 is the error handler's post-failure state, not a trace of the prior phase.

Run 003 instead failed at origin record 22 with application error 2 and a network
header in a READY payload. Only the initial and producer capture files are
byte-identical between runs 003 and 004. Receive order changed, and run004
completed one additional valid row fragment. Neither the different failure
location nor normal stop is a protocol repair. Source snapshots establish
contents at three points; they do not continuously observe DMA or arbitration.
The failing full RX body and callback counters were not exported by this
physical capture, so value 3 alone cannot identify a sender or unique cause.

## Preserved validation scope

The original length, tail, payload, order and 650-record checks remain strict.
The success checker differs from the previous physical comparison only in the
two expected network-header constants needed for control termination. The
full fixture has not been promoted to the original neural graph.

One host-only capture-test wrapper expired after its five operations and 34
raw replies had been saved. That wrapper outcome remains failed. A subsequent
read-only audit of the existing files verified hashes, dtypes, intended failure
cases and stop behavior without rerunning capture. These are separate outcomes;
neither constitutes a physical protocol pass.

The next work is source analysis of competing senders, packet boundaries and
control-tail consumption. A proposed smaller multi-source discriminator has
not run and is not an accepted milestone. All previous failures remain recorded
in the [two-PE qualification report](NATIVE-CONTROL-QUALIFICATION.md) and
[physical reproduction history](READY-FANIN-REPRODUCTION.md).

[Exact own source](../examples/ready_fanin/native_control/physical) |
[Compact accepted evidence](../evidence/ready-fanin/native-control/physical-attempts.json) |
[Source hashes](../evidence/ready-fanin/native-control/physical-source-map.json)


## Subsequent matched two-source comparison

The same small three-PE program now passes simulator and physical checks for
four packets and73 words, complete buffers and suffixes, both first-packet lease
overlap witnesses and a stable final cache. The actual physical job stopped and
released normally. This leaves the136-producer physical failure above unresolved.
See the [matched comparison](NATIVE-CONTROL-MULTISOURCE.md) for exact source,
original acceptance receipts and preserved failure history.
