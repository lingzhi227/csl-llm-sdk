# WP14 device handoff proposal

Status: independently accepted for one five-PE simulator call. The connected
case is original dense hidden case0, position1, all5120 columns, Q256/K256.
The host schedules scalar commands and supplies original inputs/parameters. It
never uploads projected Q/K values into the consumer. V, gate, attention, cache,
full layers and hardware remain outside this package. WP09 remains unresolved.

## Physical placement and leases

Application rectangle5 by1; producer ranks0..3 at(x=rank,y=0), consumer at(4,0).

| Rank | Global WP13 slab | Role | Role row offset | Consumer offset | Data color | ACK color |
|---|---|---|---|---|---|---|
| 0 | 0 | Q=0 | 0 | 0 | 2 | 6 |
| 1 | 1 | Q=0 | 128 | 128 | 3 | 7 |
| 2 | 4 | K=1 | 0 | 256 | 4 | 8 |
| 3 | 5 | K=1 | 128 | 384 | 5 | 9 |

Each route has exactly one source and one destination. For rankr data color2+r:
producer RAMP->EAST, intermediate x in(r,4) WEST->EAST, consumer WEST->RAMP.
For ACK color6+r: consumer RAMP->WEST, intermediate EAST->WEST, producer
EAST->RAMP. PEs left of the route have no route configured for that color.
No packet merging, multicast ACK, or concurrent producer arbitration is assumed.

Each producer uses OQ2 for its data color and IQ2 for its ACK color. The consumer
uses IQ(2+r) and OQ(2+r) for data and ACK for rankr, respectively. Input/output
queue namespaces are separate. Local tasks10(receive complete)/11(send complete),
UT2, destination DSR5/source1 DSR5 are reserved for transport on each PE. A PE
performs send then receive, or receive then send; it never reloads those DSRs
before the prior async completion. These are the qualified raw32 WP07 primitives.

The GEMV retains destination/source0 DSR3 and destination/source0/source1 DSR4.
The scalar Q/K kernel has no explicit DSR lease. SDK memcpy reserves colors20..23,
IQ/OQ0..1, local tasks21/24/27/28/30 and its control tasks. Application colors2..9,
queues2..5 and tasks10/11 do not overlap those reservations. Compiler/runtime
resource validity and actual placement passed the accepted SDK compile. Any new
placement or source revision requires its own compile and SRAM admission.

## Wire format and ownership

All frames use raw32 asynchronous copies, with one zero-extended BF16 word per
payload slot. BF16 words undergo integer narrowing only at the destination;
floating arithmetic never performs transport conversion.

Data is138 u32 words,552 bytes:

| Index | Meaning |
|---|---|
| 0 | Data magic0x514b4441 |
| 1 | Version1 |
| 2 | Generation |
| 3 | Token |
| 4 | Physical producer rank0..3 |
| 5 | Global projection slab0/1/4/5 |
| 6 | Role Q=0/K=1 |
| 7 | Role-relative row offset0/128 |
| 8 | Count128 |
| 9 | Consumer offset0/128/256/384 |
| 10..137 | Exact low16 BF16 words, high16 zero |

ACK is8 u32 words,32 bytes: magic0x514b414b, version1, generation, token,
producer rank, global slab, destination offset, status(0=committed, nonzero=error).
The physical send/receive arrays also have one guard word on each side; guards
are outside the fabric descriptors. Data storage140 u32 and ACK storage10 u32
are retained for final readback. All bounds and fixed metadata are checked before
writing any of the128 consumer words. A complete frame is consumed before parsing.

Host handoff_control is8 u32 words: generation, token, rank, global slab, role,
role-relative offset, count, destination offset. It is loaded identically to all5
PEs before each arm command. These are scalar metadata, not candidate operands.
The receiver requires a matching initialized generation/token, idle handoff,
rank-specific mapping above, destination+128<=512, and an unset rank commit bit.
Only after all header and high16 checks pass does it copy into input[1+offset:]
and set the rank's bit. Input order is Q256 then K256, with outside guards.

Producer GEMV FP32 and BF16 outputs stay unchanged from finalize through ACK and
postprocessing. A successful ACK marks the producer as handed off; it does not
reset or overwrite its projection buffers. A later explicit release is allowed
only after that flag is set. The receiver owns its independent copied operand.

## Commands and completion

Every exported command has an explicit implementation on every application PE.

| Command | Producers | Consumer |
|---|---|---|
| initialize_consumer | Immediate no-op completion | Validate new generation/token/position1; clear mask/counters, retain guards/parameters |
| begin | Existing WP13 begin/zero accumulator; reject unreleased generation | No-op completion |
| accumulate | Existing128x112 persistent serial FMA, last tile80 | No-op completion |
| finalize | Existing final BF16 RNE; retain FP32/BF16 and guard snapshot | No-op completion |
| arm_handoff | No-op completion | Validate scalar descriptor; arm selected full-frame receive; complete arm command |
| handoff | Selected rank sends retained BF16 frame, then waits for ACK; other ranks no-op | Join command arrival with receive/commit/ACK-send completion |
| preprocess | No-op completion | Require mask15, Q/K counts256 each, all4 ACK sends complete, no error; run RMS/RoPE; unblock after outputs/state complete |
| release | Require own successful ACK, then existing release transition | Require processing complete; no operand mutation |

The host calls arm_handoff then handoff for each rank0,1,2,3 serially. No next
descriptor is uploaded until the previous handoff command finishes on all PEs.
Consumer receive may finish before its handoff command arrives. It records
completion and joins both events: command_arrived && ack_send_complete &&
!already_unblocked. A prepare-command unblock is distinct from handoff completion.
The selected producer arms its ACK receive only after its data-send completion.
Consumer ACK transmission can backpressure until that receive is armed. Distinct
data/ACK routes and completed full-frame receive prevent a cyclic buffer wait.

Commit mask, receipt/ACK counts, received words, last metadata, per-role counts,
processing count and event ordering are observed. Duplicate/stale/missing frames
cannot qualify preprocessing. Invalid complete frames receive an error ACK based
on the expected descriptor and put the run in a terminal error state. No automatic
recovery/retry is claimed. Host protocol tests exercise the model only; malformed
device commands require separately dispatched device tests.

## Observation and resource obligations

Keep all WP13 projection observations: original tile bytes/input poison,
FP32 boundaries at112/5040/5120, BF16 RNE, states, timing and guards. Move each
full final resident-weight read and original-bit check after all handoffs and
consumer preprocessing. At that same point, read each producer's guarded FP32
and BF16 output again and compare every word with its finalize snapshot before
release. The complete weight read is moved, not duplicated. Readback is
observational and never feeds the consumer.
After each handoff, read producer/consumer protocol states and compare final
consumer input words against producer BF16 words bit-for-bit. Read both wire
storage arrays/guards for the active endpoints. After preprocessing read the
unchanged input, original norm weights, frequencies, probes, RMS stats/stages,
normalized casts, trig/casts, rotary stages/product casts, output, state/events
and control. Check parameter/control identities and handle stability.

Use the old conditional actual-operand stage validator where applicable and the
new original-hidden source intervals as separate gates. Exact tail and signed
zero/cast checks stay explicit. Reconstruct physical counts with the same sequence
function used by the driver; durable journal entries identify both halves of every
copy/launch, symbol handle, PE, shape, dtype, byte count and payload SHA.

Producer baseline ordinary end37472 plus4096 stack leaves7584 bytes; transport
adds data560+ACK40+control32+state96 bytes, plus symbol stubs/code. Consumer WP11
baseline ordinary end20080 plus4096 leaves24976 bytes; new framing/state replaces
its scheduling wrapper. These are planning comparisons, not placement proofs.
The final driver must inventory every array/stub and compile-admit each of5 ELFs
using actual ordinary section end+4096<=49152. Maximum physical host buffer stays
57352 bytes for a resident weight tile, below65536.

The equal-FMA WP13 diagnostic took233.914793 seconds, including all4 whole-weight
readbacks. The first integration proposal targets<=270 seconds and hard300, with
compile hard300. A single WP11 call contributes approximately5.6 seconds including
its observations. Four serial ACK exchanges and extra protocol/consumer readbacks
have an unmeasured cost; the runnable sequence must make the remaining allowance
explicit before controller review. If the reviewed allowance is not credible,
request a separately named communication diagnostic; do not shrink GEMV width or
claim integration from host-injected projections. No SDK launch is authorized by
this document.
