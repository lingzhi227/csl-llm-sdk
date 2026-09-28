# Routed internal control with full original banks

P37 connects a stage-local SDK gateway to the3554 internal mixer PEs using
addressed command multicast and a credited return path. This control plane is
for initialization, bounded diagnostics and explicit operation admission. It is
not a neural serving schedule, a complete layer or an inference throughput result.
Full compile017 and independent original-bank reference002 pass. Selected
full-bank simulator019 passes572 commands, exact loading/retention of60178
original32-bit bank words, descriptor and bit-pattern checks,19 rounds of
east-side SDK copies and normal stop. This is selected-role control execution. The original P31 physical fused norm/MLP graph remains the neural baseline.

## Routes, arrival and buffer ownership

Each logical32-bit command word becomes two wavelets containing a16-bit target
and a16-bit payload. Exact C12 range filters deliver them only to the addressed
endpoint. The internal port reconstructs all32 payload bits. One command owns
the gateway and return line at a time. Each endpoint has an independent sequence.
C13 forms a fixed serpentine line to the gateway. The addressed source temporarily
switches its receive direction to RAMP, sends the complete response, waits for
output-queue flush, restores the transit receive direction and admits the next
command. DMA completion alone is insufficient to restore this route.

The actual MLP controller borrows its idle frame and packet buffers; its neural
outputs remain intact. A four-word SDK streaming commit arrives on IQ4/C14 and
starts the operation only after all four words arrive. Commit words identify
target, command length, response wire length and total operation sequence.
SDK transfer completion does not mean that the operation has completed. The
host reads the completion counter before reading the response or reusing the
command frame. Device control neither blocks nor unblocks the SDK command queue.
Existing phase guards reject an overlapping controller/norm/MLP invocation.

The controller uses IQ3 for response, OQ7 for command, DSR4/5, UT6/7 and tasks12/13.
Internal control uses IQ0/OQ0, DSR0, UT0, tasks23/24 and the shared SDK teardown
handler mechanism. These resources are separate from the cohosted neural ports.
The streaming color is registered through the SDK layout parameter
`MEMCPYH2D_DATA_1_ID=14`. Only the gateway installs this SDK streaming route;
other SDK roles receive their ordinary parameters, preserving neural C14 routes.

The routed ABI returns aligned16-bit pairs packed in32-bit words. Its header
still reports logical element count and width. Odd16-bit offsets or lengths are
rejected before a transfer. The raw P35 protocol retains its original widened
format. A dispatched command no longer needs its header, so its six-word arena
also holds the four-word response until the response queue drains. Segment IDs,
length bounds, sequence checks, phase guards and explicit root-consumer leases
remain enforced.

## SDK transit is part of the composition

Internal PEs forward SDK commands on C22, H2D on C23 and eastbound D2H on C21.
They have no SDK/RPC consumer. Forwarding a teardown wavelet still sets the local
router's teardown state, so both copy paths must be rearmed by a teardown handler.
The handler clears those flags and leaves the routes intact. Task29 is explicitly
unblocked for both teardown and output-queue flush handling.

A stopped SDK2.10 simulator ELF core from diagnostic016 binds actual code bytes
to their physical positions and shows C23 teardown set on both transit PEs west
of the gateway, with the gateway submission and worker command states untouched.
The compact, hash-bound post-run inspection is retained beside the failed runtime
symbol-capture receipt. The failed capture is not retroactively accepted.

New heterogeneous SDK RPC exports are not used for this control plane. The test
showed that a sentinel-only RPC left the row containing the different gateway
role unable to perform subsequent copies. A normal H2D callback was also rejected:
the installed SDK's callback is a D2H completion hook. The explicit streaming
commit avoids relying on either behavior. The older mixer operand RPC remains
an unqualified prototype and must be replaced or scoped before neural execution
across the heterogeneous SDK roles.

## Capacity and preservation

P36's original joint banks left too little room for routed control plus transit
rearming. The candidate adds256bytes of mixer-only planning reserve to the prior
256bytes, while retaining the existing MLP reserve. Full-K QKV ownership and all
auxiliary locations are lowered together again. No original matrix, state page,
MLP prefix or396953216-byte total bank capacity is removed. Estimates do not pass
SRAM: full compile017 checks every11388 application PE in7714 ELF images against
the same48128-byte ceiling and4096-byte stack allowance. The maximum is48112
bytes. Minimum margins are16bytes on existing MLP projections,112bytes on
mixer standby/bridge roles,160bytes on norm senders and6800bytes on the gateway
(actual41328bytes). These are compiled allocation gates, not dynamic stack measurements.

Independent reference002 verifies every source/destination word exactly once,
all454400 original mixer tiles through emitted descriptors, all53661 auxiliary
pages and every MLP prefix. QKV ownership keeps85 full-K groups with59 rows,
then41,40 and24; other matrix row assignments remain unchanged. The explicit
map relocates33918 auxiliary pages, with31263 outside the mixer region and a
maximum154 Manhattan relocation hops. These are storage placements, not executed
state communication or efficient GDN placement. Future state kernels must use
this exact map and receive separate resource/route/numerical qualification.

Earlier probe attempts preserve compiler omissions, SRAM failures, the rejected
larger transmit refactor, a constant-pointer compiler rejection, RPC/transit
stalls, incomplete debug capture and missing SDK stream metadata. Diagnostic015's
normal-exit flag was insufficient because every symbol read failed; its explicit
`VALIDATION-REJECTION.json` prevents treating it as protocol acceptance. No
failed or diagnostic-only run establishes original-bank or neural correctness.

Simulator018 was proactively stopped after logging command183: observed progress
predicted the whole-bank test would exceed its frozen600-second cap. Commands
1–182 returned, but this is not full retention acceptance. Its original receipt
and released worker are retained. Simulator019 moves width, source-switch and
zero-work epoch checks before bulk transfers and uses an explicit1200-second run
cap (1460-second service,2GiB, no swap, two CPUs). No CSL arithmetic or bank data
changes between018 and019. This duration measures diagnostic simulation, not WSE
inference.

The preflight in019 completed94 commands in25.557 simulator-host seconds. It
checked complete descriptor reads, high16/high32 bit patterns, interleaved source
switching, per-endpoint counters, three zero-work arm/start/finish cycles and
continued east-side SDK copies. Whole-bank loading/readback and normal stop subsequently passed. Zero-work
cycles do not execute a neural contraction.

## Frozen scope of this milestone

The routed control code, shared gateway arenas and SDK transit are actual CSL.
Full-region routes are lowered and the complete original storage/program layout
compiles. Simulator coverage selects eight original role instances (standby
root/nonroot and norm bridge/sender ranks0,1,39), remapped onto a3x3 turning route
with the actual controller and three east-side SDK endpoints. A selected-role
pass does not execute the entire3554-endpoint route or prove its physical timing.
The tests deliberately preserve the larger stage's exact selected bank sizes
and original payload identities rather than use the reduced calibration banks.

Neural ingress fusion from P34 and the physical P31 residual/norm/MLP graph remain
preserved. Complete QKV/Z/A/B/output contractions on the new cohost candidate,
actual conv/gate/state consumers, adjacent layer handoff and model throughput
are still outstanding. No simulator result here passes the full64-layer,
correct-sentence or2000 aggregate generated-token/s acceptance contract.

## Accepted selected control result

Simulator019 completed572 commands through only the actual gateway host symbols.
All60178 original32-bit words across eight complete selected banks were loaded,
read back and retained exactly. All40 descriptor halfwords per PE and special
high-bit patterns matched. Three zero-work epochs reused both standby endpoints;
19 two-word SDK sentinel rounds on each of three east-side PEs passed. Host
simulation elapsed519.961seconds, followed by normal stop. This completed below
600seconds;018's earlier timeout projection was conservative and its cancellation
remains recorded rather than claimed necessary or accepted.

All164 source regression tests and164 publication tests pass. All21 workstation
services for this milestone are released, including failed/cancelled diagnostics.
No WSE job was added. The final ALCF job/system snapshot shows no owned active
job or assignment; provider billed node hours are not available from that API.
The compact summary is `evidence/routed-device-control-summary-001.json`.
