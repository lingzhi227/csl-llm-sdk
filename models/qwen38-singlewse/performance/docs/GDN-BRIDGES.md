# Resident GDN color domains

The next integration step separates each input multicast tree from its ordered
return chain. Requiring both directions to use the same physical path was an
unnecessary constraint. Explicit translator PEs connect regions with different
free colors while retaining every original bank and previous router entry.

The candidate adds groups 14 and 15 to P48's three original-coordinate chains.
It uses two existing compute PEs, at (109,145) and (103,60), and 1,155 new router
entries. Group 14 carries seven workers on the source side and 44 through the
translator; group 15 carries two and 40. Source/child partitions, group order,
color availability and the bridge resource reserve are explicit lowering
controls. The other eleven groups remain unresolved. Failure of the current
single-bridge search is not a proof that no static route exists.

## Data and lifetime contracts

A forward translator receives and sends 129-word blocks, preserving all nine
blocks in the three 387-word head packets. Its reverse direction independently
forwards five-word paired-value frames. Separate buffers, descriptors and
microthreads allow the directions to overlap. Input multicast is validated
against two complete filter periods; return walks independently recover switch
order and complete original-value coverage from the emitted router records.

Original workers retain their no-argument control entrypoint 40 and their
pop-on-advance protocol. A translator receives all child frames and original
markers, then emits one weighted completion on entrypoint 41. The frontend has
an additional handler for that count; old markers never become implicit data
arguments. The bridge waits for all forward and data-source callbacks before
issuing the weighted marker, and keeps that marker's buffer leased until its
own callback. A remote frontend completion still does not establish that the
local callback has run. Native projection admission must obey the local fence.

The translator requires an explicit admitted GDN arm. The complete serving
controller, automatic transition back to projection, and all original neural
connections are unfinished. This is not a device dialogue scheduler.

## Actual resources

Corrected exact-bank compilation 043 passes all ten paired programs. A bridge
adds 2,240 bytes on each selected gate/up and down program, and 1,664 bytes on
the selected mixer program. The weighted frontend handler adds 32
bytes. The production candidates retain every original bank byte; 4,096 bytes
of stack are included in all reported SRAM totals. Complete compile 030 admits the initial bridge version on all 11,388 PEs and
8,303 ELF images, with a 48,096-byte maximum. It predates the runtime retirement
fix below. Fresh corrected compile 031 also passes all 11,388 PEs / 8,303 ELF images:
maximum 48,096 bytes including stack, minimum margin 32 bytes, zero overflows.
Its complete emitted bank/native/state descriptors match P48 exactly; the
original reference004 payload proof applies unchanged. The two bridge programs
match the corrected resource043 body used by simulations 010/011.

The resolved bridge resources are:

| Direction | Input queue | Output queue | DSR | UT | Completion task |
|---|---|---|---|---|---|
| Forward on compact MLP | 7 | 6 | 5 | 7 | 10 |
| Reverse on compact MLP | 5 | 7 | 4 | 1 | 18 |
| Forward on mixer | 2 | 6 | 5 | 7 | 10 |
| Reverse on mixer | 3 | 7 | 4 | 1 | 18 |

The compact programs use `copy_transport=true` and have no sender or fusion
actor at the selected cohosts, so their original queue 5 credit receiver and
task 18 sender are inactive. These are phase and specialization requirements,
not globally available resources. The SDK owns command input queue 1 and local
D2H task 21. Preserved attempts 038–041 expose task 14 ownership, the invalid
local task 7, SDK queue 1 ownership and SDK task 21 ownership respectively.
Their resources are released; no hardware job was submitted for these attempts.

The weighted payload uses the SDK's explicit single-command argument API;
[the control library documentation](https://cerebras-sdk-docs-130.netlify.app/csl/language/libraries)
describes that interface. The installed SDK 2.10.1 compiler is the authority for
this implementation's actual admitted resource bindings.

## Static route costs to optimize next

The frozen geometry report records costs separately from measured timing:

| Group | Upstream multicast edges / depth | Child multicast edges / depth | Upstream / child return-chain hops |
|---|---|---|---|
| 15 | 97 / 97 | 41 / 8 | 119 / 245 |
| 14 | 96 / 86 | 248 / 86 | 185 / 116 |

These long return chains are a concrete optimization target. The current
lowerer finds a collision-free candidate while preserving all prior owners;
it does not minimize dependent-path latency or prove global routing feasibility.
Route geometry cannot supply hardware throughput, congestion, or overlap timing.
The manually adjustable domain split, group order, palette and bridge placement
must eventually be optimized jointly with projection/recurrent/MLP ownership.
See `gdn-bridge-spatial-cost-001.json` for exact emitted-plan identity.


A bounded offline reservation probe finds a group 5 candidate only after relaxing
future-worker palette protection. It preserves every existing route, uses a
translator at (140,145), colors (1,9) -> (12,16), and adds 1,500 entries. It is
**not selected or compiled**. Although all future cohorts still have at least two
free endpoint colors, groups 6–9 lose every connected same-color child-return
network: their remaining switch color 13 reaches only seven of 51 workers.
Endpoint color counts alone are therefore insufficient admission criteria.
Future route planning must preserve connected return domains, jointly assign
colors, or split those cohorts further. Probe001 hits its 20-second limit;
probe002 makes the frontend check constant-time and finds the candidate in
9.37 seconds. The source replacement and all scoped diagnostics are retained in
`gdn-bridge-reservation-*.json`; this does not raise the accepted count above five
of sixteen groups.

## Qualification boundaries

Thirty-seven source checks pass, including original-bank identity, emitted
broadcast and return walks, exact bridge boundaries and weighted completion
debts. Three-PE simulations 010 and 011 use the exact corrected compiled bridge
body plus three diagnostic RPC wrappers. Each retains the actual cohost bank
extent, filled with synthetic sentinels, and runs tokens 1, 2, 99 and 100 without
resetting the runtime. Both stop normally.

| Group / actual cohost | Bank words | Frames / child markers per call | Early return words per call |
|---|---|---|---|
| 15 / (103,60) | 7,304 | 160 / 40 | 519 |
| 14 / (109,145) | 4,419 | 164 / 44 | 550 |

Together the tests check 9,288 forwarded words, 6,480 returned words, 336 raw
markers and eight weighted completions. Every call observes all local callbacks
drained before rearm, unchanged bank sentinels and exact payload bits. Early
return counts mean reverse words reach the source before its full forward DMA
callback, proving transfer overlap for this fixture. They are not cycle savings
or inference throughput. A separate audit reconstructs every retained return
word and bridge status from the saved captures and checks exact compiler-source
and parameter identity. Forward and bank arrays were checked by the frozen
runner but are not retained in those captures.

### Completion is a join, including the receive queue

Attempts 001 and 003–008 preserve readback aborts; 002 preserves a host phase
logger error. Preflight 004 reads the same state export successfully before any
wire traffic. Diagnostic 005 changes reverse UT1 to UT6 without fixing the
failure; that binding change is rejected. Device trace 008 receives all nine
forward blocks and forwards all 160 frames, but receives only 39 of 40 markers.
The source has not completed, despite the host launch call having returned.

The reverse data callback blocked its input queue after the final frame. That
same queue still had to deliver the final control marker. The fix keeps it
available until **both** all frames and all child markers have arrived, then
closes it at the complete join. The synthetic source needed the corresponding
fix for its own final weighted marker; 009 preserves the remaining failure
before that source change. Selected 043 and successful 010/011 establish the
corrected bounded protocol. The SDK simulator's signal-11 behavior is retained
as an observed failure, not claimed as a proven independent simulator defect.

The bridge still retains its marker buffer until the transmit callback. The
serving controller must check that local callback fence before beginning native
projection; remote completion alone cannot authorize scratch/DSR reuse.

P47 remains the connected original numerical baseline. P48/reference004 retain
full original bank/address evidence. The new bridges still need runtime and
original numerical qualification, remaining-group lowering, serving admission,
output-projection fusion and complete 64-stage dependent conversation testing.
No component compilation, static route or synthetic wire result meets the
>=2,000 average assistant output tokens/s multi-turn target.

P49 publication checks preserve all 505 original functional files outside the
two authorized status/acceptance notices and all 9,736 previously frozen
evidence files. Thirty-seven site-adapted publication tests also pass. All 19
workstation services are released, no new WSE job is submitted, and the final
hardware audit finds no own active job or assignment.
