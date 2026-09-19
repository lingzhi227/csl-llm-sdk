# Native control termination qualification

This milestone qualifies finite two-PE protocol behavior on a static route and
with actual SDK message routing in the SDK simulator. It does not demonstrate that the physical 136-producer fixture
or full neural graph is repaired. Complete original neural-layer epochs remain 0.

## Why the boundary changed

The original physical fixture delivered a network-header value inside a READY
payload despite exact source records. Combined-frame TX and compiler-managed RX
both compiled, but their physical runs still failed. The managed-RX run returned
13 captures byte-identical to the combined-TX run. Those failures are unchanged.
The new source keeps full 32-bit application words and explicitly consumes native
control termination after the finite payload receive has completed.

An earlier filtered receive experiment failed compilation with an undefined
LLVM dfilt operation and was closed. Native control consumption uses the SDK's
existing task model instead. Control task 41 was checked against the pinned SDK
reservations; no copied example's task number was assumed to be available.
Source review, compiled resource checks and actual runtime checks remain distinct.

## Actual build changes

| Build | Difference | Application ELF bytes | Storage including4KiB stack allowance |
|---|---|---|---|
|014 | Native termination and explicit unexpected-data rejection |30752 /35400 |11104 /12928 |
|015 | Extend state30 to33 words; observe diagnostic counters inside state |30824 /35408 |11136 /12960 |
|016 | Immediate consecutive31then8 and full per-packet evidence banks |39192 /37824 |14768 /14192 |
|017 | SDK message routing replaces the static route; actual bankA readback |39528 /38176 |14912 /14336 |

Packet source is identical across these builds. Build 015 adds12 declared state
bytes; build 016 adds224 declared harness bytes relative to015. All 14 named u32
banks in016 have exact sizes, four-byte alignment and no overlap. Actual 016 and 017 compiled
communication operands use RX pair 1, frame TX pair 2, synchronous copy pair 3 and
control-tail TX pair 4; SDK memcpy uses pair 0. This is scoped static allocation
and fit evidence, not an all-task dynamic exclusivity or stack high-water proof.

## Runtime outcomes

| Attempt | Actual outcome |
|---|---|
|003 | Valid1 protocol and normal stop pass. Entire suite fails post-run inventory because the SDK adds three files. Original11 files remain unchanged. |
|004 | Valid8 passes. Valid26 child exits signal11 during a final 8-byte counter read, without normal stop; valid26 is not accepted here. The runtime cause is unproven. |
|005 | Only valid26 passes after015 aggregates diagnostic counters into the complete state snapshot;6 raw files, 2,456 bytes and normal stop. |
|006 | Valid31 plus early/extra/duplicate/missing-control cases pass;30 raw files, 12,280 bytes and all five normal stops. |
|007 | Both packets deliver, but strict initial/final counter stability fails: native count changes 1 to 2. Entire case remains failed. |
|008 | Strengthened host completion gate selects a complete baseline; all strict consecutive checks pass with11 raw files, 5,328 bytes,11 D2H copies totaling 2,424 host bytes,5 launches and one 8-byte H2D. |
|009 | Source binding rejects a stale verification contract before SDK execution. The frozen candidate is preserved; no runtime was attempted. |
|010 | Actual SDK message routing passes the same complete consecutive checks, actual bankA=[4,4],11 raw files, 5,328 bytes and normal stop. |

All executed attempts were actually waited for, reaped and independently checked for
resource release. Later runtime candidates pin the three generated SDK files by
exact path, size and hash in addition to all 11 original compiled files. The
inventory correction does not retroactively turn003 into a successful suite.
No unchanged failing device execution was automatically repeated.

The accepted008 driver retains the existing eight-poll limit, 25 ms interval and
copy/launch bounds. It requires live delivery/completion counts 2 plus cached
receiver trailer [0,2,0] and sender trailer [0,0,0] before choosing the baseline.
It then performs the same full payload, order, lease, suffix and final-stability
checks. Every one of the original007 raw files is byte-identical in008; one extra
completed snapshot supplies the correct baseline. No original failure is hidden.

Build 017 removes the static route and calls SDK mp.init. Runtime 010 verifies
actual bankA=4 on both PEs, then the same two packets and strict checks. Its eight
non-state captures are byte-identical to008; all three state captures differ only
in the two bank-enable fields. Whole resource release and all 14 artifact
identities were independently verified after the actual owner was reaped.

The next gate is the 136-producer physical fixture, followed by fresh full-graph
fit and neural checks. A small simulator pass alone does not qualify a larger
topology or model inference.

[Exact own source](../examples/ready_fanin/native_control) ·
[Compact evidence](../evidence/ready-fanin/native-control/attempts.json) ·
[Source identities](../evidence/ready-fanin/native-control/source-map.json) ·
[Earlier physical reproduction](READY-FANIN-REPRODUCTION.md)

## Subsequent physical comparison

The accepted simulator scope above was followed by full356-program compilation
and physicalrun004. Compilation passed; the physical protocol failed on ordinary
value3 at a control-tail boundary. This does not revoke the focused simulator
results or extend them to136 concurrent producers. See the [physical comparison](NATIVE-CONTROL-PHYSICAL.md).


## Subsequent matched two-source comparison

The same small three-PE program now passes simulator and physical checks for
four packets and73 words, complete buffers and suffixes, both first-packet lease
overlap witnesses and a stable final cache. The actual physical job stopped and
released normally. This leaves the136-producer physical failure above unresolved.
See the [matched comparison](NATIVE-CONTROL-MULTISOURCE.md) for exact source,
original acceptance receipts and preserved failure history.
