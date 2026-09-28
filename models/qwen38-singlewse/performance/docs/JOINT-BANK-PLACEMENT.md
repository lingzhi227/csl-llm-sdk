# Joint original matrix and auxiliary-bank placement

P36 changes the QKV output-row assignment of complete40-PE contraction groups
so the existing residual/RMS and quantization cohosts can retain their complete
programs. All original weights and expanded FP32 scales stay resident. This is
one layer00 placement candidate; it neither substitutes a smaller model nor
reuses the wafer as a sequential whole-model compute overlay.

The original88 groups still cover all10240 QKV output rows and all5120 input
columns. The first85 groups own59 output tiles each; the last three own42,41,22.
Each output tile contains two rows. A worker retains its original K128 index,
uses the same40-PE adjacent FP32 reduction order and applies the same BF16 output
rounding. Only the independent output-row assignment and corresponding packed
bank offsets change. The other four mixed matrices retain their row assignments.
The maximum QKV row-loop count remains59; no throughput result is inferred.

`joint_placement.py` jointly chooses bank extents using final P35 control-port
calibration and the actual complete P31 MLP ELF census. It reserves256 additional
bytes at changed bank allocations. These estimates choose a candidate; the
complete candidate compiler and ELF coverage check determine SRAM admission.
The unchanged49152 GDN pages represent two full48-head128x128 FP32 states.
All53661 original mixer auxiliary pages, including original convolution/gain
weights and other state/value storage, receive unique128-byte locations inside
the same78x146 layer rectangle. Total original layer bank capacity remains
396953216 bytes. Original MLP prefixes remain at the same local addresses.

24155 auxiliary pages borrow measured MLP bank slack;27058 pages change PE.
The placement uses nearest available Manhattan slots after reserving each PE's
local pages, but constrained capacity produces up to133 hops from an original
owner. These are relocation distances, not executed communication measurements.
Future GDN/conv consumers must follow the explicit new state locations, place
arithmetic near resident state and qualify their routes. Do not reuse the old
region-only auxiliary resolver or the historical800-page P31 remap as the active
candidate map. `JointAuxiliaryPlacement` resolves every original logical page,
including request/head/key/value coordinates, through the candidate spans.

The new descriptor checks cover all454400 original mixed tiles. A separate
remote payload verifier copies every original bank word exactly once from the
hash-pinned P31 fixture, then re-reads all mixer tiles using the actual emitted
CSL descriptors rather than the transfer's candidate address resolver. It also
checks every auxiliary page and unchanged MLP prefix. Model arrays and compiled
ELFs remain on remote storage; source and compact identities are published.

Internal device-control modules are present, with C22 forwarding compiled for
SDK endpoints farther east. The gateway command/response paths are not connected
in this candidate and C22 forwarding has no runtime qualification. Full-bank
SRAM admission must not be described as a runnable complete neural stage.
Original GDN/attention operations, adjacent-stage admission, real request
isolation and complete64-layer sentence/speed acceptance remain open.

## Executed capacity and payload evidence

Complete local compile016 passed in324.199742407s. All11388 application PEs are
covered exactly once by5891 ELF images. Maximum48112bytes includes the unchanged
4096-byte stack allowance; the overall minimum16-byte margin is an existing MLP
program. Internal mixer standby, norm sender and norm bridge minima are272,368,
and352bytes respectively. The controller occupies39632bytes. These are actual
compiler results, not the planning reserves above, and are not physical execution.

Remote reference001 passed in29.449652461s total. All396953216 original bank
bytes are consumed and written once across515894 transfers. Every one of454400
mixed projection tiles, including expanded FP32 scales, matches the actual CSL
setup-driven readback;53661 auxiliary pages and all original MLP prefixes also
match. The source bank hash is
`284942f0addfbb717f18e3dd6f2918d6ee0aadafb789308a2a4d058ce76277d8`;
the new bank hash is
`2d2babbe18e6a17c3bd346f1a3b9048b327d182c760e88213452c9b2e69c7e2c`.

All160 source and160 publication regression tests pass, including independent complete QKV row/K
coverage, all49152 disjoint request-state addresses and malformed-map rejection.
Both bounded workstation invocations have released their processes/cgroups and
the shared lock. No new WSE job was submitted; ALCF accounting reports no owned
active job or system assignment. Provider billed node-hours are not exposed.
