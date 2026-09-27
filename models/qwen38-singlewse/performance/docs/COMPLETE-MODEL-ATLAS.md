# Complete model ownership and lifetime lowering

P12 supplies explicit physical ownership for the complete pinned model. It is a
metadata milestone, not a compiled executable, numerical result or speed result.
The complete physical 2,000 dependent tokens/s target remains unmet.

`spatial/atlas.py` lowers the original graph and tensor headers into
`evidence/model-atlas-001.json`. It reads no weight payloads. The graph, headers,
configuration and rotary-frequency metadata are hash pinned. All 1,251 original
text tensors, 498 matrices, 64 layers, 96-position state and 1,332 produced values
are included. The embedding remains a lookup and the untied full-vocabulary head
remains a GEMV; their dimensions and original BF16 storage are unchanged.

## Geometry and ownership

The application remains 750 by 1,160 PEs. Twelve full actor rows occur at
`y = 48 + 96*i`; another 120 actor PEs occupy the final physical row's left tail.
The remaining 860,880 PEs form a serpentine FP8 bank ring. This count is divisible
by all complete K widths: 40, 48 and 136 groups of 128 input elements.

Every GDN value head has one controller and a nearby 4 by 4 state rectangle.
Each state PE holds a 32 by 32 FP32 shard, or 4,096 bytes, alongside its FP8 bank.
Thus all 2,304 heads use 36,864 state PEs. This reduces the planned elements per
state PE from the functional baseline's 8,192 to 1,024. It does not prove faster
state updates: the new distributed arithmetic, reduction order and communication
must still be implemented and qualified. Key and value axes remain complete.

State PEs do not carry BF16 matrix banks. An explicit spare 4 by 4 state-class
rectangle makes the remaining BF16 ring 824,000 PEs, also divisible by K=40.
Compact monotonic segments map its IDs to physical bank IDs. Its phase is 412,000,
which avoids combining 111 FP8 slots with 13 BF16 slots on the same PE.

Each of the 64 attention KV groups has a controller and 32 cache actors. The
cache partitions both time (eight pieces of 12 tokens) and channels (four pieces
of 64). Every actor stores both K and V in original BF16, for 3,072 bytes. Total
KV capacity is 6,291,456 bytes. Controllers remain on distributed actor rows.
Attention arithmetic and this finer state partition are not yet qualified.

The 353 non-matrix original tensors occupy 209 nearby 32-KiB canonical pages,
with explicit byte slices and no page aliasing. Planned GDN parameter/history
replication and normalization gain replication have separate reservations;
canonical ownership is not evidence that these replicated values are uploaded.

## Compact matrix dispatch

For each precision class, the start of every matrix is aligned to its number of
K blocks. A tile's class owner is `(phase + stream_start + tile) % class_pes`,
and its local slot is `(stream_start + tile) // class_pes`. Matrices retain
original row order and complete 2 by 128 tiles. FP8 scales are exact FP32
expansions of the original BF16 scale selected by `[output_row//128, k_block]`.

The FP8 stream contains 95,027,200 real tiles and 9,056 explicit padding slots;
BF16 contains 10,024,960 real tiles and no padding. Resident matrix payload,
including padding and replicated scales, is 29,842,206,080 bytes. The original
1,251 tensors themselves total 29,468,003,328 bytes.

The complete model uses 620 dispatch segments rather than a per-tile table.
A segment describes a contiguous interval in normalized class-owner space,
one local slot, and its first tensor row. No segment splits a K group. This is
an address/selection representation; multicast and reduction routes are still
an explicit next lowering stage.

| FP8 slots | BF16 slots | State bytes | PE count | Payload bytes | Bytes left for code, stack and scratch |
|---:|---:|---:|---:|---:|---:|
| 110 | 0 | 4,096 | 21,504 | 32,696 | 15,432 |
| 110 | 12 | 0 | 362,960 | 34,744 | 13,384 |
| 110 | 13 | 0 | 136,960 | 35,256 | 12,872 |
| 111 | 0 | 4,096 | 15,376 | 32,956 | 15,172 |
| 111 | 12 | 0 | 324,080 | 35,004 | 13,124 |

The ceiling remains 48,128 bytes. These are data reservations, not compiled SRAM
admission. New roles must include actual code, route state, buffers and the
unchanged declared stack before physical dispatch. P11's different executable
does not establish that these new profiles fit.

## Distributed values and manual controls

The 4,352 quant/value actors each represent four elements of a 128-element group;
32 actors constitute one group and 136 groups cover the widest 17,408 vector.
Produced values use striped 16-byte cells, with explicit birth and last consumer.
Lifetimes include both endpoints so an operation cannot overwrite its input
while producing an output. Allocation is aligned to 32 cells. The selected token
remains live through the feedback event.

The arena high-water extent is 81,600 cells, with at most 19 local cells or 304
bytes per actor. All 129 normalization gains are assigned to the corresponding
input owners, reserving 1,032 additional bytes per actor. An optional 28,672-byte
capture reservation per actor can retain every layer output, final normalization
and full logits for all 96 positions (111,575,040 total BF16 bytes). This leaves
18,120 bytes for code, stack and other scratch in this planned role, still subject
to actual compilation. Diagnostic storage does not define a speed measurement.

The policy exposes actor-row location and spacing; unsafe overlaps are rejected.
Other current dimensions are deliberately constrained to this checked model and
short-context profile. The JSON exposes per-matrix segments, role coordinates,
state rectangles, tensor slices, lifetime intervals and gain owners for review.
Future route/queue/thread choices must be separate checked decisions, not hidden
inside the final CSL layout. There are 142 spare actor PEs and one global control
actor; an efficient event schedule must avoid a global barrier per operation.

## Independent audit and remaining gates

`tools/audit_atlas.py` does not import the builder. On the workstation it visits
all 105,052,160 real matrix tile addresses in bounded chunks, reconstructs their
tensor coordinates from dispatch descriptors and verifies unique physical slots.
It independently checks the full geometry, excluded bank ring, all state axes,
canonical tensor byte coverage, role exclusivity and per-PE data census. Pairwise
checking covers all 4,851 overlapping value lifetimes. All checks pass in 4.757
seconds with a measured peak RSS of 422,316 KiB, under a 2-GiB/no-swap/240-second
service with two permitted CPUs and the shared heavy-job lock.

The frozen `atlas-audit-001` sources, manifest, result and workstation release are
retained. No physical job was submitted. A fresh hardware accounting snapshot
finds no owned active job or system assignment. Twenty-five source tests pass.

Next gates are actual co-resident bank/state code admission, original numerical
state qualification, complete matrix routes and resource leases, composed actor
code, and full model feedback. Only the complete physical sentence workload can
establish the requested token rate.
