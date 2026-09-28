# P42: complete compact-bank frontend compiler admission

The original layer00 banks and all16 shared-Q/K convolution/gate frontends now
fit together in a complete11388-PE program. Compile022 covers7919 ELF images,
with a maximum48112bytes including the declared4096-byte stack allowance.
Every coordinate is covered once. This is complete storage/compiler admission,
not execution of a complete neural layer or a conversation-speed result.

## Lossless weight and communication placement

P41's unchanged matrix ownership was too large at communication cohosts. The new
integer lowering keeps every original row and every40/48-part K contraction,
but assigns contiguous QKV, Z and output rows jointly with packet destinations.
The40-PE and48-PE groups overlap in independent240-PE components. A bounded
integer dynamic program maximizes row capacity within measured code budgets;
a monotone reservation pass then accounts for the resulting merge locations.
All existing norm/MLP/control/projection routes remain byte-for-byte unchanged.

An original2x128 FP8 tile contains256 code bytes and one FP32 scale from its
original128x128 block. Consecutive row pairs on one PE now share that exact scale
word in a local table. The reader uses46-word descriptors: the existing five
projection descriptions plus explicit local scale bases. No weight or activation
is requantized. Matrix prefixes, auxiliary pages, software queues and their
program costs are independently represented in the lowering.

Remote reference001 classifies and freshly verifies every98436512 original
source word. It checks all454400 mixed tiles, all28605 retained auxiliary pages,
and the unchanged MLP prefixes. It reconstructs readback addresses from the
actual CSL descriptors. All433136 removed scale copies are bitwise equal to
their retained aliases.450560 old scale words become17424 local scale words.
Bank storage decreases by1732544bytes, from393746048 to392013504bytes. All
original weights, one complete conversation and both work slots remain;
3301 retained pages change physical PE while keeping their logical page IDs.
The removed second-context holes remain invalid addresses.

The full compiled stage uses890048 fewer aggregate SRAM bytes than P40, after
including the new frontend programs and their caches. Minimum measured margin
is208bytes on fused roots,224 on mergers,320 on frontend consumers, and16 across
the entire stage. These margins include the declared stack allowance, not a
measured dynamic stack peak. Future recurrent actors still need fresh admission.

## Native packet fusion and transport retirement

Roots send their existing leased five-word frame directly:
`[token, row, rows, packed BF16, matrix]`. Semantic decoding moves to the
destination, reducing the selected root's additional code cost from P41's
388bytes to68bytes before the compact reader. No separate root packet copy is
introduced. The compact reader adds64–80bytes in matched selected programs.

The new routes batch up to three roots onto one horizontal row with independent
colors and retain only needed destination groups. There are88 roots,168 merge
actors and16 consumers, with311 root-to-group paths. Source traffic remains
8288 five-word packets; multicast produces29367 received projection packet
copies across the consumers. Foreign rows are discarded only after checking the
token and frame. Transport retirement waits for every projected packet copy and
the192 eventual recurrent-result frames per group. Semantic output retirement
alone cannot authorize a new token while foreign packets remain in flight.

Reducing1216 merge actors to168 does not establish a runtime speedup. Native
frames trade root parsing/storage for extra destination packet processing.
Actual numerical execution, backpressure behavior and service time still need
device qualification. Recurrent-core return paths and automatic output
projection ingress are not connected by this milestone.

## Preserved attempts and validation

Backend024 fails on a16-bit target address relocation in the native frontend.
Explicit range-checked index masks correct it in025. Selected025–027 programs
compile, but their intentionally unchanged consumer banks exceed the SRAM gate;
they are preserved calibration failures, not whole-stage admissions. Packing
only enabled merge mailboxes in026 does not show a code-size improvement and
is not reported as one.027 measures the compact scale reader. The complete
relocated-bank compile022 subsequently passes every SRAM gate in508.256seconds.

Thirty-four selected source tests pass, covering complete original tile and
scale-block identities, dialogue page conservation, exact native multicast
ownership, full transport drain counts, rejected corruptions and P41/P40/P39
regressions. All34 site-adapted publication tests also pass, including exact
regeneration of the admitted CSL and bank metadata. This is not a full-suite run. Remote bank verification takes39.776
seconds. Original arrays and ELF images remain remotely stored; only source and
compact evidence are published.

All four selected compiler services, the complete compiler, and the bank
verification service release. No new WSE job is submitted. The closing account
snapshot has no owned active job or hardware assignment; unrelated assignments
are left untouched. Provider billed node-hours are unavailable in that snapshot.

Evidence: `layer-backend-compile-024` through `027`, `layer-mlp-compile-022`,
`layer-frontend-reference-001`, `compact-frontend-census-001`,
`compact-frontend-tests-001.json`, and `resource-audit-p42-close.json` under
`performance/evidence`.

Next qualify the new frontend against independent original-parameter references,
then connect real GDN state traffic, recurrent returns, gated output projection
and the existing norm/MLP graph. The full64-stage stateful multi-turn2000tokens/s
target remains active and unmet.
