# MLP-anchored residency and physical chunk communication

P20 addresses two concrete P19 integration gaps: original resident weight slots
at the new MLP coordinates, and memory-admitted communication helpers. The full
MLP is not executable yet. No new projection latency or model token rate is
claimed. Complete original Qwen3.8-27B-FP8 at >=2000 dependent tokens/s remains
the acceptance target.

## Original weights stay at their compute coordinates

`spatial/mlp_residency.py` assigns all1251 original text tensors and short-context
persistent state within750x1160 application PEs. All64 MLPs share the136-region
geometry from P19. Gate and down tiles share a coordinate in alternating layer
slots; up tiles use one slot per layer at the facing worker. Each original2x128
FP8 tile has its own exact FP32 copy of the original BF16 scale. The complete
embedding and248320-row output head remain present.

The new candidate has860896 matrix banks and9104 dedicated actors. Its5440 down
collectors and360 gate-header turns also hold68 original embedding tiles each:
5800 embedding-only helpers,34816 weight bytes per helper. This removes the old
5784-helper deficit at the metadata capacity level. Assigning embedding-only
data avoids imposing an additional projection operation on those helpers.
The total matrix payload including scale copies is29839851520 bytes.

Both helper embedding rows are stored contiguously for lookup. Other matrix
tiles use the native column-major format. The address record states this format;
a loader must honor it. No per-tile descriptor array is assumed in PE SRAM.
Forty norm owners each hold33024 bytes of nearby original norm gains. All2304
GDN heads have16 contiguous32x32 FP32 state shards per head, placed in4x4 up-worker
blocks. The remaining small original weights,96-position KV state and convolution
history have disjoint128-byte pages after matrix/state payloads.

`mlp-residency-plan-001.json` freezes the candidate and its independent audit.
The audit enumerates870000 physical profiles,36864 GDN shards,103232 used auxiliary
pages and every norm owner. Matrix bijections use disjoint fixed MLP domains,
layer slots and contiguous stream/capacity intervals;998 matrix boundary addresses
and every helper's first/last tile are checked explicitly. The audit does not
read every model weight value. Its maximum allocated matrix/state/auxiliary
payload is35232 bytes, below the unchanged35256-byte bank ceiling.

This is not all-role SRAM admission. Native compute, state operators, norm actors,
all dynamic buffers and control must still be compiled together with these banks.
Other matrices currently have storage addresses without qualified compute locality
or routes. The metadata candidate does not replace or mutate earlier frozen plans.

## Chunk completion drives the next communication stage

`protocols/mlp_chunk_actor.csl` implements the two tested helper bodies. Each
joins its own128-value partial with a child's partial, sending8 FP32 words at a
time directly to the next join. It retains a512-byte own buffer and only32 bytes
of child credit storage. A child receive is rearmed after the outgoing send
callback. The next actor can consume early chunks without waiting for the full
128-value result. This is the communication interface intended for the inter-region
down reduction; the actual down producers remain to be connected.

The header first receives and forwards a65-word input packet through the collector
to a dependency source. Only after that forward callback does it reuse the same
buffer for down partials and rebind its input/output queues. Every warm arm restores
the first input and forward colors. This callback proves this buffer's send is
finished; it is not used as proof that unrelated forwarded native routes drained.

The physical3x3 qualification contains one internal collector and one internal
header, two ungated synthetic sources, one source waiting for the real input
packet, and a kick/result sink. It exercises one color parity with child inputs.
All region parity and leaf variants, original native producers, successor consumers
and the production device epoch controller are still required. The host arm/start
calls are qualification scaffolding, not the intended full-model execution loop.

## Simulator and physical evidence

`mlp-chunk-sim-001` and `mlp-chunk-hw-002` both complete normally. Three ordered
FP32 cases include cancellation, zero and a different nonzero warm input. All1152
intermediate/final values, all callback counters,390 relay packet words,512 lookup
words and both complete original banks plus padding pass exactly.

The two helper banks contain the same68 original embedding tiles from the pinned
checkpoint, independently read after verifying the outside shard hash. Each bank
allocates35256 bytes, including440 bytes of padding. Reading first/last slots and
both row halves after all network epochs checks its actual embedding operation.
This is representative helper qualification, not loading all5800 distinct banks.

The physical artifact has8 program images covering9 PEs. Header SRAM through the
low-section end plus4096 bytes of declared stack is46128 bytes; collector total
is46000 bytes. Margins are2000 and2128 bytes below48128. The stack allowance is a
conservative declaration, not a measured dynamic peak. Neither helper executes
the native projections, and no full-role runtime admission is inferred.

Physical attempt001 compiled successfully, then the host driver executed on import
before the wrapper selected the physical backend. It failed before submitting a
runtime job. Its failure and original cleanup receipt remain intact; a separate
account snapshot established that no owned hardware was allocated. Attempt002
adds a callable main/import guard and reuses the same verified artifact, CSL,
ELFs and fixture. No duplicate compiler job was submitted.

Compile job `wsjob-g8okmzcennfa3krodvnwz8` and runtime job
`wsjob-jt6vmz6ypukdfeqmpdr5eu` both succeeded and released normally. The final
account snapshot has151 terminal owned jobs, no active owned jobs and no owned
system assignment. Other assignments are untouched. Compile/run lifecycle
durations are71.4004/81.3882 seconds, including service overhead and checks;
these are neither inference timings nor provider billed node-hours. The workstation
service also exited and released its heavy-job lock. All69 source tests pass.

## Next executable boundary

Connect these ownership and communication interfaces to the complete original
MLP: native gate/up contraction, packed activation fusion, group quantization,
down routing/contraction, chunked residual consumption and the actual next-layer
interface. Lower the missing RMS, group max/scale, gather/broadcast, route-drain
fence, readiness and reset dependencies; admit the combined role programs against
actual SRAM. Then measure the complete graph, including redistribution and stalls.
Further optional fusion is guided by that path, not a prerequisite checklist or
an open-ended series of isolated helper benchmarks.
