# Original stage parameters and the first complete static compilation

September 23, 2026 UTC. Qwen3.8-27B remains at its original dimensions and BF16
parameter bits. The intended deployment uses one physical CS3 sequentially for
layers 0–19, 20–43, and 44–63 with final normalization and all 248,320 vocabulary
rows. This milestone does not establish full-model physical inference.

## Original parameter preparation

Previously accepted preparations for layers 0–3 are reused. Three bounded,
separately owned preparations complete the remaining original tensors:

| Original scope | Tensors | Prepared arrays | Array-file bytes | Independently read original bytes |
| --- | ---: | ---: | ---: | ---: |
| Layers 4–19 | 212 | 2,588 | 12,324,781,056 | 24,353,196,544 |
| Layers 20–43 | 318 | 3,882 | 18,487,171,584 | 36,529,794,816 |
| Layers 44–63, final norm and head | 267 | 3,433 | 17,981,850,784 | 35,527,109,760 |

Read counts include packing and independent original-bit decoding. Every
matrix coordinate, low/high halfword and padded entry is checked. The head
alone contains 1,940 row groups and 54 column tiles per group: 104,760 matrix
PE positions, represented by 198 array files totaling 2,575,874,464 bytes.
The last input tile retains its exact zero padding. Preparation streams one
owner row at a time rather than retaining the whole packed head matrix.

Across these preparations, observed hard-cgroup peaks are 997,101,568,
1,061,376,000 and 1,026,179,072 bytes. Job swap and memory/PID pressure counters
are zero. Every original owner exits and its unit is independently confirmed
inactive, with an empty cgroup and released ownership lock. No SDK or model
forward is invoked by the preparation jobs. Host embedding rows are a separate
causal input, selected from original token IDs; preparation is not a device run.

## Actual first-stage program and SRAM inspection

The actual 404,552,365-byte artifact covers an application of 595 × 1,160 PEs.
Its 24,743 programs cover 690,200 PEs, including 626,100 matrix PEs. Complete
inspection checks 8,957,066 coordinate banks. An independent reader exhausts
all 49 program-index shards, totaling 97,220,498 bytes, checks all hashes,
rejects overlaps, verifies original layer/control identities and independently
recomputes bank address and static SRAM results.

Minimum static margin is **48 bytes** against the actual 49,152-byte ceiling,
including the unchanged declared 4,096-byte stack. The report preserves all
**149 failures** of earlier 48,128/48,640 family policy ceilings. The physical
capacity policy was admitted before compilation. Static fit does not prove
dynamic stack consumption, DSR lifetimes, host-copy correctness or neural
behavior. The small [complete compiler summary](../evidence/sequential-stages/compiled-stage0.json)
preserves these limits and every legacy failure.

The original compiler owner takes 5,741.713 seconds, with a sampled client-group
RSS peak of 1,106,321,408 bytes. Its filesystem-independent parent records a
maximum polling gap of 0.448795 seconds. This observed value is not a future
worst-case cancellation guarantee. The actual backend succeeds and releases;
all four original host owners are independently confirmed absent. A unique
readonly snapshot preserves artifact and metadata with full source/destination
hashes and unchanged original modes and timestamps. It is reused for runtime
binding and preservation; it is not another compilation.

## Remaining qualification

No full-stage physical runtime or full-model result is accepted by this
milestone. Actual host-copy binding, dynamic execution, every original operator
comparison, checkpoint reloads, remaining-stage compilation, full-vocabulary
dependent token generation and matched performance remain open. The previously
published four-layer physical chain and fresh 39-bank restoration retain their
separate, narrower acceptance.

The [selected source](../examples/sequential_stages) is an exact archive of
reusable preparation, head geometry and bounded index-writing components.
Large generated layouts, compiler binaries, original weights, private paths,
raw captures and complete program-index payloads are not published.
