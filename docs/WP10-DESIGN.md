# WP10: D256 attention with persistent cache8

This package implements one full-size query head and one KV head on one PE.
Synthetic BF16 Q/K are supplied at the explicitly post-normalization/post-rotary
boundary. Original V and raw gate are also BF16. Projection GEMVs, Q/K RMS,
partial RoPE, all24Q/4KV GQA routing and a complete layer are outside this package.

Pinned eager source reference uses Transformers4815a0a6a064214f2d8208c094464a5a6b76ca8d,
modeling_qwen3_5.py SHA256762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b.
Extracted repeat_kv/eager_attention_forward and the output sigmoid multiplication
retain their arithmetic. Observation wrappers return unchanged tensors. The CPU
reference observes BF16 QK matmul/scaling, FP32 softmax, BF16 probability/V matmul,
BF16 sigmoid and BF16 final product. Internal CPU matmul accumulation is not
instrumented; the candidate explicitly uses FP32 reduction and BF16 output casts.
An independent FP64 source oracle preserves each BF16 boundary and propagates
bounded errors through the entire output. Conditional actual-input stage checks
isolate device errors; they cannot replace the original-input final oracle.
The CPU decode and explicit full causal last-row references both pass10tokens,
with0 nominal BF16 differences and single-value output intervals for all2560outputs.

The device appends K/V exactly once into the expected cache slot, computes QK over
only the valid prefix including the current token, rounds to BF16, scales1/16,
rounds again, and initializes maximum from the first valid score. It computes
stable FP32 softmax with the qualified exp helper, rounds probabilities to BF16,
reduces against BF16 V, rounds attention to BF16, applies sigmoid(rawgate) rounded
to BF16, and rounds the final product. There is no SiLU raw-gate multiplier.
All scalar stages, rounded arrays, entire physical cache, inputs, guards, state,
and events are observed at every successful token. Actual exp domains must remain
within[-24,0]. Input bounds guarantee score gaps<=8 and raw gate magnitude<=8.

Eight tokens fill capacity, including a zero-query uniform case, all-negative
scores and current-token dependence. An explicit ninth-position request is refused
before cache or stage writes. All those arrays remain exactly equal to their
pre-overflow snapshots; only error/rejection metadata and events change. A new
generation invalidates metadata while retaining physical cache bytes. The first
post-reset token is zero and must produce exact zero despite stale old slots;
the next token changes the valid prefix. Host readback never feeds candidate math.

The fixed transport sequence is131copies,104757hostu32slots,242088native payload
bytes and13launches. Largest physical host buffer is16392bytes. Initial uploads
set finite cache poison and actual guards; they are not candidate computations.
One runtime persists through both generations and overflow recovery. Each physical
transfer has durable entry/exit records. Finally stops the runtime even on Python
validation failure. Successful result additionally requires normal stop and
unchanged frozen source/compiled hashes; C++ aborts cannot satisfy this gate.

Budget: compile300s; simulation estimated150s with180s hard deadline; at most2
substantively different frozen SDK candidates. Actual ordinary section end plus
4096stack must fit49152bytes before simulation. Workstation single-heavy lock,
20GiB memory/zero swap,8GiB available RAMreserve,20GiB cache/32GiB diskreserve remain.
No downloads, new environment, GPU, MPI or hardware use. WP09 remains unaccepted.

## Exported arrays and command state

`cache` holds two guarded BF16 planes: K[8,256], then V[8,256]. `input` holds
guarded Q/K/V/rawgate[256] arrays. `scores` holds five guarded FP32 groups of8:
QK reduction, scale result, shifted score, exponential and probability.
`score_casts` holds three BF16 groups of8: QK, scaled score and probability.
`stats` holds max, exponential sum and inverse. `stages` holds five guarded FP32
groups of256: V reduction, raw gate, gate exp, sigmoid and output product.
`casts` holds three BF16 groups of256: attention, sigmoid gate and output.
Inactive score/probability entries are explicitly zeroed and checked.

State indices0–15 are generation, last successful token, lifetime successes,
initializations, valid length, phase, error, lifetime overflow refusals,
command-unblock marker, lifetime appends, epoch successes, capacity, head dimension,
exp-domain errors and the two observed input guard words. Successful phase3 and
overflow phase4 are distinct. Overflow changes only error/rejection metadata and
events; retained successful-token/count/cache/stage values are all checked.
Normal events are[1,2,3,4,5,6,7,8,0,0,0,1], tracking admission through completion;
overflow events are[1,99,0,0,0,0,0,0,0,0,0,1]. This single synchronous PE uses no
application fabric colors or asynchronous callbacks. Memcpy owns its normal
SDK resources; the application issues no extra DSR instructions.
