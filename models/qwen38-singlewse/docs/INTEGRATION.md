# Remaining full-model integration

This is the implementation contract for unfinished work, not evidence that a
full resident device program exists. The physical acceptance boundary remains
ACCEPTANCE.md. Storage coordinates alone cannot satisfy it.

## Placement and program sharing

The complete text weight inventory has 17,962 horizontal contraction strips:
15,530 of width 40, 1,216 of width 48, and 1,216 of width 136. Giving every layer
its own rounded vertical region needs 761 columns and fails the 750-column
budget. The mixed patterns in `build_placement.py` fit the weight strips into
1,150 rows. Every scale maps back to its original 128x128 block, including the
eight phases introduced by 272-row tiles. The current BF16 communication candidate uses140x128 tiles after144x128 exceeded the conservative SRAM gate. The new atlas has18,060 strips across1,156 rows,849,313 weight PEs and13,674 free PEs after the1909-cell bus and96-token state. Original144-row atlas files remain for provenance.

`placement-96.json` is a collision-free atlas with 19,503 unallocated PEs after
all weights, recurrent matrices, convolution history and short-context KV.
It does not allocate or route all nonlinear and control roles. In particular,
interleaved strips cannot use the simple rectangular matrix experiment's
vertical broadcast without a new distribution network.

Avoid introducing unique compile-time parameters for each logical layer or
output row. Weight programs need dtype/tile dimensions, scale phase, endpoint
flags and route parity. Runtime identity/configuration can be loaded once with
the weights. Redundant specializations are a material whole-wafer compile risk.

## Device-only token path

1. Input controller receives integer prompt tokens, request identity and bounded
   generation options. Reject prompt plus reserved generation beyond 96 tokens.
2. Gather original BF16 embedding rows from all 40 hidden-coordinate groups.
3. For each of all 64 layers, perform original input RMSNorm and residual setup.
   FP8 projections quantize the BF16 activation in groups of 128. Keep FP32
   partial sums across contraction blocks and round to BF16 only after the
   complete projection. The original BF16 A/B and output-head matrices do not
   receive an invented FP8 activation boundary.
4. Linear-attention layers need original convolution state, Q/K normalization,
   beta/decay, both 64x128 recurrent shards per value head, gated RMSNorm and the
   output projection. Full-attention layers need original Q/K RMSNorm, RoPE,
   full causal KV coverage, softmax, attention output gate and output projection.
5. Apply original post-attention residual, post-attention norm, both gate/up
   projections, SiLU product, down projection and final residual.
6. After layer 63, apply final norm and the complete 248,320-entry BF16 head.
   Reduce argmax on wafer, resolving equal logits to the lowest token ID.
7. Feed the selected token back to embedding on wafer until EOS or the declared
   length. Return token IDs and actual timing observations to the host. No CPU
   hidden state, CPU logits or per-layer weight streaming may enter this loop.

The official model uses the Qwen3.5 architecture identifier; this does not
authorize substituting a Qwen3.5 checkpoint. The pinned current Transformers
arithmetic bodies, original Qwen3.8 FP8 tensors and tokenizer define the CPU
reference profile. Backend-specific FP32 reduction differences still need a
frozen full-model numerical acceptance policy before candidate outputs are read.

## Initialization and evidence

`runtime/weights.py` reads at most 8 MiB per strip and verifies original shard
hashes and tensor extents before packing. FP8 bytes remain packed, scales are
losslessly expanded from BF16 to FP32, and unused rows are zero padded. Its
2,560 FP8 tile and 40 BF16 tile identity audit passed; a complete compiled role
map and full-checkpoint physical upload remain to be implemented.

Record initialization separately from inference. Measure complete prompt time,
first generated token, each dependent decode token and the entire request.
Current microkernel and matrix timing fields must not be presented as full-model
latency or extrapolated into a claimed sentence-generation rate.

Reference `reference-full-002` ended normally with `Hello!` and EOS after two
dependent decode steps. The longer fixed two-sentence workload is in
`reference-full-003`; it completed39 generated tokens including EOS and two continuous sentences. Neither CPU run establishes
physical generation. Preserve every run and confirm hardware resource release.
