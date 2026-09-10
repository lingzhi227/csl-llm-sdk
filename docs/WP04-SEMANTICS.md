# WP04 — Pinned text-path equations and compatibility limits

Target: `Qwen/Qwen3.8-27B`, revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`.
Candidate implementation: Hugging Face Transformers commit
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`.
The model configuration declares `Qwen3_5ForConditionalGeneration`, `qwen3_5`,
and nested `qwen3_5_text`, with a recorded `transformers_version=5.8.0.dev0`.
That version string is metadata, not a lockfile tying weights to this candidate.
The audit establishes structural agreement and the equations below; it does not
establish whole-model numerical agreement or successful checkpoint loading.

## Primary evidence and structural agreement

- [Pinned model configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)
  supplies dimensions, layer order, dtype, gate flags, RoPE and epsilon.
- [Pinned candidate implementation](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
  SHA-256 `762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
- [Candidate configuration](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/configuration_qwen3_5.py)
  SHA-256 `19966c3200cee92cc4bccaef5f94550c56d731dbf03858a9bd2ed6aebcc3f7da`.
- Saved headers from all 18 safetensors shards and the pinned weight index match
  851 text-backbone/head tensors in name, shape, BF16 dtype and shard assignment.
  No unexpected text-backbone tensor remains. Another 348 vision/MTP tensors are
  excluded from this text-path contract. This checks metadata, not their contents.

All linear weights use `[out_features, in_features]`; equations below use column
vectors, so projection is `W x`. Unless stated otherwise, projections have no bias.
The 64-layer sequence has 48 DeltaNet layers and full attention at zero-based
indices 3,7,...,63. The candidate's classes and observed tensor families agree.

| Weight name after `model.language_model.` | Observed BF16 shape |
|---|---|
| `embed_tokens.weight` | [248320,5120] |
| `norm.weight`; each layer's two block norms | [5120] |
| `layers.i.mlp.gate_proj.weight`, `up_proj.weight` | [17408,5120] |
| `layers.i.mlp.down_proj.weight` | [5120,17408] |
| Full attention `q_proj.weight` | [12288,5120] |
| Full attention `k_proj.weight`, `v_proj.weight` | [1024,5120] |
| Full attention `o_proj.weight` | [5120,6144] |
| Full attention `q_norm.weight`, `k_norm.weight` | [256] |
| DeltaNet `in_proj_qkv.weight` | [10240,5120] |
| DeltaNet `in_proj_z.weight` | [6144,5120] |
| DeltaNet `in_proj_a.weight`, `in_proj_b.weight` | [48,5120] |
| DeltaNet `conv1d.weight` | [10240,1,4] |
| DeltaNet `A_log`, `dt_bias` | [48] |
| DeltaNet `norm.weight` | [128] |
| DeltaNet `out_proj.weight` | [5120,6144] |

The separate `lm_head.weight` is [248320,5120]; `tie_word_embeddings=false`.
The multimodal wrapper owns `model.language_model`, while the candidate text-only
`Qwen3_5ForCausalLM` owns `model`. Prefix remapping and complete loader behavior
remain integration work, not something established by the shape audit.

## Embedding, block order, RMS and FFN

Embedding selects the token row without a sqrt(hidden-size) multiplier. Each layer
computes `h1 = h + mixer(RMS(h))`, then
`h2 = h1 + Wdown(SiLU(Wgate RMS(h1)) * Wup RMS(h1))`.
Final block output passes through another RMS then the untied language head.
Inference logits are not explicitly upcast to FP32 in the candidate unless loss
computation needs it. These paths are at [lines 824–913](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L824-L913)
and [1222–1308 / 1664–1739](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1222-L1739).

Ordinary RMS is zero-centered learned gain:
`y = cast_input_dtype((x_fp32 / sqrt(mean(x_fp32²)+1e-6)) * (1+w_fp32))`.
It applies to hidden-size block/final norms and head-size Q/K norms. Casting occurs
after gain multiplication; `w=0` means unit gain and `w=-1` gives zero. This is
different from DeltaNet's direct-gain gated norm. The frozen implementation is at
[841–856](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L841-L856).

The declared model dtype is BF16. Linear operations, elementwise activation,
products and residual additions use the actual tensor/operator dtype; backend
GEMM reduction order is not fixed by these Python equations. WP01–WP03's FP32
output contractions therefore establish useful kernels, not BF16-boundary parity
with a complete model execution.

## Full attention and positions

For each token, reshape the 12288 Q-projection outputs to `[24,512]`, split each
head into 256 Q and 256 raw gate values. This is interleaved per head, not all Q
followed by all gates. K/V each reshape to `[4,256]`. Apply the zero-centered RMS
above to Q and K, then partial RoPE; V and the raw gate are not normalized/rotated.
The output is `Wo(flatten(attention(Q,K,V)) * sigmoid(raw_gate))`.
There is no upstream factor of raw gate in this branch. See
[749–821](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L749-L821).

Attention uses scale `256^-0.5 = 1/16`. Q head h maps to KV head `floor(h/6)`.
The eager path adds an additive causal/padding mask, computes softmax in FP32,
casts probabilities to Q dtype, then multiplies V. Dropout is zero for inference.
For query at absolute position p, only existing valid keys at positions <=p are
eligible; cached decode includes the newly appended current key/value. Cache
updates happen after Q/K normalization and K rotation. See
[712–746](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L712-L746).

RoPE dimension is `256*0.25=64`, with 32 frequencies
`inv[i]=10000000^(-2i/64)`. Text-only temporal/height/width positions coincide;
the `[11,11,10]` interleaving selects identical per-axis values in this case.
Frequency/trigonometric evaluation is FP32; cos/sin are cast to input dtype.
Rotate `(x[i],x[i+32])` for i=0..31 using the split-half convention, preserving
dimensions64..255 exactly. Default positions start at cached length and increase
by one; supplied text positions are expanded for the multi-axis interface.
`max_position_embeddings=262144` is a configured capacity, not an automatically
verified bound check. Padding, arbitrary position IDs, multimodal offsets and
mask-builder integration require further integration tests. Sources:
[143–213](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L143-L213),
[666–709](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L666-L709),
[1256–1289](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L1256-L1289).

## DeltaNet projection, recurrence and gated norm

Project QKV to 10240 channels, z to6144, and a/b to48 each. QKV channel order is
2048 Q,2048 K,6144 V. Apply depthwise causal width4 convolution followed by SiLU
to QKV only, then reshape Q/K to16 heads×128 and V to48 heads×128. Repeat each Q/K
head three times to pair with V heads. Convolution casts its input to weight dtype
and returns the original input dtype; the cached state stores pre-convolution
projected QKV history. Sources: [250–289](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L250-L289),
[504–622](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L504-L622).

`beta=sigmoid(b)` is evaluated before the recurrence's FP32 cast.
`g=-exp(A_log.float()) * softplus(a.float()+dt_bias)` is FP32. In the fallback,
Q,K,V,beta,g are then FP32; normalize Q/K with
`v/sqrt(sum(v²)+1e-6)`, and scale Q by `1/sqrt(128)`. For each value head, state
S is `[key_dim,value_dim]=[128,128]`:

```
D = exp(g) * S_previous
delta = beta * (v - k^T D)
S = D + outer(k, delta)
y = q^T S
```

Prediction uses the decayed state, and output uses the updated state. Initial
state is zero unless explicitly supplied; the fallback returns FP32 state and
casts y to original Q dtype. One-token cached decode uses the recurrent function;
prefill/longer sequences use the chunked triangular-system formulation. Their
floating-point reduction order differs; equality must use declared tolerances.
Sources: [294–296](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L294-L296),
[301–434](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L301-L434),
[438–497](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L438-L497).

Per value head, normalize y in FP32 over128 elements, cast that normalized value
to original dtype, multiply direct learned weight w in its tensor dtype, multiply
by `SiLU(z.float())`, then cast to original dtype. Flatten48 heads and project
6144→5120. This earlier cast distinguishes the gated norm from ordinary RMS.
Sources: [218–234](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L218-L234),
[650–663](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py#L650-L663).

## Why `output_gate_type=swish` does not change full attention

The saved candidate configuration/modeling source does not read this extra field.
Branch tracing in primary inference implementations resolves its intended use:
[SGLang qwen3_5 lines355 and462–473](https://github.com/sgl-project/sglang/blob/9a2f17f41d185614a17f740c006c4aca84c016bf/python/sglang/srt/models/qwen3_5.py#L355-L473)
passes it to DeltaNet's gated RMS, whereas
[1441–1448](https://github.com/sgl-project/sglang/blob/9a2f17f41d185614a17f740c006c4aca84c016bf/python/sglang/srt/models/qwen3_5.py#L1441-L1448)
uses sigmoid in full attention. Independently,
[vLLM GDN lines480–492](https://github.com/vllm-project/vllm/blob/9521c60bdc0ccd1264961c284c64f4873335aed1/vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py#L480-L492)
maps `swish` explicitly to `silu` for that same gated norm. Thus the target's
`attn_output_gate=true` and DeltaNet `output_gate_type=swish` agree with the
candidate's distinct branches. This resolves the apparent gate mismatch through
actual field use, not model-name similarity.

For raw gate z=0 and branch input1, sigmoid gives0.5 and swish gives0; for z=-2,
swish is negative while sigmoid is positive. The CPU fixture freezes both branches
for z=[-2,0,2], so silently swapping them cannot pass. DeltaNet uses z itself in
SiLU(z)=z*sigmoid(z); full attention has no such extra z factor.

## Persistent state and reset

| Per layer, batch B | Prefill/decode retained state | Expected dtype |
|---|---|---|
| Full-attention K and V | each [B,4,cached_tokens,256]; append along sequence | normalized/rotated K and V tensor dtype, ordinarily BF16 |
| DeltaNet convolution | [B,10240,4], left-zero padding when fewer than4 tokens | projected-QKV dtype, ordinarily BF16 |
| DeltaNet recurrence | [B,48,128,128], fixed size across sequence length | FP32 fallback state |

Across all layers and batch1, recurrent state alone is144 MiB; convolution state
is3.75 MiB in BF16. The earlier capacity plan used an FP32 convolution-history
profile (7.5 MiB); that distinct storage choice must not be conflated with the
candidate's projected-QKV BF16 history. Full-attention BF16 K+V requires65536 bytes per cached token
across16 layers (16 GiB at262144 tokens), excluding temporary buffers and metadata.
These are shape-derived storage counts, not measured runtime requirements.

The pinned cache implementation lazily allocates conv/state buffers; prefill
retains the last4 projected QKV positions, cached decode updates history and
recurrent state. Reset clears tensors and previous-state flags; a new request must
also reset attention length, masks/positions and any wrapper RoPE offsets. Preserve
generation ownership across all these components. Sources:
[LinearAttentionCacheLayerMixin reset](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/cache_utils.py#L947-L975)
and [allocation/update](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/cache_utils.py#L1017-L1110).
Full cache/mask integration was inspected but not executed in the extracted-body
fixtures; recurrent zero-state and split decode were exercised separately.

## Text tokens and unresolved integration

Saved tokenizer metadata names `Qwen2Tokenizer`, max length262144, no BOS string,
EOS `<|im_end|>`=248046, and pad `<|endoftext|>`=248044. The text config records
BOS/EOS248044 and pad null; generation configuration supplies stop IDs248046 and
248044 and pad248044. Do not collapse these three metadata roles. Chat markers
include `<|im_start|>`=248045, vision start/end248053/248054 and image/video
248056/248057. The complete tokenizer/chat-template pipeline remains unexecuted.

Remaining work includes pinned full-package loader/key mapping, BF16 projection
and residual boundaries, backend kernel substitutions, real layer activation
comparison, actual cache/mask reset integration and generation. These limits do
not block implementing the independently specified full-width RMS operator.
