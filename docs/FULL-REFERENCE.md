# Full 64-layer nominal CPU reference and state restoration

The pinned Qwen3.8-27B text model completed a five-token prefill and three
dependent one-token input steps through all 64 layers and the full 248320-token
head. The raw prompt `The capital of France is` uses no chat template or added
special tokens. Greedy decoding explicitly overrides the checkpoint's sampling
default and generates `[11751, 13, 198, 760]`, decoded as ` Paris.\nThe`.

| Input step | Valid context | Next token | Top-two raw logit gap | Maxima |
|---|---:|---:|---:|---:|
| Prefill | 5 | 11751 | 2.75 | 1 |
| Decode 1 | 6 | 13 | 1.5 | 1 |
| Decode 2 | 7 | 198 | 0.625 | 1 |
| Decode 3 | 8 | 760 | 1.5 | 1 |

All 851 text tensors, totaling 53791996928 original BF16 bytes, were bound by
full name, shape and dtype. The reader rehashed all 18 whole checkpoint files
and the tokenizer/configuration assets before use. It loads one decoder layer
at a time; the original full vocabulary-head operation remains intact.

The reference extracts arithmetic and cache definitions from Transformers commit
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`. Model revision is
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. Optional integration decorators are
removed. Two nonpersistent rotary buffer declarations use equivalent
`register_buffer` calls for Torch 2.4, and four read-only observer sites save
actual nonlinear/attention domains. Independent AST review checked these named
adaptations and the unchanged original text-model forward body. A separate tiny
API fixture checked observer and cache-restoration behavior before the full run.

Before each dependent decode, all 128 original cache buffers were restored from
the saved files into fresh original cache objects and checked bitwise against
the live state. Canonical storage is BF16 attention KV, BF16 four-position
projected-QKV convolution history, and FP32 DeltaNet recurrent matrices. Every
step saves all 64 layer inputs/outputs and the three logical stage boundaries.
This validates CPU serialization; device state import/export remains open.

Independent review rehashed 932 archives containing 1476 arrays and checked
252 adjacent-layer links, 12 stage boundaries, 512 saved state buffers, 96 KV
prefixes, 144 convolution shifts and all four full-vocabulary argmax results.
This audit uses saved evidence; it is not a second independent full-model forward.
The 800445000 bytes of arrays remain outside the repository. Their exact hashes
and dtypes are recorded in the [array ledger](../evidence/full-reference-array-ledger.json).

The nominal prefill uses the original chunk recurrence. A diagnostic runs the
original recurrent scan on the same actual operands for all 48 recurrent layers,
without feeding its outputs back. Across these observations there are 200 BF16
output differences, maximum absolute output difference 7.62939453125e-6, and
maximum state difference 1.9073486328125e-6. This does not qualify altered scan
state propagation through the complete model.

Observed domains also identify where earlier device contracts need extension:
MLP gate/up magnitudes reach 25.25/24.625, convolution preactivation reaches
40.75, raw recurrent gate reaches 42, and recurrent decay exponent reaches
-44.20218276977539. Old accepted kernels and guards remain unchanged. The
[domain table](../evidence/full-reference-domains.json) describes this one nominal
request and does not provide a general bound or a device accuracy guarantee.

The bounded supervisor completed in 2431.144 seconds with a 6040936448-byte
memory peak, no swap or memory/task pressure events, and verified unit release.
The independent saved-artifact audit took 7.282 seconds. These are CPU reference
and audit durations, not device inference latency. Complete original CSL layers,
three-stage device state restoration, longer-context execution and full-model
physical generation remain to be qualified.

[Worker source](../examples/full_reference) · [Independent acceptance](../evidence/full-reference.json)
· [Saved-artifact audit](../evidence/full-reference-independent-audit.json)
· [Nominal result summary](../evidence/full-reference-summary.json)
