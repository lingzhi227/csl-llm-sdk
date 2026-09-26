# Numerical qualification for the complete resident backend

Contract fixed before reading the new `full-model-hw-003` capture. The original
strict whole-prefix comparisons remain in every physical receipt, including
failures. This additional contract qualifies BF16 execution across backends;
it does not assert bitwise identity with CPU Torch or model quality on a corpus.

All 24 layers and both actual autoregressive steps must be present. Tokens must
match the unchanged original reference (13225 -> 11 -> 5922). The physical run
must verify all active/inactive PE epochs, upload every original tensor once,
exhaustively read back unchanged weights, stop normally and release its jobs.
Only the initial/feedback token and fixed control commands may enter after
initialization. Diagnostic arrays are read after a complete token and are never
fed back to the device. CPU replay is post hoc validation only.

## Local numerical checks

Each check uses the actual captured input at that boundary, rather than hiding
accumulated error in a relaxed whole-model tolerance.

* RMSNorm: within one BF16 ULP of FP64 evaluation with direct ties-to-even BF16
  quantization. Direct quantization avoids FP64 -> FP32 -> BF16 double rounding.
* Router: every logit lies in the rounding interval for its exact dot product
  with the original matrix and bias; top four are descending score, lower expert
  ID first for exact ties. Torch's different choice among equal values is
  recorded. Selected softmax is within one BF16 ULP of the original operation.
* Each selected expert: every gate/up and down value lies in its FP32 dot-product
  rounding interval with original decoded MXFP4 weights and actual captured
  input. Preserve BF16-before-bias and BF16-after-bias boundaries. SwiGLU is
  compared to the original function on the captured interleaved input, within
  one BF16 ULP. A separate absolute bound of 1e-32 is allowed only for values
  discarded by the documented exp cutoff below -80 (|gate| <= 48 in that
  boundary region, |up + 1| <= 8, exp(-80) < 1.81e-35).
* Weighted expert reduction and residual: verify using original BF16 expert
  outputs/probabilities, a four-term FP32 summation bound, BF16 round, then
  residual addition and BF16 round.
* Attention: independently replay the original attention on the sequence of
  actual incoming states, substituting the separately qualified captured norm
  outputs at that boundary. Require finite output, L2 error no greater than one
  BF16 epsilon (2^-7) times the L2 norm of the reference residual update, and
  maximum absolute error no greater than 2^-7 times max absolute reference
  output or update. Report absolute/ULP errors as well. This is a numerical
  integration test of attention, not a proof for all inputs; the earlier
  attention-prefix hardware test additionally checked persistent K/V directly.
* Full vocabulary head: original head on actual last-layer state must select
  the same token as the device and the end-to-end reference at each step.

For a chain of exact BF16 products accumulated in FP32, use
`gamma(n) = n * 2^-24 / (1 - n * 2^-24)` times the sum of absolute products.
Expert tiles have 288 terms followed by nine chain additions (n=297). Dense
router tiles have 96 terms followed by 29 additions (n=125). Bias additions have
their own FP32 rounding allowance. Interval endpoints are rounded monotonically
to BF16 at the same operation boundaries. This handles cancellation without
choosing a tolerance based on the observed output. Checks reject nonfinite data.

Acceptance retains capture/checkpoint/source hashes and all strict-reference
differences. Two generated tokens validate the bounded physical integration;
they do not qualify long-context accuracy, arbitrary-prompt quality or throughput.
