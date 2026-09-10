# Selected-head DeltaNet input preprocessing

This package selects matching128-channel Q/K/V groups from the architectural
head mapping. It implements384 independent width4 temporal convolutions,
activation, Q/K L2 normalization and scalar update/decay gates. Inputs and
weights are synthetic BF16. It does not implement projections, all10240
convolution channels, head repetition, recurrent composition or a complete layer.

## Pinned boundaries

The reference source is Transformers commit
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`, with model implementation SHA256
`762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b`.
`causal_conv1d_update` lines250–267 concatenates old history and new projected
input in the weight dtype and copies the last four projected values into history.
Its convolution output is BF16, followed by BF16 SiLU output. Weight index0
multiplies the oldest of the four causal samples, index3 the newest.

The recurrent body first promotes Q/K/V/beta to FP32. `l2norm` lines294–297 uses
sum(x*x)+epsilon, not a mean. Q is then divided by sqrt128. V is an exact BF16
expansion. The forward assignments at lines617/619 produce beta via BF16
sigmoid(b), whereas g uses `-exp(A_log.float())*softplus(a.float()+dt_bias)`.
With the selected BF16 parameter profile, dt_bias promotes to FP32 in that sum;
g and exp(g) remain FP32. Candidate beta is explicitly RNE-cast to BF16 before
its FP32 expansion.

The CPU reference extracts the unchanged convolution and L2 bodies, removing
only optional hub decorators. It executes the unchanged beta/g assignment AST
nodes with explicit dependencies. Cached update is compared against an entire
causal-prefix replay on every token, independently of the mathematical oracle.
History is compared exactly. This is selected-function qualification, not a
complete Transformers runtime.

## Arithmetic gates

All four convolution products and accumulation occur in FP32 before explicit
BF16 RNE. The FP64 convolution gate uses gamma8 times the absolute-product sum.
The repaired bounded exponential uses range reduction and a degree8 FP32
polynomial; see [candidate diagnosis and error accounting](WP08-CANDIDATE-REPAIR.md).
SiLU evaluates exp(-abs(x)) and a sign-dependent sigmoid branch, then rounds
its FP32 result to BF16. Both pre-cast vectors and BF16 bit vectors are observed.

L2 gates propagate gamma256 sum error, denominator rounding and independent
sqrt/inverse approximations from actual BF16 activation values. Separate tests
check normalization against the actual inverse and Q scaling against the actual
normalized Q. Exact V expansion and beta promotion are checked independently.
Conversion is always checked against actual preceding FP32 bits; mathematical
BF16 parity remains a separate observation. This avoids silently treating
rounding discontinuities as continuous error propagation.

For scalar gates, the device records t=a+dt_bias, sigmoid's exponential and
pre-cast beta, exp(A_log), exp(-abs(t)), the log1p ratio and series, softplus, g,
exp(g) and expanded beta. Actual-argument exponential, sqrt and inverse gates
have2^-20 relative allowances. SiLU, sigmoid, log1p and softplus use2^-18
relative allowances. Direct FP32 additions/products and the log1p ratio have
their own rounding gates.

Softplus is max(t,0)+log1p(exp(-abs(t))). To avoid cancellation for negative t,
log1p(e) uses r=e/(2+e) and seven odd terms:

```text
2*(r + r^3/3 + r^5/5 + ... + r^13/13),  0 <= r <= 1/3.
```

The truncation remainder is at most `2*r^15/(15*(1-r^2))`. This is only the
exact-arithmetic truncation bound, not the total implementation bound. The
implementation also incurs FP32 ratio/series arithmetic and error in exp(-abs(t)).
The independent log1p gate uses the actual exponential argument; the independent
softplus gate compares against FP64 exp/log1p of the actual t. Both therefore
check more than the series truncation alone.

The declared finite BF16 domain has abs(input)<=1, abs(weight)<=1, abs(a/b)<=8,
abs(A_log)<=1 and abs(dt_bias)<=0.5. The chosen cases stay in finite normal
arithmetic or exact zero; overflow/subnormal correctness is not claimed.

## History and resource ownership

One PE owns384×4 BF16 projected-input history and immutable384×4 BF16 weights.
A vector command shifts one position and appends its input exactly once, then
enters pending-gates. A separate finalize command materializes observed global
parameters, computes scalar gates and commits the token. The host checks the
intermediate phase and unchanged parameters; it performs no candidate arithmetic. Valid
history length saturates at four. Reset zeroes all1536 entries and generation
token counters while preserving total-token count. The host uploads history's
guarded initial buffer only once; all subsequent history mutation/reset is device
code. Input, parameter and weight arrays are checked for immutability.

Six consecutive tokens cross the four-slot warm-up boundary. The first two
include isolated V and Q/K impulses, with asymmetric weights to expose reversed
orientation or look-ahead. Reset is followed by zero then changed projected
input. Every call observes the complete history, all vector/scalar stages,
guards, stable handles and generation/token counts. Sixteen signed RNE probes
exercise the conversion path each call.

Nominal guarded buffers occupy about15 KiB before code/runtime allocation.
The compiler inventory must admit the actual highest application section end
plus4096 stack bytes within49152 before simulation. No application fabric queues,
colors, microthreads or explicit DSR leases are added. The frozen ledger records
compiler-managed arithmetic and memcpy reservations.

Eight tokens are planned below150 seconds with a300-second hard simulation
deadline; the estimate is not a throughput guarantee. Compilation is limited to
300 seconds. One heavy task,20 GiB RAM, zero swap, available-RAM/disk reserves and
cache bounds remain unchanged. No environment install, checkpoint or GPU is needed.
