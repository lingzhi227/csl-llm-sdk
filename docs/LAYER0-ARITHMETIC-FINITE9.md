# Layer0 arithmetic: finite nine-PE simulator qualification

Accepted September 22, 2026 (UTC). This milestone qualifies synthetic arithmetic and
state lifetimes in the SDK simulator. It does not execute original projection
weights, a complete layer, packet transport, a physical CS-3 run, or text generation.

The nine roles are one exp evaluator, Q/K/V convolution owners, four recurrent
value shards, and a scalar-gate/gated-RMS owner. Each recurrent shard holds all
128 key coordinates and 32 value coordinates. The new full 128 reduction order
is explicitly checked; the earlier two 64 key partition's result is not inherited.

All 35,181 finite exp inputs over [-48, 1] pass the original relative budget 2^-20.
The executor observes maximum relative error 7.911365883e-8; independent scalar
reconstruction observes7.911365899e-8. These are finite samples, not a continuum
proof. Convolution, scalar gates and gated RMS include original-observed range
extrema beyond older fixtures. The arithmetic error budgets remain unchanged.

Six computations cover continuation, beta-zero decay, asymmetric initial state,
three zero resets, and exact first-position replay at serials 3 and 6. Every state
word, prediction, correction and output is checked against conditional FP64
bounds and the qualified integer FP32 primitives. An independent column-major
reconstruction checks 294,912 ordered FMAs and 98,304 state words, with zero numeric
or zero-sign mismatches. Every BF16 boundary is checked from actual FP32 values;
parameters are retained exactly. This uses the same qualified integer rounding
primitives and is not a second arithmetic engine.

The full operator audit has 253 passing nodes and 240 conditional-bound nodes;
the maximum error-to-bound ratio is 0.9790609464. FP64 nominal differences are
preserved: serial 4 output maximum 1.0; serial 5 prediction 0.03125 and output 0.0234375.
The adversarial full 128 cancellation result is 1, versus 2 in real arithmetic and 0
under the old two 64 order. Agreement with the accepted FP32 order is distinct
from bitwise FP64 agreement.

The runtime completes 452 operations: 85 launches, 130 uploads and 237 readbacks.
The 367 captured arrays total 1,279,656 bytes, with 1,357,236 host-transfer bytes.
It stops normally, all owned processes are reaped, and resource release is
independently verified. The bounded simulator child takes 954.517 seconds; this
is diagnostic simulator time and is not hardware token latency.

The exact compiler artifacts contain nine programs and 88 active role-array
bindings. Maximum ordinary storage plus the declared 4096-byte stack allowance
is 27,824 bytes, within the 48,128-byte gate. This is not a measured dynamic-stack
proof. Actual compiler output uses DSR1 as well as explicit DSR3/4. A future
packet graph must serialize packet TX and any arithmetic/collective use of
DSR1 through actual completion callbacks; this standalone run has no application
packet TX and does not qualify that coexistence.

## Preserved failures and budget correction

SDK001 failed on unnamed pointer export expressions. SDK002 changed only the
export declarations to named typed pointers. SIM001 retained its original
240-second child timeout and partial captures. Its measured per-batch exp
readback averaged 5.62 seconds and its first 16,384-word state readback took 26.31
seconds, making the original budget insufficient. SIM002 preserved the exact
27 compiler files, all 452 calls, all fixture values and every numeric threshold.
Only its time ceilings and budget documentation changed: 1800-second child,
2100-second outer guard and 180-second progress-idle limit, with the same CPU0,
4 GiB memory, zero swap and storage limits. Neither failed attempt was rewritten.
A missing local admission also produced a local failure before any SSH/SDK call;
that prerequisite repair did not create an extra remote attempt.

An initial independent checker canonicalized 190 zero-beta product signs. Its
corrected version used the existing qualified multiplication primitive and
matched all captured signs. Both checker versions are retained; no simulator
rerun, capture edit or tolerance change was used to obtain the accepted result.

## Published material and remaining work

The [exact source map](../examples/layer0_arithmetic/finite9/source-map.json)
binds all nine CSL files. The [full operator audit](../evidence/layer0-arithmetic/finite9/audit.json)
and [scoped summary](../evidence/layer0-arithmetic/finite9/summary.json) retain the
numeric evidence and limitations. Model arrays, SDK distributions, binaries,
raw captures and private machine inventories are excluded.

Complete recurrent Layer0, dense-stage packet/math coexistence, actual 0→1→2→3
hidden flow, all 64 layers, full-vocabulary logits and generated tokens remain
unfinished. Stage geometry and color assignments are being checked separately;
this arithmetic milestone does not establish full-stage fit or progress.
