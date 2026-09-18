# HW01: complete original layer-3 MLP on physical WSE-3

One physical runtime on September 18, 2026 executed the complete original
Qwen3.8-27B layer-3 MLP: 5120 inputs, 17408 intermediate channels and 5120 outputs.
All 534773760 original BF16 weight bytes remained resident for four frozen module
inputs: dense, changed, last-column one-hot and zero after nonzero. This result
qualifies the MLP module at those inputs. It does not qualify a decoder layer,
full-model generation, token accuracy or general bitwise reference parity.

## Device computation

The 184×314 application rectangle contains 21968 matrix PEs and 136 nonlinear
PEs, inside the 762×1172 WSE-3 fabric at offset (4,1). Each matrix PE stores one
128×96 BF16 tile and computes FP32 local accumulations. Gate/up rows span 54
column shards, and down rows span 182. The right-to-left FP32 chain consumes
actual local partials. Root BF16 conversion, SiLU, product and downstream input
broadcast all execute on device. Host diagnostic readbacks do not feed neural
results back into the graph. Tail input/weight cells are explicitly zero padded.

There are 57776 application PEs, including routing and idle positions. Replicating
this standalone rectangle per layer cannot fit the three-stage target. Stage
integration must reuse the matrix kernels and qualified arithmetic in the
existing compact placement, with real routes, live state and compiled SRAM gates.

## Evidence and precision

The capture completed 253 copies and nine launches, including four complete
epochs and final readback of every original weight and padding word. The
predeclared audit checked 11247616 actual-operand local row enclosures. A separate
independent integer/Decimal auditor checked 87872 exact FMA sample rows, covering
every matrix PE for every input; it also checked all 159744 chain rows and 69632
nonlinear rows, casts, handoffs, input tails, state and zero-after-nonzero behavior.

Nominal official BF16 output differences were **39, 43, 1 and 0** out of 5120
outputs per input. All frozen source and actual-operand bounds passed. The
accepted FP32 reduction order differs from the nominal CPU reference; this is
numerical qualification under the published [precision contract](../examples/hw01/capture/FULL-PRECISION.md),
not bitwise official parity. Existing full CPU references were reused, not rerun.
The exponential-domain check remains specific to these qualified inputs.

All 1224 application ELF identities and embedded CSL sources passed inspection;
their unique physical mapping covers the entire application rectangle. Maximum
ordinary static storage plus the 4096-byte stack allowance was **38032 bytes**.
This is a static bound, not a dynamic stack measurement.

## Lifecycle, timing and preserved failures

The physical job succeeded and exited normally. Fresh scheduler and system
queries independently confirmed no owned device assignment, active job or client
process. Both mathematical audits ran only after release. The remote watchdog
persisted job identity before readiness waits and enforced a 600-second runtime
deadline, bounded host memory/storage and independent release checks.

The host runtime stage took 272.269 seconds. The instrumented capture operation
sequence took 25.484 seconds and includes uploads, readbacks and host work. These
are not pure kernel timing, token latency, throughput or an official charge.

Earlier reduced simulation completed two useful epochs but timed out at 420
seconds during the third; its failed status remains preserved. The full compile
succeeded, then an overly small host archive inventory limit failed. A separate
bounded check reused that exact artifact and passed; compilation was not repeated.
The first separate checker could not see its input through the SDK container bind;
the corrected read-only bind passed. These failures are not relabeled successes.

Next work is full-checkpoint acquisition, original-input complete full-attention
and recurrent layers, then the three connected stages (0–19, 20–43, 44–63), state
save/restore, final norm, full-vocabulary logits and tokens. Stage weights may be
reloaded while one physical system is reused sequentially.

[Device and capture source](../examples/hw01) · [Independent acceptance](../evidence/hw01-full-mlp.json)
