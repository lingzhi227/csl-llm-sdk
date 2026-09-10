# WP08 — Selected-head causal preprocessing

Repaired candidate002 passed eight tokens on one simulated WSE-3 PE: six
consecutive updates crossing the width4 history boundary, then a request reset
followed by zero and changed projected input. All384 convolution channels,
128-dimensional Q/K/V outputs, scalar gates, full1536-element history, exact
BF16 conversions and state/guard checks passed. This is synthetic standalone
preprocessing, not projection, all10240 channels, recurrent composition or a layer.

The device computes causal convolution, SiLU, FP32 L2 normalization using a sum
and epsilon, Q scaling, BF16 beta and FP32 g/decay. History stores original
projected BF16 input. Asymmetric temporal weights, isolated impulses and complete
history readbacks expose reversed orientation, look-ahead and stale reset state.

## Numerical results

All convolution and activation BF16 conversions matched the independent RNE
oracle applied to actual preceding FP32 bits. Beta conversion and FP32 promotion
also passed. Sixteen signed conversion probes passed on every token. Separate
mathematical-oracle convolution/activation BF16 mismatch counts were zero in
all eight cases, without substituting for the exact conversion gates.

| Stage | Maximum fraction of its frozen error allowance |
|---|---:|
| Convolution exponential | 0.06948 |
| Beta exponential | 0.02608 |
| exp(A_log) | 0.03345 |
| Softplus exponential | 0.02608 |
| exp(actual g) | 0.05558 |
| log1p series | 0.00794 |
| Softplus | 0.01195 |
| Q normalization | 0.02533 |
| K normalization | 0.01958 |

Every actual exponential argument stayed in the declared[-24,1] domain. Scalar
gate inputs recorded by the device exactly matched a, b, A_log and dt_bias.
The eight BF16-promoted beta results were0.000335693359375,0.376953125,0.5,
0.62109375,1.0,0.26953125,0.5 and0.73046875.

The independent CPU reference passed all eight cases and exact history checks;
cached updates matched full causal-prefix replay at every step. It used pinned
official convolution/L2 functions and unchanged beta/g assignment expressions,
with BF16/FP32 promotion checks. Reference002 took2.088327 seconds with observed
207,777,792 bytes;10 frozen files matched afterward and the unit was absent.
Reference001 had also passed before the candidate implementation-policy update.
No environment installation or weight download was needed.

## Failed candidate and repair

Candidate001 failed numerical acceptance despite correct history, guards,
parameter readbacks and conversions. Its beta result matched sigmoid(x[383])
instead of sigmoid(b), including the early BF16 beta store. Several installed
exp evaluations also exceeded the frozen2^-20 gate, reaching1.09384 times its
allowance. The run completed normal SDK shutdown and then exited with a failed
result. It is retained as a failure, not counted as qualification.

Candidate002 keeps every fixture and numerical threshold. Vector/history work
enters a pending-gates phase; a separate device finalize command reads fixed
global parameters into observable FP32 slots, computes gates and commits once.
The host verifies parameters between commands and checks both phase records.
The repaired exponential uses split-ln2 range reduction, a degree8 polynomial
and exact normal power-of-two scaling. See the
[causal hypotheses and error accounting](WP08-CANDIDATE-REPAIR.md). The observed
repair does not establish the original compiler/input-source failure's root cause.

## Resource and artifact receipts

| Candidate | Compile seconds | Simulation seconds | Observed compile / sim bytes | Outcome |
|---|---:|---:|---:|---|
| 001 | 4.105494 | 67.504771 | 419,995,648 / 186,986,496 | Numerical failure, clean SDK stop |
| 002 | 4.111108 | 68.590410 | 420,499,456 / 186,511,360 | All checks passed, normal exit |

Observed memory values are live-cgroup samples, not guaranteed lifetime peaks.
Both candidates used the same300-second compile/simulation deadlines,20 GiB
RAM ceiling, zero swap, one heavy task,8 GiB available-RAM reserve and32 GiB
disk reserve. Candidate002's application sections end at24176; adding4096 stack
bytes gives28272 within49152, leaving20880 bytes. Actual SRAM admission passed
before simulation. The complete accepted run occupied1,298,432 allocated bytes.

All24 frozen source/input files and11 compiled files matched after candidate002.
Candidate001's23/11 identities also remained unchanged. Both owned job groups
were quiescent with PID0 and empty cgroups before subsequent work. The failed
unit alone was cleared after preserving diagnostics. Forty-nine host tests pass,
including exponential error accounting, a1025-point grid and reduction-boundary
neighbors. Those CPU checks do not claim exhaustive device-domain coverage.
Both preparation tools reproduce the executed frozen files except launch paths.

Complete stage/history arrays remain in development receipts; public
[evidence](../evidence/wp08.json) retains sanitized metrics and identities.

## Reproduction and next work

```sh
python3 tools/prepare_wp08_reference.py --run /path/to/cache/wp08-reference-new \
  --source-dir /path/to/pinned-wp04-sources --python /path/to/existing-torch-python
python3 /path/to/cache/wp08-reference-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp08-reference-new \
  --spec /path/to/cache/wp08-reference-new/steps.json --profile cpu-reference
python3 tools/prepare_wp08.py --run /path/to/cache/wp08-new \
  --reference /path/to/cache/wp08-reference-new --image /path/to/installed-sdk.sif
python3 /path/to/cache/wp08-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp08-new \
  --spec /path/to/cache/wp08-new/steps.json
```

The next package should connect these device outputs to the existing two-PE
recurrence and gated-output consumer, preserving history, recurrent state,
precision boundaries and completion ownership. That integration, all-head
composition, original-weight layers and complete model inference remain open.
