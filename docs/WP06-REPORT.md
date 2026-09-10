# WP06 — Full128x128 persistent DeltaNet head

Candidate001 passed four token updates in one SDK2.10.1 runtime across two
simulated WSE-3 PEs. Every token passed independent prediction, delta,
full16384-element state and output checks, with frozen propagated FP32 error
bounds. These are synthetic full-head recurrence results, not original-weight
model-layer execution or physical-wafer performance.

| Token / generation | Prediction max error | Delta max error | State max error | Output max error |
|---|---:|---:|---:|---:|
| 1 / 1, nonzero initial state | 4.1564e-10 | 1.2628e-8 | 2.6427e-9 | 4.4415e-10 |
| 2 / 1, persistent state, beta=0 | 1.6431e-9 | 0 | 1.3213e-9 | 2.5559e-10 |
| 1 / 2, reset then changed nonzero input | 0 | 0 | 1.7462e-9 | 5.4482e-10 |
| 1 / 3, reset then zero Q/K/V | 0 | 0 | 0 | 0 |

The largest error-to-bound ratio was0.36494. The last token additionally required
exact numerical zero throughout. The second token has beta=0 and decay=0.5, so
its nonzero state must come from the previous token; delta is exactly zero. The
first reset uses decay=1 and changed nonzero inputs, preventing forgetting from
masking reset failures. CPU adversarial tests reject stale state, forgotten
state and output computed before state update.

Both PEs passed generation/token/commit counters and exact event ordering.
All packet identities, immutable inputs, guards, exported handles and output
transport bits passed. Global delta was read from the peer's retained phase2
receive buffer because the root send body is later reused for local output.
Complete actual/expected/bound arrays and initialization observations remain in
the development receipt; public [evidence](../evidence/wp06.json) contains compact
stage metrics and identities. The [design](WP06-DESIGN.md) records arithmetic,
buffer ownership and the limits of successful command completion.

## Resources and reproducibility

The first candidate compiled and passed; no second candidate was needed.
Compilation took4.113955 seconds and simulation81.677549 seconds, both below
their300-second deadlines. Observed live-cgroup maxima were419,053,568 and
209,616,896 bytes respectively; these are sampled observations, not guaranteed
lifetime peaks. The task used no swap and one simulator thread. The complete
remote run occupied8,581,120 allocated bytes at post-run verification.

Both actual SRAM checks passed before simulation:

| PE | Highest application section end | Stack allowance | Total / ceiling | Margin |
|---|---:|---:|---:|---:|
| 0 | 42,256 | 4,096 | 46,352 / 49,152 | 2,800 |
| 1 | 41,936 | 4,096 | 46,032 / 49,152 | 3,120 |

All22 frozen source/input files and15 compiled files matched after execution.
Normal SDK stop and process exit completed. Both owned units were absent and
inactive, with PID0 and empty cgroups. Forty-one lightweight host tests passed.

The separate CPU reference passed8 state/output comparisons in2.079814 seconds,
with observed207,511,552 bytes under a2 GiB/60-second CPU-only limit. Its eight
frozen files remained unchanged and the owned unit was absent afterward. It
used an existing read-only interpreter and the pinned official source hash;
there was no environment installation or checkpoint download. Requested decay
matched the actual exp(log(decay)) result for all four fixtures.

Reproduction uses an existing CPU PyTorch interpreter, previously fetched
pinned WP04 source directory, and an installed SDK image:

```sh
python3 tools/prepare_wp06_reference.py --run /path/to/cache/wp06-reference-new \
  --source-dir /path/to/pinned-wp04-sources --python /path/to/existing-torch-python
python3 /path/to/cache/wp06-reference-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp06-reference-new \
  --spec /path/to/cache/wp06-reference-new/steps.json --profile cpu-reference
python3 tools/prepare_wp06.py --run /path/to/cache/wp06-new \
  --reference /path/to/cache/wp06-reference-new --image /path/to/installed-sdk.sif
python3 /path/to/cache/wp06-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp06-new \
  --spec /path/to/cache/wp06-new/steps.json
```

The next proposed operator is128-element direct-gain gated RMS with SiLU(z),
including the early input-dtype cast. It will then compose with the recurrent
head through an explicit precision/ownership boundary. Q/K normalization,
convolution, gate generation, all48 heads and full-model inference remain
separate work.
