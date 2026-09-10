# WP02 — Two-PE projection, fabric join and RMS normalization

September 10, 2026. Four calls passed in one SDK runtime on two adjacent simulated
WSE-3 PEs. The original WP01 128×112 tile was reused with no new network download.
Columns 0:56 and 56:112 form two local GEMVs. PE1 sends its 128 FP32 partials to
PE0; PE0 performs SUM and 128-element RMS normalization with FP32 epsilon 1e-6
and unit gain. This is a reduced operator chain, not Qwen hidden5120 normalization,
a complete layer, or physical multi-wafer inference.

## Independent stage checks

Both local partials, the transferred raw bits, sum, normalized output, squared sum,
epsilon-adjusted denominator, sqrt(actual denominator) and reciprocal(actual root)
were checked independently against the [frozen policy](WP02-DESIGN.md). No final
output pass can substitute for an intermediate or approximation gate.

| Call | SUM max error | RMS max error | Largest stage error / bound | Mode |
|---|---:|---:|---:|---|
| bounded | 1.862645149e-09 | 1.459352896e-07 | 0.002518687625 | 0 |
| changed | 1.862645149e-09 | 2.343628625e-07 | 0.003789524047 | 1 |
| last_column | 0 | 1.240451009e-07 | 0.004635396418 | 0 |
| zero_after_nonzero | 0 | 0 | 0 | 1 |

All weight/input/output guards, immutable weight bits, stable handles and call
counters passed. Received FP32 words exactly matched sender partials. The zero
case additionally produced exact numerical zeros in both partials, sum and RMS.
Maximum observed sqrt relative error was 4.8759830e-8, and reciprocal error
3.4146283e-8; each independent frozen budget was 2^-20.

## Actual device joins

Mode0 root events were local=1, arm=2, receive-complete=3, sum=4, RMS=5, unblock=6.
Mode1 root events were arm=1, receive-complete=2, local=3, sum=4, RMS=5, unblock=6.
Each mode was exercised twice on device. Every root call recorded exactly one
receive, commit and unblock. Sender events were local=1, issue=2, send-complete=3,
unblock=4. Sender completion is not claimed to establish receiver completion.
Negative host fixtures reject missing/duplicate commits, premature unblock, wrong
mode order and reserved-resource collisions. These complement the actual device
events rather than substitute for them.

## Execution and resources

Candidate 001 failed at type checking because the variable `color` shadowed the
CSL builtin type. Candidate 002 renamed it `partial_color`; no kernel, route,
arithmetic, input, numerical bound or task/DSR policy changed. No simulation ran
for the first candidate, and neither candidate was retried unchanged.

Successful compilation took 4.087212 s and simulation 43.832493 s. Both exited zero;
SDK stop returned normally. Observed live-cgroup maxima were 428,699,648 and
187,138,048 bytes; these are not guaranteed complete lifecycle peaks. Both steps
used MemoryMax 20 GiB, MemorySwapMax 0, one simulator thread, a 300-second deadline, and
the inherited RAM/disk/cache/lock policy. Run file blocks totaled about 740 KiB.

The two application ELFs contained 23,990 / 23,938 bytes of allocated sections;
ordinary low-address sections accounted for 23,598 / 23,546 bytes. Higher sections
contain device configuration. These inventories exclude unrepresented runtime
stack demand and are not dynamic SRAM peak measurements. Fifteen compiled files
were unchanged from before simulation, and all 22 frozen source/input files were
unchanged. Both services were absent/inactive with MainPID 0 and empty cgroup after
cleanup. Full arrays and failure evidence remain in development; public results
contain metrics, source hashes and device events, without model payload.

## Reproduction

Prepare a synthetic fixture without any download:

```sh
python3 tools/prepare_wp02.py --run /path/to/project-cache/wp02-new \
  --image /path/to/already-installed-sdk-2.10.1.sif
python3 /path/to/project-cache/wp02-new/tools/guarded_run.py \
  --cache /path/to/project-cache --work /path/to/project-cache/wp02-new \
  --spec /path/to/project-cache/wp02-new/steps.json
```

To reuse the existing original WP01 tile, add `--tile /path/to/existing-tile` to
preparation. This workflow requires the already qualified SDK/Singularity/systemd
environment. Inspect the frozen input, ledger, policy and execution manifests
before running. Synthetic preparation was unit-tested but not separately run on
SDK during this milestone.

See [machine-readable evidence](../evidence/wp02.json) and the
[resource/ownership and numerical derivation](WP02-DESIGN.md).
