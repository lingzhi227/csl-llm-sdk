# WP03 — Full-width streamed contraction for 128 outputs

September 10, 2026. Four calls passed on one simulated WSE-3 PE in one SDK 2.10.1
runtime. An original BF16 up-projection slab covers rows 0:128 and all 5,120 input
columns. Device-owned FP32 accumulation persists across 45 tiles of width 112 and
an 80-column tail. The original matrix has 17,408 output rows, so this milestone
does not establish a full projection, layer, model or physical-wafer performance.

## Results

| Call | Final maximum absolute error | Maximum error / bound | Exact case |
|---|---:|---:|---|
| bounded | 1.0151416063e-7 | 3.0145269463e-5 | no |
| changed | 1.9371509552e-7 | 4.4895960861e-5 | no |
| one-hot at global 5119 | 0 | 0 | yes |
| zero after nonzero | 0 | 0 | yes |

All eight boundary checks at consumed columns 112 and 5040 passed independently.
Each call completed exactly 46 accumulations, consumed 5120 valid columns and
committed its own generation. Nonzero padding was excluded. Input/output guards,
fresh finalize-time weight-guard snapshots and stable exported handles passed.
Snapshot generation matched each call. Snapshot/commit markers record source
order and are not independent timestamps. The final call additionally read the
whole actual resident tile and verified its bits and endpoint guards. Historical
tiles were not each read back; this limits the device immutability claim.

The [design](WP03-DESIGN.md) describes the frozen forward-error contract, independent
FP64 reference, ownership and exact-case gates. [Evidence](../evidence/wp03.json)
contains aggregate errors, state, raw counter words and source hashes, with no
model payload or actual output arrays.

## Resources and timing

Compilation took 4.100940 seconds; simulation took 205.784953 seconds, leaving
94.215 seconds before the 300-second hard deadline. Both exited normally and SDK
stop returned. Observed live-cgroup maxima were 452,415,488 bytes for compilation
and 222,920,704 bytes for simulation. These are observed maxima rather than
guaranteed complete lifecycle peaks. Limits remained MemoryMax 20 GiB, no swap,
8 GiB available-memory reserve, 32 GiB disk reserve and one heavy job at a time.

Each call recorded 20,377 kernel cycles for every full tile and 14,585 for the
tail, totaling 931,550 kernel cycles. The simulator recorded 8,224,114 total cycles
and 204.030269 seconds of simulation. Kernel timestamps exclude host transfers;
simulator wall time is not hardware performance or an isolated bandwidth result.
Reducing repeated whole-weight readbacks saved observation work; it is not a
kernel speedup claim.

The application ELF contained 37,370 bytes of allocated sections, including 36,982
bytes at ordinary low addresses and 388 bytes of device configuration. This
inventory does not measure dynamic stack demand. All 12 compiled files and 72
frozen source/input files matched pre-run hashes after simulation. Both owned
services were absent/inactive with MainPID 0 and empty cgroups after cleanup.
The complete remote run occupied 3,280,896 allocated bytes at verification.

## Acquisition and candidate history

One bounded HTTP range fetched 1,310,720 weight bytes, using the already verified
shard header. The slab's first 112 columns matched WP01 bit-for-bit. No full shard,
model checkpoint or SDK image was downloaded. Slab SHA-256:
`9d4db091b2597ed6304d1ff001e89016caae17196b1d73e55c0d1d2ca86c64e0`.

Candidate 001 was a separately labeled 304-column diagnostic (112+112+80), with
four passing calls and 69.190303 seconds of simulation. It supported a frozen
250-second planning window for full-width work after reducing redundant readbacks.
Candidate 002 failed compilation in 3.090048 seconds because a global guard alias
used an unsupported compile-time pointer cast; no simulation ran. Its logs and
cleanup record were preserved. Controller-authorized candidate 003 replaced
that observation mechanism with a fresh finalize snapshot. Arithmetic, weights,
four full-width calls, numerical policy and hard resource limits were unchanged.

## Offline reproduction

Generate an explicitly synthetic 1.25 MiB fixture, then freeze a run:

```sh
python3 tools/make_wp03_fixture.py --destination /path/to/project-cache/synthetic-slab
python3 tools/prepare_wp03.py --run /path/to/project-cache/wp03-new \
  --image /path/to/already-installed-sdk-2.10.1.sif \
  --slab /path/to/project-cache/synthetic-slab --columns 5120
python3 /path/to/project-cache/wp03-new/tools/guarded_run.py \
  --cache /path/to/project-cache --work /path/to/project-cache/wp03-new \
  --spec /path/to/project-cache/wp03-new/steps.json
```

The existing qualified SDK/Singularity/systemd environment is required. Synthetic
preparation and tail packing were host-tested; this synthetic slab was not run
through SDK for this milestone. Its `slab-source.json` identifies it as synthetic;
`full5120_acceptance` describes the contraction test's dimension and never proves
original-model provenance. Thirty host tests pass, including deliberately lost
prior accumulation and incorrect global one-hot results rejected by the oracle.
