# WP05 — RMS5120 with offset gain and device BF16 output

Four calls passed in one SDK2.10.1 runtime on one simulated WSE-3 PE. The operator
normalizes5120 synthetic BF16-representable inputs in FP32, applies learned offset
gain `(1+w)` and performs actual device-side BF16 round-to-nearest-even. This is
a full-dimension ordinary RMS operator fixture, not a model layer, original-weight
validation or DeltaNet gated RMS.

| Call | Maximum pre-cast error vs FP64 oracle | Gain-before-cast adversary differences | Device RNE checks |
|---|---:|---:|---|
| zero | 0 | 0 | all5120 exact |
| varying input, unit gain | 1.0383e-7 | 0 | all5120 exact |
| signed nonuniform offset gain | 3.1323e-7 | 368 | all5120 exact |
| rounding-order fixture | 3.6747e-7 | 1692 | all5120 exact |

All squared sums were exact for these fixtures. Denominator and pre-cast errors
remained within frozen forward-error bounds. Maximum independently checked sqrt
relative error was6.6759443e-8 and inverse relative error1.6114541e-8, both below
their separate2^-20 gates. Gain application was also checked against the actual
device inverse, preventing upstream approximation errors from hiding a gain bug.

Every zero input or gain=-1 position produced exact numerical zero. Sixteen
separate signed midpoint/neighbor/exponent-carry/zero bit probes used the same
device conversion function and all passed in every call. All guards, generations,
call counts, conversion-probe counts and exported handles passed; the reduction
was freshly initialized on every invocation.

Exact BF16 conversion was checked against each actual FP32 pre-cast bit pattern.
The separate FP64-oracle BF16 mismatch count happened to be zero in all four
fixtures. That coincidence does not replace the conversion gate or establish
parity for arbitrary inputs near rounding midpoints. See the [design and bounds](WP05-DESIGN.md)
and [compact evidence](../evidence/wp05.json).

## In-place storage and SRAM

The5120 FP32 staging values are overwritten with pre-cast results. A separate
5120 BF16 buffer initially stores gain; each gain is consumed before its slot is
overwritten with BF16 output. No later operation reads overwritten slots as gain.
Every new call uploads the complete gain again. Device gain immutability is thus
explicitly not claimed; original host input/gain fixtures remain frozen and
hash-verified. Both complete result vectors are simultaneously observable.

The30 KiB payload plus compiled code/runtime sections ended at address36368.
Adding the4096-byte declared stack allowance gives40464 bytes, below the49152-byte
application SRAM ceiling by8688 bytes. Ordinary allocated sections total36364
bytes;388 bytes of device-configuration sections bring total allocated sections
to36752. High configuration addresses add no application SRAM. The stack allowance
is a conservative budget, not a measurement of dynamic stack peak.

## Candidate history and resource receipts

Candidate001 used an incorrect61440-byte SRAM admission ceiling. Its low section
end46624 plus4096 stack allowance exceeded the actual48 KiB budget by1568 bytes.
The controller correction arrived after simulation had started; the owned process
was immediately stopped, after35.687630 seconds of simulator supervision. This
candidate is preserved as controller-interrupted/noncompliant and is not accepted
as a pass. Compilation had taken4.101996 seconds. PID0/empty-cgroup cleanup was
verified before the approved repair.

Candidate002 replaced separate gain/output buffers with the declared shared
buffer and enforced49152 bytes including the same4096-byte allowance. It changed
no dimensions, arithmetic, numerical thresholds or fixture values. Tests reject
the original capacity error, one-byte overflow, a larger ceiling, a smaller stack
allowance and application text at high addresses mistaken for device configuration.

The repaired compile took4.110701 seconds and simulation57.104007 seconds, with
normal SDK stop and process exit. Observed live-cgroup maxima were462,970,880 bytes
and194,994,176 bytes; these are not guaranteed lifetime peaks. The unchanged host
limits were compile300 seconds, simulation180 seconds, MemoryMax20 GiB, no swap,
single heavy job,8 GiB available-memory reserve and32 GiB disk reserve. Both owned
services were absent/inactive with PID0 and empty cgroups after completion.

All18 frozen source/input files and11 compiled files matched after the run. The
complete remote run occupied2,879,488 allocated bytes at verification. Kernel
cycles were256223 for zero and256301 for each other call; these are simulator
kernel counters, not physical-wafer performance. Thirty-eight host tests pass.

## Reproduction and next operator

```sh
python3 tools/prepare_wp05.py --run /path/to/project-cache/wp05-new \
  --image /path/to/already-installed-sdk-2.10.1.sif
python3 /path/to/project-cache/wp05-new/tools/guarded_run.py \
  --cache /path/to/project-cache --work /path/to/project-cache/wp05-new \
  --spec /path/to/project-cache/wp05-new/steps.json
```

Preparation needs no network or checkpoint. It freezes all synthetic inputs,
conversion probes, numerical policy, SRAM allowance and execution deadlines.
Compilation must pass the48 KiB SRAM admission gate before simulation. Keep the
original qualified SDK image installed separately; it is not distributed here.

The next proposed operator is DeltaNet's128-element direct-gain gated RMS with
SiLU(z), including its earlier dtype cast and independent activation-error gates.
It can reuse the validated ordinary-RMS infrastructure and BF16 conversion, but
must not reuse `(1+w)` semantics. After that, connect the full128×128 recurrent
head to gated normalization with explicit state ownership and reset checks.
