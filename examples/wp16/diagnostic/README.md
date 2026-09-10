# Accepted original-weight partial MLP source

The accepted SDK2.10.1 simulator experiment connects four PEs:128 gate/up output
rows over5120 input columns, whole SiLU and BF16 product, then128 down rows over
only128 intermediate channels. It covers one dense input with normal shutdown.
It does not qualify full17408-channel MLP output, reset/reuse or whole-model inference.

Read the [milestone report](../../../docs/WP16-PARTIAL-MLP-REPORT.md) and
[evidence summary](../../../evidence/wp16-partial.json). The complete
[physical schedule](physical-sequence.json) and [interfaces](interfaces.json)
describe846 operations and24 exported arrays. Source and publication hashes are
listed in the [provenance map](../../../evidence/wp16-partial-source-map.json).

The device code, numerical kernels, checker, driver and observation schedule match
the accepted frozen source. The public `core/qwen38/wp16_sdk_admission.py` is a
**separate, unexecuted configuration template**: private working/SDK paths and
machine stat identity were replaced by explicit placeholders. Its defaults refuse
execution. This path-only portability work is not a new qualified SDK run.

Host/source checks require Python and NumPy and can be run from the repository root:

```sh
python3 -m unittest discover -s tests -p 'test_wp16_*.py' -v
```

Reproduction on another host requires a separately installed compatible SDK,
the exact original selected weight slices and input-dependent prepared source
artifacts, a fresh candidate workspace, a checked SDK image identity, and an exact
source/profile admission appropriate to that host. The public template must be
configured before the source is frozen; accepted private admission files cannot
authorize a differently configured workspace. Stage the scripts and `core/` in
the candidate root with the named `prepared/` and `prefix-source/` artifacts before
using the guard. Running these scripts directly in this repository folder is not
the admitted experiment. No source guard creates execution authorization itself.

Weights, original mixed parameter archives, compiled ELFs, SDK distributions and
private machine/owner inventories are excluded. Published typed JSON preserves
473 nonparameter device arrays,28 source arrays, the original hidden inputs and
their exact dtype/shape/payload hashes. Nine full weight readbacks are represented
by metadata/hashes only. The journal retains its original bytes.

Each admitted003 stage usedCPU0, one worker,1GiB hard memory,Swap0,pids128 and a
300-second hard deadline. Simulation took285.327 seconds, exceeding the240-second
estimate without exceeding the hard limit. API durations include queued simulator
work. There is no hardware-throughput claim. ELF section-end-plus4096 values are
static SRAM gates rather than measured dynamic-stack peaks.

Middle projection tiles1–44 have no complete bitwise weight readback. Their host
upload hashes, explicit device pre/post guards and independent source-prefix
checks are separate evidence. Every completed observation group is validated and
saved before overwrite. All observed transport joins were command-before-receipt;
640 exact cast checks use observed FP32 values rather than full-model parity.
