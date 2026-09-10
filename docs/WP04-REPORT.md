# WP04 — Structural audit and bounded CPU reference foundation

The target's 851 text-backbone/head tensor names, shapes, BF16 dtypes and shard
assignments match the pinned candidate's declared structure. This checks metadata,
not all tensor contents or whole-model execution. [The semantic contract](WP04-SEMANTICS.md)
records equations, casting boundaries, configuration fields, source locations,
state shapes and the resolved distinction between DeltaNet swish gating and
full-attention sigmoid gating.

## What actually executed

The CPU harness extracts selected definitions from the exact saved official
`modeling_qwen3_5.py` using Python AST. It removes decorators from an in-memory
copy, preserving function/method bodies, and provides torch/nn/functional/SiLU
dependencies explicitly. Hub/kernel-dispatch decorators are removed to run the
Python fallback deterministically without accelerator dispatch or downloads.
Consequently this is **extracted execution of pinned official function bodies**,
not full Transformers runtime, loader or model verification.

The chunked recurrence also uses the exact `is_torchdynamo_exporting` helper from
the same commit's `utils/import_utils.py`, SHA-256
`e147f99cf68352e3260e9780930501c3becc0c40a5ff04150ff641b724161732`.
Its own exception handling selects the eager path on this older torch runtime;
the harness does not replace the helper with a constant. All source hashes and
definition line spans are preserved in the result. No source math was rewritten
to make a test pass. Operator substitutions, compilation/export and hub kernels
are outside the test scope.

Twenty-seven checks passed in the second frozen CPU run:

| Fixture | Dimensions and adversarial case | Maximum observed absolute error |
|---|---|---:|
| Ordinary RMS | 5120; offset gain, zero input, zero/-1 weights | 1.8366e-7 vs FP64; selected BF16 output exact |
| Gated RMS | 128; direct gain, negative/zero gates | 3.7109e-7 vs FP64 |
| Recurrent DeltaNet | one full128×128 head, two tokens, nonzero initial state | output1.1299e-9; state7.8513e-9 vs FP64 |
| Cached decode split | two one-token calls vs two-token recurrent call | exact output/state |
| Chunked prefill | same two tokens vs recurrent fallback | output3.0268e-9; state7.4506e-9 |
| Zero recurrent input/state | reset-equivalent fresh state | exact zero |
| Causal convolution | three channels, width4, six tokens; prefill vs cached steps | exact |
| Eager attention | 24 Q/4 KV heads, head256, two causal tokens | 3.4852e-8 vs FP64 |
| Partial RoPE | head256, rotary64; position0 identity, position1 pair, untouched192 tail | pair1.3402e-8; identity/tail exact |
| Q/gate packing | distinguish per-head256+256 from global concatenation | exact index assertions |
| Gate distinction | raw z=-2,0,2 and unit branch input | sigmoid/swish demonstrably unequal |

Nonexact comparisons use frozen `atol=2e-6, rtol=2e-6`; exact cases use zero
tolerance. These thresholds apply to the small source-equation fixtures, not
future CSL approximation budgets. No original weights, full model or GPU were
loaded. Padding-mask construction and full cache-object integration were not
executed; the harness supplies its explicit two-token causal mask.

## Run receipts and resource limits

CPU run001 failed after2.0645 seconds because the extraction harness had omitted
the chunk function's external export-detection helper. That failure is preserved.
Run002 added the pinned helper body and passed in2.0558 seconds. Both used a
separate `cpu-reference` supervisor profile: MemoryMax2 GiB, MemorySwapMax0,
RuntimeMax60 seconds, CPUQuota100%, single-thread torch/BLAS, no CUDA visibility,
and the existing project lock/RAM/disk/cache admission reserves. Observed live
memory maxima were206,733,312 and206,217,216 bytes, not guaranteed lifetime peaks.
All19 frozen files in run002 remained unchanged remotely:16 source/input files
and3 incidental CPython3.13 cache files inherited during initial preparation.
Only the16 text files were copied locally; the final reusable preparer excludes
bytecode caches. Its unit was absent,
inactive, MainPID0, with an empty cgroup after normal exit.

The existing workstation learning environment was read without modification:
Python3.11.14, torch runtime2.4.0+cu121, NumPy1.25.0. CUDA was hidden and all tensors
were CPU tensors. No package installation, SDK launch, GPU job or full checkpoint
download occurred. New primary-source payload is tracked separately from reused
configuration/source snapshots:559,639 new bytes, below the12 MiB package ceiling.

## Host refresh and proposed reference environment

Read-only host inspection found `/usr/local/bin/python3`3.13.11, package metadata
torch2.10.0+cpu, Transformers5.2.0, NumPy2.2.3, safetensors0.7.0,
tokenizers0.22.2 and accelerate1.12.0. The RTX4070 reports12,282 MiB total,1 MiB
used, driver595.84. GPU visibility does not make that CPU-only torch a CUDA
reference environment. The user systemd bus was unavailable in this SSH context.
Available RAM was approximately59.9 GiB and root free space936.7 GB at inspection;
neither is a future reservation. Workstation had about29.2 GiB available RAM and
93.6 GB SSD free. These observations justify reusing the existing bounded CPU
path for now.

For later full-package CPU integration, propose a new project-owned host venv
under the development reference directory, based on the observed Python3.13.11,
with CPU torch2.10.0+cpu and the exact audited Transformers commit. Resolve wheel
metadata and produce an exact dependency/hash lock before installation; reject
the plan if the pinned source does not support that interpreter/dependency set.
Use an initial1 GiB aggregate wheel/source-download ceiling,4 GiB environment-disk
ceiling, and2 GiB/60-second CPU microfixture limit. No original model allocation
is needed to instantiate tiny components. User-systemd enforcement must first be
restored or another controller-reviewed process container provided on host;
otherwise continue on workstation. This is a proposed package, not an install.

A CUDA environment is a separate later proposal requiring compatible wheel/driver
verification, its own measured dependency budget and an8 GiB application VRAM
ceiling with at least2 GiB reserve. This report authorizes no CUDA installation,
multi-machine execution, shared-environment mutation or full checkpoint load.

## Next CSL package proposal

Implement the audited **5120-element zero-centered learned-gain RMS** on one PE.
Use FP32 input staging restricted initially to exactly decoded finite BF16 values,
BF16 gain bits, FP32 reduction and multiplication by `(1+w)`, plus explicit
device-side BF16 round-to-nearest-even output. The FP32 pre-cast values remain
observable so reduction/reciprocal error and BF16 rounding can be checked
separately. Returning FP32 alone must be labeled as a partial operator.

An in-place FP32 staging/output array costs20 KiB, BF16 gain10 KiB and BF16 output
10 KiB, plus guards, scalar reduction scratch, code and runtime metadata. Confirm
the compiler's actual section inventory before simulation; the40 KiB buffer
estimate is not a total SRAM peak. No cross-PE route is required. Use one runtime
with four changed calls: zero, unit-gain varying input, signed learned gain
including w=-1, and a BF16 rounding-boundary fixture. Validate sum of squares,
epsilon denominator, reciprocal-root approximation, pre-cast vector, output bits,
guards, generation/reset and normal stop.

Freeze independent stage error bounds before execution. Include adversarial
cases that distinguish `(1+w)` from `w`, gain-before-cast from cast-before-gain,
and correct BF16 ties-to-even from truncation. Obtain any original norm-gain
slice only in a later explicitly budgeted range request. Limit the first SDK
candidate to one compile≤300 seconds and simulator≤180 seconds under existing
memory/disk policy, with a measured follow-up decision. This operator proceeds
from resolved RMS semantics without requiring full-model integration first.


## Reproduce the source-body fixtures

```sh
python3 tools/fetch_wp04_sources.py --destination /path/to/reference-sources
python3 tools/prepare_wp04.py --run /path/to/project-cache/wp04-new \
  --source-dir /path/to/reference-sources --python /path/to/prequalified-venv/bin/python
python3 /path/to/project-cache/wp04-new/tools/guarded_run.py \
  --cache /path/to/project-cache --work /path/to/project-cache/wp04-new \
  --spec /path/to/project-cache/wp04-new/steps.json --profile cpu-reference
```

The fetch helper retrieves two fixed-commit Python files with30-second request
limits,512 KiB per-file limits and required SHA-256 matches. Their upstream Apache
license notices remain in the fetched files; they are not republished as our MIT
code. Preparation performs no installation or execution. Use the qualified CPU
environment described above. The preparer and reference-profile mismatch rejection
were checked locally; the original successful frozen run predates the preparer's
explicit required-profile field but its receipt confirms the same2 GiB/60-second
profile was enforced. The new field prevents accidentally using the larger SDK
profile for this CPU bundle.
