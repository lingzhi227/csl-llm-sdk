# WP01 — Original BF16 GEMV and warm calls

September 10, 2026. This milestone computes a **128×112 slice** of the original
Qwen3.8-27B layer-0 up-projection on a single WSE-3 simulated PE. It is not the
full 5,120-input projection, a neural layer, or model generation.

## Target and data acquisition

Pinned checkpoint `Qwen/Qwen3.8-27B` at
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, tensor
`model.language_model.layers.0.mlp.up_proj.weight`, shape `[17408,5120]`, BF16.
Rows 0–127 and columns 0–111 were retrieved from the first safetensors shard.
The downloader made 130 strict HTTP range requests: 8 header-length bytes,
45,392 header bytes and 128 row fragments of 224 bytes. Total response payload
was **74,072 bytes**; original weight payload was 28,672 bytes. Every response
was 206 with the exact requested Content-Range. HTTP 200 and mismatched ranges
are rejected before reading the body. A process payload cap of 8 MiB applies,
with no retry or fallback to a whole-shard download.

The download finished in 29.4 seconds within the same resource policy as SDK
work. Model payload remains outside Git. Header/tensor/range identities and both
row/column-major SHA-256 hashes are in [compact evidence](../evidence/wp01.json).
A row-major → column-major → row-major bit round trip was exact. Full-shard
integrity was not checked because the full shard was not downloaded.

## Kernel and checks

The [first-party C01 kernel](KERNEL-PROVENANCE.md) is byte-identical to the reviewed
source. BF16 high-halfword expansion preserves value bits; FP32 FMA accumulates
112 columns into 128 outputs. Synchronous DSR bank 3 handles expansion; bank 4
handles accumulation. Each call reconstructs descriptors and zeroes expansion
and result storage. Guarded static buffers keep exported storage stable.

One programmed runtime executed: bounded input, changed input, a one-hot in the
last valid input column, then all-zero input after nonzero work. Every call
checked all 128 outputs, full original weight readback, input/output guards,
call counter and unchanged exported IDs. The last selected column contains no
positive or negative zero, so the exact one-hot bit check has no signed-zero
ambiguity for this tile. The all-zero case allows either sign of numerical zero.

The independent CPU oracle uses `math.fsum` over exactly represented BF16×dyadic
input products in FP64. Policy was frozen before either candidate ran:
`abs(error) <= gamma112 * sum(abs(weight*input)) + 112*2^-126`,
where `gamma112 = (112*2^-24)/(1-112*2^-24)`. This absolute/scaled bound remains
meaningful under cancellation. One-hot outputs additionally require exact FP32
bits, and zero-after-nonzero requires exact numerical zero. The policy was not
relaxed after observing device output.

| Case | Maximum absolute error | Maximum error / bound | State checks |
|---|---:|---:|---|
| Bounded | 1.8626451492e-9 | 0.001262554 | All passed |
| Changed | 1.8626451492e-9 | 0.000996852 | All passed |
| Last-column one-hot | 0 | 0, exact bits | All passed |
| Zero after nonzero | 0 | 0, exact zero | All passed |

## Execution evidence and limits

Candidate 001 failed during CSL type checking: a pointer to one guarded element
(`*u16`) was passed to an API requiring a many-element pointer (`[*]u16`). No
simulation ran. Candidate 002 adds explicit typed pointer conversions in the
wrapper; kernel, weights, oracle and policy remain unchanged. This was the second
and final authorized candidate, with no unchanged retry.

Compilation took 4.082610 s and simulation 57.968313 s, including container/service
overhead. Both exited zero; SDK stop returned normally. Observed live-cgroup
memory maxima were 461,520,896 and 188,473,344 bytes respectively. Final lifetime
peaks were not retained after cgroup removal. Actual MemoryMax was 20 GiB and
MemorySwapMax zero, with the WP00 cache/RAM/disk reserves and one heavy task.
The successful run used approximately 484 KiB of allocated file blocks at audit.

ELF allocated sections total 35,060 bytes: 34,672 bytes in the ordinary low-address
code/data/task-table sections and 388 bytes in higher device configuration
sections. This is static section accounting, **not** a measured runtime stack or
complete dynamic SRAM peak. The compiler successfully linked the one-PE image.
All 11 compiled-output files were hashed before simulation and unchanged after
execution; all 19 frozen source/input files also remained unchanged. After cleanup
both service units were absent/inactive, with MainPID=0 and empty ControlGroup.

The original tile and full failure/results remain in development evidence. Public
evidence contains identities and summary metrics, not original model values or
the one-hot output (which would reveal selected weights).

## Reproduction

Existing SDK 2.10.1/Singularity and systemd user memory delegation are required.
To generate a **synthetic** standalone fixture without network access:

```sh
python3 tools/prepare_wp01.py --run /path/to/project-cache/wp01-new \
  --image /path/to/already-installed-sdk-2.10.1.sif
python3 /path/to/project-cache/wp01-new/tools/guarded_run.py \
  --cache /path/to/project-cache --work /path/to/project-cache/wp01-new \
  --spec /path/to/project-cache/wp01-new/steps.json
```

For original data, run `tools/download_wp01_tile.py --output /path/to/new-tile`
under the resource supervisor (one download step, at most 300 seconds), then pass
`--tile /path/to/new-tile` to preparation. The downloader never overwrites an
existing attempt. Review source-manifest.json, numerical-policy.json and steps.json
before launch. A new reproduction is a new result; the supplied synthetic fixture
was prepared/tested on CPU but was not separately run in SDK during this milestone.
