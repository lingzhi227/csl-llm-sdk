# Original selected-head projections

Accepted coverage is layer3/head0 Q256/rawgate256/K256/V256, all5120 input columns,
four common synthetic BF16 inputs, **eight separate single-PE four-call runtimes**.
It does not produce all1024 outputs in one runtime or connect them to attention.
See the [design](../../docs/WP13-DESIGN.md), [report](../../docs/WP13-REPORT.md)
and [complete evidence index](../../evidence/wp13.json).

Use an existing Linux SDK2.10.1 installation and CPU Python3.11 with torch2.4.0 and
numpy1.25.0. No dependency installation is performed by these tools. Obtain the
pinned [official source file](https://github.com/huggingface/transformers/blob/4815a0a6a064214f2d8208c094464a5a6b76ca8d/src/transformers/models/qwen3_5/modeling_qwen3_5.py)
as `modeling_qwen3_5.py` in your source directory; preparation verifies its SHA256.
Choose fresh **absolute** run directories inside your bounded cache.

```sh
python3 tools/prepare_wp13_download.py --run "$DOWNLOAD_RUN"
python3 tools/guarded_run.py --profile cpu-reference \
  --cache "$CACHE_ROOT" --work "$DOWNLOAD_RUN" --spec "$DOWNLOAD_RUN/steps.json"

python3 tools/prepare_wp13_reference.py \
  --run "$CPU_RUN" --source-dir "$PINNED_SOURCE_DIR" \
  --weights "$DOWNLOAD_RUN/weights" --python "$TORCH_PYTHON"
python3 tools/guarded_run.py --profile cpu-reference \
  --cache "$CACHE_ROOT" --work "$CPU_RUN" --spec "$CPU_RUN/steps.json"

python3 tools/prepare_wp13.py --slab 0 \
  --run "$SDK_RUN" --image "$EXISTING_SDK_IMAGE" --reference "$CPU_RUN"
python3 tools/guarded_run.py --profile sdk \
  --cache "$CACHE_ROOT" --work "$SDK_RUN" --spec "$SDK_RUN/steps.json"
```

Review slab0's complete result, hashes, source intervals, physical sequence, journal,
SRAM and normal-stop/owned-unit cleanup before deciding to run slabs1..7 in fresh
directories. Run them serially, with all four cases retained per slab; a failure
stops the remaining queue and is preserved. The tool does not launch other slabs.
Omitting `--slab` prepares the older four-PE/one-case Q/rawgate timing diagnostic;
that diagnostic cannot substitute for any missing four-case batch.

Acquisition uses12 exact HTTP206 ranges totaling10,486,784bytes, with a16MiB cap,
1MiB chunks, no full-shard fallback and no automatic retry. CPU/acquisition jobs
use2GiB/Swap0/60seconds. SDK runs use20GiB/Swap0, compile300seconds and simulate
300seconds, one heavy job total,8GiB RAMreserve,20GiB cache cap and32GiB diskreserve.
The single-slab estimate is240seconds. Every actual ELF end+4096stack must fit49152;
each physical host buffer is at most57352bytes. Original weight copies remain local.

The portable acquisition/reference preparers reproduce the accepted source closure
and exact hidden bytes. Their path and dependency-description metadata may differ
from the frozen experiments; the numerical source, formulas and inputs do not.
Raw downloaded weights, complete weight-readback payloads and SDK products are
excluded from this repository. Public typed numerical arrays and journal hashes
retain the input/output/prefix/cast/state evidence needed for independent comparison.
