# D256 attention / cache8

One handwritten CSL PE implements cache append, valid-prefix attention and raw
sigmoid output gating. Q/K inputs are already normalized and rotary transformed
synthetic BF16 values. This example does not implement QK RMS/RoPE, projection,
all-head GQA or a complete attention layer.

See `docs/WP10-DESIGN.md` and `docs/WP10-REPORT.md` for boundaries and measurements.
The CPU reference extracts pinned eager source arithmetic from a preexisting
`modeling_qwen3_5.py` whose SHA256 is checked by the preparer. It needs an existing
CPU-capable torch environment. No model weights or new environment are required.

From the repository root, use fresh run directories on a qualified Linux SDK host:

```sh
python3 tools/prepare_wp10_reference.py \
  --run "$CPU_RUN" --source-dir "$PINNED_SOURCE_DIR" --python "$TORCH_PYTHON"
python3 tools/guarded_run.py --profile cpu-reference \
  --cache "$CACHE_ROOT" --work "$CPU_RUN" --spec "$CPU_RUN/steps.json"
python3 tools/prepare_wp10.py \
  --run "$SDK_RUN" --image "$EXISTING_SDK_IMAGE" --reference "$CPU_RUN"
python3 tools/guarded_run.py --profile sdk \
  --cache "$CACHE_ROOT" --work "$SDK_RUN" --spec "$SDK_RUN/steps.json"
```

Set the named variables to absolute paths. The SDK wrapper reuses an existing
qualified SDK2.10.1 image and Singularity installation. Its prepare step verifies
the CPU artifacts before freezing sources. Compilation precedes actual SRAM
admission; the simulator only runs after the48KiB ceiling including4096stack
allowance passes. Inspect `supervisor.json`, `result.json`, `sram-admission.json`,
`observations.json`, `initializations.json`, `overflows.json`, `lifecycle.json` and
`transfer-journal.jsonl`. Completion requires all gates and normal runtime stop.

Default limits are CPU2GiB/60s, SDK20GiB/zero swap, compile300s and simulation180s.
Run one heavy job at a time, retain8GiB available RAM and32GiB disk reserve, and
keep the bounded cache within20GiB. Never rerun an existing frozen directory.
