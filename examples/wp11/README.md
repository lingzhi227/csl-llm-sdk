# Q/K RMS256 and partial RoPE64

This standalone handwritten CSL stage consumes original BF16 Q/K and offset
gains, then computes normalization and bounded text-position rotation on device.
Static FP32 inverse frequencies are extracted from the pinned official source;
runtime positions, angle multiplication and sin/cos execute in CSL. Supported
positions are0–7, not arbitrary long contexts. The first64dimensions rotate and
the remaining192 pass through exactly from BF16 normalized values.

See `docs/WP11-DESIGN.md` for numerical/transfer contracts and
`docs/WP11-REPORT.md` for the actual validation status. Use existing qualified
CPU torch and SDK2.10.1 environments; no model payload or new installation is needed.
From the repository root, choose absolute paths and fresh run directories:

```sh
python3 tools/prepare_wp11_reference.py \
  --run "$CPU_RUN" --source-dir "$PINNED_SOURCE_DIR" --python "$TORCH_PYTHON"
python3 tools/guarded_run.py --profile cpu-reference \
  --cache "$CACHE_ROOT" --work "$CPU_RUN" --spec "$CPU_RUN/steps.json"
python3 tools/prepare_wp11.py \
  --run "$SDK_RUN" --image "$EXISTING_SDK_IMAGE" --reference "$CPU_RUN"
python3 tools/guarded_run.py --profile sdk \
  --cache "$CACHE_ROOT" --work "$SDK_RUN" --spec "$SDK_RUN/steps.json"
```

Do not overlap heavy runs or reuse frozen directories. CPU limits2GiB/60s; SDK
20GiB/zero swap, compile300s/simulation180s,8GiB available RAMreserve,20GiB cache
and32GiB diskreserve. Actual SRAM plus4KiBstack must fit48KiB before simulation.
Inspect result, supervisor, source/compiled hashes, SRAM admission, observations,
initializations, lifecycle and the complete per-transfer journal. A normal stop is
required in addition to all numerical and state gates.
