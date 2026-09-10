# Original Q/K through persistent attention

A single CSL PE runs ordinary Q/K RMS256, bounded device partial RoPE64, copies
the resulting BF16 operands on device, and runs D256 attention with an eight-slot
persistent KV cache and raw sigmoid gating. Original Q/K/V/gate inputs remain
separate and immutable. This is one synthetic head for positions0–7, not a complete
layer/model or long-context/hardware result.

The source profile propagates preprocessing/current-Q/cached-K uncertainty through
the final output. It does not widen the accepted standalone attention policy or
substitute actual device values for the original-input oracle. Consult
`docs/WP12-DESIGN.md` and `docs/WP12-REPORT.md` for current validation status.

Use existing CPU torch and SDK2.10.1 installations, absolute paths and fresh runs:

```sh
python3 tools/prepare_wp12_reference.py \
  --run "$CPU_RUN" --source-dir "$PINNED_SOURCE_DIR" --python "$TORCH_PYTHON"
python3 tools/guarded_run.py --profile cpu-reference \
  --cache "$CACHE_ROOT" --work "$CPU_RUN" --spec "$CPU_RUN/steps.json"
python3 tools/prepare_wp12.py \
  --run "$SDK_RUN" --image "$EXISTING_SDK_IMAGE" --reference "$CPU_RUN"
python3 tools/guarded_run.py --profile sdk \
  --cache "$CACHE_ROOT" --work "$SDK_RUN" --spec "$SDK_RUN/steps.json"
```

CPU2GiB/60s; SDK20GiB/zero swap, compile300s/simulation240s. One heavy job at a time,
8GiB available RAMreserve,20GiB cache and32GiB diskreserve. Actual SRAM plus4KiBstack
must fit48KiB before simulation. Inspect result/supervisor, source and compiled
hashes, SRAM admission, observations, initializations, overflow snapshots, complete
transfer journal and normal-stop lifecycle. Preserve failures and never reuse a
frozen directory. No new model payload or environment is needed.
