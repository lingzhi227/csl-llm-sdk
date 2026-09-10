# Original hidden input through Q/K preprocessing

Accepted scope: one dense projection-input hidden vector, all5120 input columns,
Q256/K256 produced on four PEs, exact device handoff to a fifth PE, trained RMS256
and partial RoPE64 at text position1. See the [report](../../docs/WP14-REPORT.md),
[protocol](../../docs/WP14-PROTOCOL.md) and [evidence](../../evidence/wp14.json).
The original input is the frozen projection input, not an embedding or completed
upstream model block. This milestone does not include V/gate attention or KV.

Reuse the pinned selected weights and completed CPU reference from
[WP13](../../docs/WP13-REPORT.md). No additional model download is needed. Set
`QWEN_CACHE` to the qualified working cache, `QWEN_REFERENCE_PYTHON` to the existing
CPU Python interpreter with the reference dependencies, and `QWEN_SDK_IMAGE` to
the separately installed SDK2.10.1 image. Commands below run from the repository
root. Work directories must be fresh; the supervisor refuses an already attempted
run or a surviving project unit.

```sh
python3 tools/prepare_wp14_reference.py \
  --run "$QWEN_CACHE/wp14-reference" \
  --previous "$QWEN_CACHE/wp13-reference" \
  --python "$QWEN_REFERENCE_PYTHON"
python3 "$QWEN_CACHE/wp14-reference/tools/guarded_run.py" \
  --profile cpu-reference --cache "$QWEN_CACHE" \
  --work "$QWEN_CACHE/wp14-reference" \
  --spec "$QWEN_CACHE/wp14-reference/steps.json"
python3 tools/prepare_wp14.py \
  --run "$QWEN_CACHE/wp14-sdk" --image "$QWEN_SDK_IMAGE" \
  --reference "$QWEN_CACHE/wp14-reference" \
  --projection-reference "$QWEN_CACHE/wp13-reference"
python3 "$QWEN_CACHE/wp14-sdk/tools/guarded_run.py" \
  --profile sdk --cache "$QWEN_CACHE" \
  --work "$QWEN_CACHE/wp14-sdk" --spec "$QWEN_CACHE/wp14-sdk/steps.json"
```

CPU limits are one thread,2GiB RAM,Swap0 and60seconds. SDK compile and simulation
each have a300second ceiling,20GiB RAM,Swap0 and one heavy job. The active cache
has a20GiB ceiling, with8GiB available RAM and32GiB free SSD reserves. Compile
must admit all five actual ELF ordinary ends plus4096byte stack allowances under
49152bytes before simulation. No environment installation is performed.

The accepted code retains its frozen planning metadata and some draft comments;
the report and evidence identify the successful experiment. Preparation paths and
run-directory metadata will differ on another machine. The arithmetic, fixed
inputs, framing and704-copy/59-launch sequence remain the reproduction contract.
Original projection and Q/K source intervals are separate from actual-operand
stage checks. Full private parameter buffers are represented publicly by typed
shape/hash records; recreate those buffers from the original selected files.
