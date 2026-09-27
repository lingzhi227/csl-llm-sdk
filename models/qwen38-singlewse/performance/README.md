# Qwen3.8 WSE-3 spatial performance work

This is the performance successor to the immutable functional baseline at
`6c2f4f5685478ee100f167e42a9f7f22f57dca35`. The baseline implementation, successful
sentence captures and failed strict CPU comparisons are retained unchanged.

The acceptance target is **one physical WSE-3, the complete pinned original
Qwen3.8-27B-FP8 text model, correct dependent sentence generation, at least
2,000 output tokens/s per request**. Batched throughput, simulated time,
partial layers, projected operator rates and commercial serving claims cannot
satisfy that target. The full-model speed target remains unmet; qualified component milestones are recorded below.

The first executable slice is a compiler-generated 2D multicast / acknowledgement
microbenchmark, now qualified on 256 physical WSE-3 PEs. It establishes explicit routes, queues, event ownership and
same-PE cycle timing before adding model computation. It is not model inference.

Run the source checks with `python3 -m unittest discover -s performance/tests`.
Run `python3 performance/tools/build_mesh.py --output <new-directory>` to lower
a checked mesh plan into CSL. See `docs/MILESTONES.md`, `docs/ARCHITECTURE.md` and `docs/MEASUREMENT.md`.

P3 qualifies packed-vector FP8 decoding and original-weight 2x128 native dot
products on real WSE-3: 1,286 cycles versus 11,316 for the unchanged scalar tile,
with exact matched outputs for the frozen cases. This is a local operator speedup;
complete model performance is still unqualified. `spatial/banks.py` supplies a
compact all-matrix ownership/storage candidate with explicit unproven placement
and SRAM gates. Run `performance/tools/plan_banks.py --output <fresh-json>` to
reproduce the estimate without loading checkpoint payloads.

P4 compiles and simulates the largest candidate resident bank at 46,768 bytes
including its declared stack; all112 FP8 slots and retained buffers pass. Prefetch
and compute are measured separately. There is no network overlap or whole-model
SRAM admission yet; see MILESTONES.md for the unimplemented paths and timing scope.
