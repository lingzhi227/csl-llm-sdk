# Qwen3.8 WSE-3 spatial performance work

This is the performance successor to the immutable functional baseline at
`6c2f4f5685478ee100f167e42a9f7f22f57dca35`. The baseline implementation, successful
sentence captures and failed strict CPU comparisons are retained unchanged.

The acceptance target is **one physical WSE-3, the complete pinned original
Qwen3.8-27B-FP8 text model, correct dependent sentence generation, at least
2,000 output tokens/s per request**. Batched throughput, simulated time,
partial layers, projected operator rates and commercial serving claims cannot
satisfy that target. No performance success is claimed yet.

The first executable slice is a compiler-generated 2D multicast / acknowledgement
microbenchmark. It establishes explicit routes, queues, event ownership and
same-PE cycle timing before adding model computation. It is not model inference.

Run the source checks with `python3 -m unittest discover -s performance/tests`.
Run `python3 performance/tools/build_mesh.py --output <new-directory>` to lower
a checked mesh plan into CSL. See `docs/ARCHITECTURE.md` and `docs/MEASUREMENT.md`.
