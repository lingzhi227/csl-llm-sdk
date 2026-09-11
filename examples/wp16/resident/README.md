# Eight-PE resident spatial MLP

The accepted fragment evaluates192 ->128 gate/up ->SiLU/product ->128 down outputs
for dense, changed and zero inputs. Six matrix PEs hold original BF16 weights
across all three generations. An ingress/output PE and a nonlinear PE complete
the4x2 device graph. This is a reduced resident fragment supporting the three-wafer
architecture, not a full MLP layer or full-model result.

[Report](../../../docs/RESIDENT-SPATIAL-RUN002.md) ·
[Evidence and independent acceptance](../../../evidence/wp16-resident.json) ·
[Exact-source and adaptation map](../../../evidence/wp16-resident-source-map.json)

`layout.csl` places the graph. `tile.csl` is the exact generated device source;
`tile_body.csl.inc` is its reusable generation body. The arithmetic kernels,
filtered input receiver, pair reduction and packet adapter are included beside
it. `core/qwen38/resident_spatial.py` describes the roles, arrays and layout.
`resident_sdk_sequence.py`, `resident_sdk_driver.py` and `resident_sdk_checks.py`
implement the125-operation qualification, actual SDK adapter, bounded evidence
and numeric/state checks. `resident_elf_placement.py` validates actual native SDK
segment rectangles before constructing the simulator.

From the repository root, with Python and NumPy installed:

```sh
PYTHONPATH=core PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 -B -m unittest discover -s tests -p 'test_resident_*.py' -v
```

These are synthetic host fixtures and admission tests. They do not execute the
SDK or original weights, and they do not replace the published device evidence.
The source map distinguishes38 exact accepted files, one exact generator body,
one unexecuted host configuration template and two test-directory adaptations.

For a fresh SDK reproduction, obtain the separately licensed SDK2.10.1 and the
original model locally, use the accepted source recipe and input/oracle hashes
in the evidence, and prepare a new bounded candidate. The historical compile
arguments were:

```text
layout.csl --arch=wse3 --fabric-dims=11,4 --fabric-offsets=4,1 -o=out
--memcpy --channels=1 --max-parallelism=1 --dump-dsr-alloc-graph
```

Compilation belongs in its own one-worker/CPU0/1GiB/300-second guarded stage;
simulation uses512MiB/CPU0/zero swap/128tasks/420seconds hard and the shared lock.
Configure the placeholder cache/image paths and image identity in the public
admission template, inventory the newly compiled outputs, apply the separately
qualified exact11x4 auxiliary identities, then freeze and admit that new candidate.
Do not copy a historical admission into a different environment. The included
SDK002 guard is simulation-only and deliberately disables recompilation.
Follow the [resource guide](../../../docs/DEVELOPMENT.md) for runtime enforcement.
Weights, binary ELFs, local installation inventory and vendor SDK source are not
included. The metadata/source recipe alone does not provision a physical system.

Both original failures remain evidence: SDK001's host gate incorrectly interpreted
program-ID filenames as XY coordinates after a successful compile. SDK002 reused
those compiled files, passed all125operations and normal stop, then retained
exit1 when its old auxiliary hashes did not match the11x4 outputs. Independent
saved-file review qualified the new auxiliary identities and the numerical
fragment without rerunning or rewriting that historical failure.
