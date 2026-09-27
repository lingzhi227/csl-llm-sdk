# Full-model route epochs: next integration gate

Status: design notes, not generated or admitted routes. P12 owns all model data;
P13's cohost state component does not supply full-matrix communications.

The atlas reuses physical banks for different matrices and K widths. Static routes
for one contraction therefore cannot silently stand in for the complete model.
Its two GDN state bands per actor row also share vertical corridors: duplicating
one isolated head's colors would create conflicting router entries. The next
lowering must check the whole region, including intermediate router PEs.

The SDK distinguishes limited switch-position changes from route reconfiguration
in teardown. Switch commands advance predefined positions after passing through
a router; they do not replace an arbitrary routing table. See the primary
[switch/control-entrypoint tutorial](https://cerebras-sdk-docs-140.netlify.app/csl/code-examples/tutorial-topic-07-switches-entrypt).

The official [7-point stencil example](https://cerebras-sdk-docs-130.netlify.app/csl/code-examples/benchmark-7pt-stencil-spmv)
shows runtime `tile_config.color_config.reset_routes`, switch reset and teardown
exit. Its transition task waits for both operation completion and teardown
arrival. This establishes an API candidate, not qualification of our route plan.
Installed SDK 2.10.1 contains these APIs; their module hashes and signatures are
recorded in `evidence/route-api-inventory-001.json`. The route description argument
to `reset_routes` is compile-time even though the call executes at runtime.
A compact route-word backend and a bounded actual experiment remain required.

Proposed lowering obligations, inferred for this model:

- Represent a route epoch separately from tensor storage, with participating and
  forwarding coordinates, routes, queue/color/DSR/microthread leases and payload
  lengths. Describe every inactive forwarding PE explicitly.
- Preserve original matrix row/K identity through each whole-K atlas segment.
  P9's fixed preorder trees can be translated along a serpentine bank path, but
  actor-row holes and the BF16-excluded state ring require independent path checks.
- Distinguish local send-buffer release from fabric quiescence. A sender callback
  alone is insufficient permission to rewrite routes. Require both numerical
  completion and the documented teardown event before reconfiguration.
- Account for source multicast, reduction, nonlinear producer routes and returned
  values simultaneously; do not count a color twice on an intermediate PE.
- Keep compact operation descriptors and precomputed local ranks. A full schedule
  replicated on every almost-full bank PE would invalidate SRAM admission.
- Measure reconfiguration and next-packet latency in the dependent critical path;
  do not subtract them from the full-model token measurement.

Separately, the scalar-per-value division in qualified spatial FP8 quantization
remains a latency target. `GUARDED-QUANTIZATION-NEXT.md` is still unimplemented;
the original bit-exact gate must precede use of any reciprocal approximation.
