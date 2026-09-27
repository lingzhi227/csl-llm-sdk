# Full-model route epochs: next integration gate

Status: full-model integration remains unfinished. P14 now generates and audits
complete abstract reduction forests and memory completion edges; see
[PROJECTION-LOWERING.md](PROJECTION-LOWERING.md). The route-word backend and a
bounded alternating-path teardown/rebind experiment are implemented. Neither
that experiment nor P13's cohost state component supplies full-matrix operand
multicast, output return or arbitrary-forest epoch transitions.

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
`csl/route_word.csl` now provides a compact single-RX/single-TX backend using SDK
enum values and masked register writes; `route-api-inventory-002.json` records
the installed definitions. SDK 2.10.1 requires per-color
`@set_teardown_handler` callbacks to compose with memcpy's shared dispatcher.
Binding task 29 directly conflicts with that dispatcher. Input/output queue
color rebinding APIs are available and are exercised by the bounded experiment.
Their admission to a complete matrix schedule remains required.

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

## P15 columnar candidate follow-up

The paired-column overlay in `COLUMNAR-INPUT.md` changes the K136 storage ring
and BF16 phase while preserving all state/value coordinates. Before executing a
full matrix, regenerate shared-input bundle segment descriptors and all four
reduction forests against these new storage classes; do not reuse P14's old
owner addresses. Independently verify the K136 748-column serpentine bridges,
exact row/K owners, BF16 exclusions/phase and every concurrent input/reduction
color and endpoint. The new input route format contains a TX bitmask and is not
compatible with the existing single-TX `route_word.csl` encoding.

Then connect original quant/value actors to horizontal producers and roots back
to the value arena, qualify bidirectional vertical forwarding and generic epoch
readiness, and compile the combined neural/input/reduction worker under the same
48,128-byte ceiling and 4,096-byte declared stack. Only a complete matrix with
actual original operands, exact owner coverage and full output/error/retention
checks can close this next integration gate.
