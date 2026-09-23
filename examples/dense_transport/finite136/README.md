# Finite 136-PE dense transport source snapshot

This is the source snapshot for an accepted physical WSE-3 component: synthetic
matrix arithmetic, collective reductions, routed rows, held acknowledgements,
continuation and reset replay. It is not a complete model or a standalone runtime
distribution. See the [report](../../../docs/DENSE-TRANSPORT-FINITE136.md) and
[evidence](../../../evidence/dense-transport/finite136/summary.json).

`device/` contains the 11 byte-exact CSL sources from the accepted appliance
compile. `fixture/` contains the exact synthetic operand and finite numerical /
boundary routines, typed copy contract and transport ledger used by HW003.
The [source map](source-map.json) records their hashes and the canonical compile,
runtime and acceptance bindings. No weights, SDK distribution or appliance
binaries are included. The host runtime/supervisor and private raw captures are
not part of this source snapshot.

To inspect the accepted contract, follow `layout.csl` into `app.csl`,
`matrix_lane.csl` and `tree_router.csl`; compare exported banks with
`fixture/copy-contract.json`. `probe_fixture.py` defines the exact finite input
domain; `probe_numeric.py` and `probe_boundary.py` describe the executor checks.
Running these modules alone does not recreate the physical result. Physical
reproduction requires a compatible Cerebras environment, compiling the exact
layout, checking actual banks and code, supplying the declared operands, running
positions 0/1/reset/0, and retaining full raw snapshots and operation journals
for independent audit. Do not reinterpret host capture wall time as token latency.
