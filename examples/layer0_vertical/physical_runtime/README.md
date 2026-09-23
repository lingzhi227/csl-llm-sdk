# Accepted original vertical Layer0 host capture and audit

This directory publishes the exact accepted Python capture, watchdog, bounded
weight uploader and post-release numerical audit closure for positions 0/1/reset/0.
The [device source](../top_spine) is unchanged from the accepted static artifact.
The [physical report](../../../docs/LAYER0-VERTICAL-PHYSICAL.md) records the
0/438/0 nominal BF16 differences and the limits of conditional acceptance.

`source-map.json` pins every byte-identical source file. The qualified arithmetic
namespace has its own file manifest. No model weights, raw arrays, compiled
artifact, private paths or executable admission is included. The sanitized input
template preserves artifact, binding and prepared-input hashes.

These are historical deployment sources, not a portable one-command launcher.
`runtime_gate.py` deliberately requires the original frozen manifest, independent
input/bank qualification, separate physical/offline admissions and the qualified
storage device. A new deployment must establish those inputs and resource
conditions and record a new manifest; do not treat a placeholder as authorization
or remove the gates. Runtime and audit resource limits are in `runtime-profile.json`.
The runtime performs actual device execution and capture; mathematical checking
runs only after physical release. No CPU model-forward substitute is used.

Private checkpoint control restoration and adjacent-layer execution remain
future work. Published source does not imply full-model or token-performance
qualification.
