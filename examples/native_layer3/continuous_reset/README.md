# Original Layer 3 continuous execution and reset

This directory records the exact device source for the accepted physical
position 0 → position 1 → reset → position 0 sequence. The
[source map](source-map.json) pins all 37 compiled CSL files: eight files are
stored here and 29 unchanged files are referenced from previous public examples.
Copy the mapped files to one flat directory to reconstruct the device source.
The map is byte exact; it does not include proprietary SDK files, weights,
compiled binaries or private remote execution configuration.

The [report](../../../docs/LAYER3-CONTINUOUS-RESET.md) explains state ownership,
candidate-specific SRAM policy, the preserved original supervisor failure and
the separately admitted postcapture audit. The [evidence summary](../../../evidence/native-layer3/continuous-reset/summary.json)
binds the physical and audit records. The [full operator audit](../../../evidence/native-layer3/continuous-reset/audit.json)
contains the actual conditional checks and nominal comparison for each serial.

The result qualifies this finite Layer 3 sequence and reset replay. It does not
establish whole-model inference, bitwise CPU parity, a propagated whole-layer
error enclosure or an unbounded decode loop. Original capture exit 1 and the
absence of a COMPLETE record are preserved.
