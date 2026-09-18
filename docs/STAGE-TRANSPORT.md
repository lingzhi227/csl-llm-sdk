# Connected dense-stage physical transport

A 600-PE application on physical WSE-3 completed two changed synthetic token
epochs using the same dense input-frame, matrix-lane and row-packet modules
intended for original complete layers. The 408 matrix PEs occupy five adjacent
stripes spanning input widths 5120, 6144 and 17408. A fixed 17472-word frame
gives every lane an explicit phase and a 96-word ordinal; inactive lanes drain
their slots, including the padded final input columns.

The four phases run from device events within one compute command per epoch.
Each lane rearms only after its own input/reduction work and retained output
release. Row consumers acknowledge completed packets independently of sender
buffer release. A diagnostic holds the long zero tail after the final active
96-word slot, forcing six early next-phase READY events over the two epochs.
The source retains one pending READY and finishes the previous period before
opening the next. Production source defaults to no diagnostic tail hold.

All 5013504 BF16 weight halfwords occupy one 24576-byte bank per matrix PE.
Native32 upload packs two halfwords per transfer word, and native16 reads the
same bank before arithmetic. Complete final readback of both aliases checks
retention; the upload totals 10027008 bytes. This validates the actual server
and device behavior, extending the earlier host API/source evidence.

Independent integer/bit review checked 78336 input words, 104448 local output
rows, 104448 complete stripe-broadcast rows, 2560 root/consumer BF16 outputs,
all weight halfwords, all descriptors, six early READYs and the exact 188-event
journal covering 89 copies and five launches. Dyadic one-hot coefficients make
the arithmetic exact for this transport fixture. Original floating-point
contraction accuracy remains covered by the separately accepted kernels and
original-operand audits; this fixture does not establish full-layer accuracy.

Compilation produced 309 application ELF programs with complete disjoint
600-PE placement. All 301 distinct matrix programs contain exactly one aligned
24576-byte bank. The maximum ordinary static section end plus a 4096-byte stack
allowance is 39440 bytes, below the 48128-byte gate. This is compiled allocation
evidence, not a measurement of maximum dynamic stack use. Earlier compile
attempts failed on a global comptime pointer cast and a builtin-shadowing name;
their failure/release records remain preserved. The accepted correction changes
the cast's execution point and the variable name without changing arithmetic.

The runtime succeeded and was independently verified released before both
offline audits. The host runtime stage took 198.229 seconds, capture including
normal stop took 9.494 seconds, and the journal reached final readback in
1.513 seconds. These include different initialization/transfer/cleanup scopes
and are not full-model token latency or an accounting charge. The 14 saved
archives total 21356320 bytes and remain outside Git; their hashes are published
in the [capture receipt](../evidence/stage-transport-capture.json).

The reusable transport is accepted for this exact synthetic geometry and
sequence. Original complete attention/recurrent layers, full-stage placement,
state restoration and complete CSL generation remain integration work. No
unchanged simulator or physical rerun is needed for this accepted fixture.

[Source](../examples/stage_transport) · [Independent acceptance](../evidence/stage-transport.json)
· [Compile acceptance](../evidence/stage-transport-compile.json)
· [Frozen offline audit](../evidence/stage-transport-audit.json)
