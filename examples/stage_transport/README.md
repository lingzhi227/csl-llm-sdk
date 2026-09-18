# Connected dense-stage transport on physical WSE-3

This accepted synthetic fixture exercises the input-frame, matrix-lane and
row-packet modules intended for complete model stages. It is not a neural layer
or complete-model result. The published files are exact frozen compile/capture
sources; `source-map.json` records their hashes and origin.

`layout.csl` places 408 matrix PEs in a 200 by 3 application. Five adjacent
stripes use 54, 54, 64, 54 and 182 matrix columns, including one 37-row output.
Every matrix PE consumes its ordinal's 96 words of a shared 17472-word frame,
even in inactive phases. Two changed synthetic inputs traverse all four phases
without a host neural intermediate or a host launch between those phases.

`transport_capture.capture(runner, types, order, root)` accepts an already-created
bounded physical runtime and performs exactly 89 copies and five launches.
It closes that runtime before returning. The caller must bind the exact accepted
artifact, enforce the 300-second runtime and resource limits, and verify scheduler
and system release. The published capture does not allocate or supervise a job by
itself. A separate `transport_audit.audit(root)` checks the saved arrays after
release; independent integer/bit review corroborated the result.

The compile used WSE3, fabric 762 by 1172, offset (4,1), memcpy with one channel,
and one compiler worker. Actual 309 ELF programs cover all 600 application PEs
without overlap. Each matrix program has one aligned 24576-byte weight bank;
maximum ordinary static end plus 4096-byte stack allowance is 39440 bytes.
`inspect_full_elf.py` uses the SDK's ELF decoder plus `elf_symbols.py` to check
placement and the single bank. It requires the native SDK Python environment.

Native32 uploads pack two original BF16 halfwords per word into that one bank.
Complete native16 readback precedes arithmetic; both aliases and all weights
remain equal after two epochs. Saved arrays and compiled/vendor artifacts are
not included. The [report](../../docs/STAGE-TRANSPORT.md) and
[acceptance](../../evidence/stage-transport.json) describe exact scope and limits.
