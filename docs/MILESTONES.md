# Accepted milestones

## WP00 — September 10, 2026

Resource admission, serial heavy-task locking and three host bit codecs; a single
WSE-3 PE copied seven BF16 raw bit patterns with native MEMCPY_16BIT on SDK 2.10.1.
H2D, device copy and D2H preserved `0000 8000 3f80 bf80 7f80 ff80 7fc1`.
SDK stop returned normally and both compiler/simulator processes exited zero.

Outer wall times were 4.083871 s for compilation and 2.058936 s for simulation.
These include container/service overhead and are not hardware performance.
The first attempt failed before compilation due to host/container mounting; the
controller approved a corrected launch with unchanged CSL/driver source.

This milestone establishes a tiny transfer control, not BF16 arithmetic, request
reset, packed-stream SDK support, model generation or physical cluster inference.
Fourteen lightweight host tests pass. See [WP00 report](WP00-REPORT.md) and
[compact evidence](../evidence/wp00.json). Future entries require scoped acceptance.
