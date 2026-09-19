# Finite physical fan-in compilation

Compile exactly the six frozen ready-fanin-sdk-001 CSL files, including the
unimported production-packets.csl baseline. The independent record checker is
also byte-identical. Application178x2, physical fabric762x1172, offset4,1;
origin0,0; actual phase2 peer2,0; all136 concurrent producers40..175/y0; idle
row1 vertical diagnostic sinks. No model, neural kernels, reset or SDK rewrite.

The predecessor produced356 SDK ELFs with maximum25,472 bytes including4KiB
stack. Its driver failed at180s during origin-records D2H. Six durable status
captures show complete expected record counts but cannot establish record
contents, source equality, reserved-word correctness or normal stop. Preserve
that failed attempt; do not rerun it or call it a transport pass.

This candidate changes physical compiler wrappers and inspection geometry only.
The actual compiled artifact must retain all frozen CSL source bytes and cover
all356 application PEs once; inspect state/status/history/header/payload banks,
independent live-buffer extents and48,128-byte static SRAM ceiling. A4KiB stack
allowance is not a measured dynamic peak. Compiler implicit DSR allocation is
not independently disassembled. Any runtime requires its own frozen admission.

One physical compile:600s, worker1CPU/4GiB; hostCPU0/AS4GiB/sampledRSS1GiB,
zero observed swap, file128MiB/log8MiB/candidate128MiB and32GiB free reserve.
Use the existing owner-correlated watchdog and exact single-attempt admission.
Require actual waiter reap and independent terminal/unassigned/PIDs-gone release.
No runtime creation, physical payload transfer or inference acceptance is included.
