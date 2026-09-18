# Synthetic Q/K/V fanout fixture

This 370-PE SDK simulator fixture tests the complete phase 0 fanout and collector
pattern with synthetic finite BF16 markers. It reads no model weights and runs
no neural math. The source graph and driver correspond to independently
accepted captures; small portable import/path adaptations are noted below.

The accepted final state is all 370 PEs complete, 112 root senders and 24 heads
complete, 576 source/received packets, 192 row fanouts and 24576 exact u16 channels.
A fixed 85-second diagnostic wait precedes one core capture and shutdown. This
wait is not a performance result. No memcpy occurs after compute.

## Inputs, limits and stages

Use a configured SDK 2.10.1 Linux environment with its Python runtime, NumPy and
compiler tools. The qualified run used CPU 0 with one SDK simulation thread,
2 GiB hard memory, zero swap and 128 processes. Keep all compile outputs and cores
on the execution disk. Apply hard limits before running: 120 seconds for the
compile subprocess, 180 seconds for the driver, 64 MiB per file, 96 MiB aggregate
core output and 128 MiB total working directory. Keep 8 GiB RAM and 32 GiB disk free.
The source drivers check CPU 0 but do not replace the external resource guard.
Run one workload at a time and use a fresh directory after a failed attempt.

The stages are:

1. `compile.py` compiles the exact layout and checks all actual ELF placements and
   ordinary-section ends with a 4096-byte stack allowance under 48128 bytes.
2. `driver.py` uploads metadata, initializes, verifies the low 16 metadata bits and
   full initial state, launches compute once, captures once after 85 seconds, then
   stops. It writes raw host initialization, journal, core identities and result.
3. After shutdown and release, `offline_inventory.py` reads actual ELF symbols.
4. `csdb_extract2.py` reads the existing core through one csdb shell and invokes
   `csdb_audit.py`. It saves raw text, exact words, arrays and a derived report.
   Only `completed_graph: true` with all declared checks is graph acceptance.

The accepted later run reused its earlier 25 compiled files unchanged, instead of
recompiling. The included compiler supports independent reproduction. Its only
portable adaptation imports the unchanged ELF helper from this folder. Offline
readers omit private host/candidate-directory assertions; parsing and arithmetic
checks are unchanged. No new simulation was performed merely for these path
adaptations. Do not treat successful capture, normal stop or CLI exit 0 alone as
proof that the graph completed.

[Result and limitations](../../docs/LAYER3-INTEGRATION.md) ·
[Core reading notes](../../docs/QKV-CORE-DEBUGGING.md)
