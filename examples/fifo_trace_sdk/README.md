# Finite FIFO trace and device-gated drain fixture

This 16-PE SDK 2.10.1 simulator fixture independently qualified physical FIFO
capacities 64 and 128, exact finite legal bursts 62 and 106, and device-triggered trace
drain while the source main task had not completed. It retains the original
packet adapter and sideband path and checks their coexistence. It uses synthetic
markers, no model weights and no physical WSE allocation.

The device gate requires the sink to receive the 61/105-token prefix before it
notifies a release producer. The producer sends the exact release token to the
source, which then appends its final marker. The host does not issue a D2 H call
until the blocking device launch finishes. A separate full legal burst and
capacity-overflow probe retain their own exact histories and counters.

This qualifies the measured capacities, finite histories, main-task completion
gate and packet/sideband coexistence. It does not prove that the CPU was stalled
throughout draining, or reproduce a synchronous SDK output stall.

Under an external hard resource guard, `entry.py` runs `compile.py` then
`driver.py`; `check_capture.py` validates the retained words. Apply the qualified
150s total deadline, CPU0/one SDK simulation thread, 1 GiB hard memory, zero swap,
128 processes, 8 MiB per file, 2 MiB logs and 32 MiB total candidate. Retain 8 GiB RAM and
32 GiB free on both system and execution disks. Subprocess limits are 70s compile
and 30s driver. These source scripts do not replace the external resource guard.
Use a fresh directory and preserve failures. No automatic retry is qualified.

The accepted run used 8 D2 H copies, 28992 host bytes, three launches and 27 journal
entries, with zero H2 D. All 16 actual programs passed storage/placement checks;
maximum ordinary storage plus 4096-byte stack was 14288 bytes. Total guarded time
was 20.411393s with 512913408-byte observed peak memory. These are fixture resource
observations, not full-model performance. Every published code file is byte-exact
with the accepted fixture; site-specific installation and cgroup launch receipts
are not distributed.

[Small result](../../evidence/fifo-trace-sdk.json) ·
[Combined diagnostic milestone](../../docs/LAYER3-FIFO-TRACE.md)
