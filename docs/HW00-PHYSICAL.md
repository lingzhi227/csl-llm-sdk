# HW00: original-weight eight-PE physical qualification

On September 18, 2026, the original Qwen3.8-27B layer-3 resident fragment completed
one physical WSE-3 runtime with three inputs. This qualifies the appliance backend
for the 192→128→128 graph; it does not qualify the full 5120→17408→5120 MLP, decoder,
or 64-layer text model. All prior failures and simulator receipts remain unchanged.

The new physical compiler artifact contains the exact eight accepted CSL files.
Native ELF metadata places all eight application PEs at the intended coordinates
inside a 762×1172 physical fabric. The largest ordinary static footprint plus a
4096-byte stack allowance is 39744 bytes. This is not a dynamic stack measurement.

An independent saved-output audit checked all 112 arrays: 2304 conditional dot
rows through exact integer arithmetic, 1152 actual partial-pair sums, 1920 FP32 to
BF16 casts, 384 Decimal exponential/sigmoid/product rows, handoffs, state and
timestamps. All 125 operations completed. Six original weight uploads preceded
the three inputs; complete final readbacks equal the original weights. This
establishes initial/final identity, not continuous bitwise observation.

Both compile and runtime scheduler jobs succeeded. The runtime context exited
normally; independent fresh scheduler and system queries confirmed no remaining
owned allocation. No manual cancellation was necessary. The detached supervisor's
failure paths separately exercise log overflow, termination errors, old IDs and
ownership refusal without submitting fault jobs.

Compilation took 72.37 seconds as observed by the host supervisor; the physical
stage took 190.62 seconds including initialization and cleanup. The scheduler's
RUNNING interval was 37 seconds. None is a pure kernel benchmark, token latency
or charged node-hour total. SDK/appliance packages were 2.10.0; the existing SDK
2.10.1 image was used only to inspect the physical artifact's ELF metadata.

The current full-model target uses three sequential logical stages, with all
boundary activations and per-layer persistent state saved/restored on the host.
A physical system can be reused between stages. Simultaneous three-system
reservation or direct inter-wafer transport is not a prerequisite. Next work is
the full-dimension resident MLP, then complete layers and the three-stage graph.

[Reproduction adapter](../examples/hw00) · [Acceptance summary](../evidence/hw00-physical.json)
