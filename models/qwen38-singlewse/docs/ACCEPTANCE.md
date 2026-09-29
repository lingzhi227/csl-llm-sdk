# Active performance acceptance

The complete pinned original Qwen3.8-27B-FP8 must execute on one real WSE-3 with
all64 layers in resident spatial stages, a complete head and actual token feedback.
The September28 target is >=2000 average output tokens/s for one continuing
multi-turn conversation, preserving actual prior user/assistant context. The
primary rate is total assistant content tokens divided by summed turn response
times, including new-input processing, TTFT and generation; user think time and
initial loading are excluded. Report pure decode rate, ITL and complete response
time separately. Independent-request aggregate throughput no longer qualifies.
The complete contract, including persistent state and growing context, is in
[performance/docs/DIALOGUE-ACCEPTANCE.md](../performance/docs/DIALOGUE-ACCEPTANCE.md)
and [MEASUREMENT.md](../performance/docs/MEASUREMENT.md).

Full model/weight identity, correct complete sentences, independent numerical
qualification of changed operations, multi-turn context retention, explicit
capacity, actual request isolation and warm reset are required. Layout/byte estimates and component rates do not pass this contract.
Current metadata/protocol checks, original packing samples and selected backend
compiler admission do not establish connected neural or model acceptance.
The original strict functional criteria and unsuccessful comparisons below remain
preserved as historical evidence; they are not retroactively marked passed.

---

> Historical strict numerical target: this contract was not met by the initial
> functional release. The user closed the phase after complete sentence generation
> and the final local norm check. See PHYSICAL-INFERENCE-RESULT.md and
> evidence/functional-milestone-001/COMPLETE.json. Original criteria and failures
> remain unchanged below.

# Complete inference acceptance

The user's September 26 clarification makes **complete single-system sentence
generation** the acceptance boundary. Parameter count is secondary. The model
must be Qwen3.8. Smaller official models can supersede 27B if they are actually
released; community distillation into Qwen3.5 is not silently substituted.

Current official target: Qwen3.8-27B-FP8, pinned in configs/hub.json. The 64-layer
language network and all 248,320 output entries are required. Text-only ordinary
autoregressive inference does not invoke the optional vision or MTP pathways.

Required evidence:

1. Every required original tensor and tokenizer file is pinned and checked
   against the publisher. Weights remain compressed/resident after initial
   loading; state remains on the wafer between positions.
2. A single physical CS-3 runtime executes all neural operations: embedding,
   48 Gated DeltaNet layers, 16 full-attention layers, all MLPs, normalization,
   complete head and next-token selection. Host-side neural computation or
   per-layer weight streaming cannot satisfy the resident target.
3. Prompt ingestion and dependent decode are continuous. Each generated token
   feeds the next step; no reference hidden states are injected. Device output
   must decode into complete sentences from fixed prompts, with EOS/length
   handling, positional consistency and clean request reset.
4. The independent reference uses the **same official FP8 checkpoint**, with
   explicitly qualified dynamic activation quantization and BF16 boundaries.
   BF16 checkpoint outputs and dequantized-weight-only outputs are separate
   comparisons, not interchangeable reference definitions. Freeze full-model
   tolerances and prompts before inspecting candidate full-model outputs.
5. Measure actual prompt time/TTFT, each dependent decode step, generated-token
   count and end-to-end elapsed time. Report initialization/weight transfer
   separately. Microkernel timing and partial-layer extrapolation cannot
   substitute for full-model measurements.
6. Bind source/artifact/weight/reference identities, preserve failures and raw
   outputs on remote storage, and verify all owned jobs have terminated and
   released the physical system after capture.

The initial integration may use a declared short context capacity (currently
96 compiled; physical numerical acceptance pending). Capacity must accommodate the fixed sentence
workload, including its prompt and full generated output; over-capacity input
must be rejected rather than silently truncating attention or overwriting KV.
The entire model remains intact at every accepted context capacity.

No individual kernel, 20-layer stage, CPU reference, storage calculation or
compile-only result satisfies this acceptance document.

## P49 transport and resident SRAM scope

P49 adds bounded bidirectional color translators for resident groups 14 and 15,
bringing the static original-coordinate plan to five of sixteen groups. It
preserves all original banks and prior router owners while adding 1,155 entries.
Fresh corrected compile031 covers 11,388 PEs / 8,303 ELF images, with a maximum
48,096 bytes including 4,096 bytes of stack and a 32-byte minimum margin.
Two exact-cohost simulations each pass four continuing calls: 9,288 forward
words, 6,480 returned words, 336 raw markers, eight weighted completions,
unchanged bank sentinels, transfer overlap and local callback drain. Independent
retained-capture audits pass. The final receive queue now remains available
until its last control marker joins all data callbacks.
Thirty-seven source and 37 publication checks pass; all 19 workstation services
are released and no new WSE job is submitted. These are synthetic transport
results, not original neural or full-model speed qualification. Eleven groups,
the serving phase controller, full-layer/model integration and >=2,000 average
output tokens/s for actual multi-turn conversation remain unfinished.
See [Resident color-domain bridges](../performance/docs/GDN-BRIDGES.md).
