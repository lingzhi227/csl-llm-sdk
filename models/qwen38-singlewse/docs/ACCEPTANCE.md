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
