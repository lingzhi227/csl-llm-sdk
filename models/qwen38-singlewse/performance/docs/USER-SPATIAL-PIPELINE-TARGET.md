# Current user target: resident west-to-east spatial pipeline

The spatial placement instruction below remains active. The September28
multiturn clarification supersedes its former aggregate throughput metric; see
[DIALOGUE-ACCEPTANCE.md](DIALOGUE-ACCEPTANCE.md).

Spatial clarification received by the control session on 2026-09-27, recorded at
23:45 UTC. This document and the later dialogue clarification supersede conflicting
performance-target and placement statements in active plans. It does not alter the scope or results of frozen
experiments, historical publications, or the original functional baseline.

Immediate follow-up instruction: the user explicitly ordered an immediate
redesign of layer kernels, spatial placement, operator fusion and the related
pipeline mechanisms. This is an execution directive, not an optional suggestion
or work deferred until another temporal-overlay milestone. Switch active design
and source implementation now. Existing bounded work must be reconciled without
blocking the switch; cancel only an identified task-owned obsolete/conflicting
job when appropriate, and preserve its failure/termination and release evidence.

## Architecture

Run the complete original Qwen3.8-27B-FP8 text model on one physical WSE-3 with
all 64 layers represented as spatial pipeline stages in model order. The macro
flow is west to east, with resident weights/state and direct inter-stage dataflow.
Preserve genuine two-dimensional compute/communication within stages; choose
stage areas and internal tiling from capacity, traffic and balanced service time,
not from an assumption that every layer needs an equal-width strip.

The previous proposal that stores all 64 layers at common compute coordinates
and sequentially reuses almost the whole wafer for each layer is not the target
architecture. Local time reuse within an operator/stage remains an implementation
choice; it must not silently reintroduce that whole-model temporal overlay.

Provide input ingestion, adjacent-stage activation transfer, request-specific
recurrent/KV state, backpressure, buffer ownership and next-token feedback.
Physical host I/O endpoints must be checked against the actual ALCF SDK/runtime;
logical west-to-east flow does not by itself prove a native east-side D2H port.

## Primary performance metric, updated September28

A single continuing conversation must accept new user text, generate a correct
reply, retain the actual conversation and produce the next context-dependent
reply. The target is an average of at least2000 emitted assistant content tokens/s
across multiple turns. The active contract uses total output tokens divided by
summed turn response times, including new-input processing and generation while
excluding user think time and initial model load. Report decode-only rate, TTFT,
ITL, context lengths and turn response time separately.

The former aggregate sum across independent requests is superseded. Preserve
original model/precision/state, the64 spatial stages and genuine autoregressive
dependencies. Context/state must survive reply completion; increase supported
history explicitly rather than resetting or silently truncating at each turn.
The detailed counting, timing and correctness gates are in
[DIALOGUE-ACCEPTANCE.md](DIALOGUE-ACCEPTANCE.md) and [MEASUREMENT.md](MEASUREMENT.md).

## Work continuity and active plan reconciliation

Preserve existing kernels, numeric evidence, communication protocols, compilation
results and failures. Reuse them where they fit the spatial stages. Reconcile the
currently running bounded compile and retain its outcome/resource receipt; do
not cancel it merely to restate this clarification, and do not automatically
extend the superseded whole-wafer shared-layer design with another hardware run.

Update the live STATE, status, acceptance, performance README, architecture,
measurement and next-step documents to this target. Mark temporal-overlay plans
as historical candidates, not the final architecture; preserve frozen evidence.
Any active task wording that omits multi-turn state or retains the independent-
request aggregate metric is superseded by DIALOGUE-ACCEPTANCE.md. Keep the goal
active until the actual complete-model dialogue objective is achieved; changing
the acceptance wording does not complete it.

Next, make the spatial stage placement and admission/feedback schedule concrete:
full-model weight/state/code/buffer capacity, inter-stage paths, stage service
times and bottleneck balance, then a connected multi-stage execution and complete
64-layer pipeline. Use these results to revise the plan rationally. Do not reset
the project or replace integration with an open-ended series of isolated probes.
