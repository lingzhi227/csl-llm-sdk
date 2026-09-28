# Current user target: resident west-to-east spatial pipeline

User clarification received by the control session on 2026-09-27, recorded at
23:45 UTC. This document supersedes conflicting performance-target and placement
statements in active plans. It does not alter the scope or results of frozen
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

## Primary performance metric

The target is at least 2,000 completed generated output tokens per second in
aggregate during sustained steady-state operation of the filled spatial pipeline.
Use a declared reproducible stream/concurrency of independent real requests as
needed to occupy the stages. Each request must preserve ordinary autoregressive
dependencies and produce correct outputs through all original layers and the
complete vocabulary head. This is not a 2,000 tokens/s per-request requirement.

Compute throughput as completed generated output tokens divided by measured
steady-state wall-clock duration. Declare request concurrency, prompt lengths,
generation lengths, context/state capacity and the measurement window before
observing results. Do not count input prompt tokens, internal layer tokens,
synthetic independent hidden states, incomplete passes or projected kernel rates
as generated output throughput. Include scheduling, backpressure, token feedback
and transfers required by the actual serving path during that window.

Report initialization, pipeline fill/drain, prefill, TTFT, per-request inter-token
latency and full-run throughput separately. The goal does not promise that one
request's later tokens bypass the 64-layer autoregressive feedback dependency.
The previous 500 microseconds-per-request-token and 7.81 microseconds-per-layer
budgets are not the acceptance constraints for this aggregate-throughput goal.

## Work continuity and active plan reconciliation

Preserve existing kernels, numeric evidence, communication protocols, compilation
results and failures. Reuse them where they fit the spatial stages. Reconcile the
currently running bounded compile and retain its outcome/resource receipt; do
not cancel it merely to restate this clarification, and do not automatically
extend the superseded whole-wafer shared-layer design with another hardware run.

Update the live STATE, status, acceptance, performance README, architecture,
measurement and next-step documents to this target. Mark temporal-overlay plans
as historical candidates, not the final architecture; preserve frozen evidence.
Any active task goal text retaining the old per-request interpretation must be
explicitly identified as superseded, without falsely marking unfinished work
complete merely to change the wording.

Next, make the spatial stage placement and admission/feedback schedule concrete:
full-model weight/state/code/buffer capacity, inter-stage paths, stage service
times and bottleneck balance, then a connected multi-stage execution and complete
64-layer pipeline. Use these results to revise the plan rationally. Do not reset
the project or replace integration with an open-ended series of isolated probes.
