# Current measurement contract: aggregate resident pipeline throughput

This supersedes [HISTORICAL-MEASUREMENT-V1.md](HISTORICAL-MEASUREMENT-V1.md) for
future performance acceptance. It does not alter old experiment scopes/results.
Model: Qwen/Qwen3.8-27B-FP8 revision017b9c7af6b5689d5dd426a76e0bc077eb5ca20a,
all64 original layers and full248320-row untied head, on one physical WSE-3.
The target is >=2000 **completed generated output tokens/s in aggregate** during
sustained steady-state serving. Independent requests may overlap; each request
uses ordinary dependent autoregressive decoding and resident isolated state.

Before observing any performance result, freeze tokenized prompts, request
concurrency, arrival/replacement policy, prompt/generation limits, context capacity,
EOS counting convention, generation settings, warmup/fill rule and measurement
window. Declare a fixed steady-state time interval [start,end); count only actual
generated-token completions observed inside it, divided by end-start in seconds.
The same model/request completion rule must apply at both window boundaries.
Keep raw per-request position/token/timestamp records and full model epoch evidence.
No window selected afterward for its favorable rate can satisfy acceptance.

Prompt ingestion tokens, internal stage completions, injected hidden states,
partial layer passes, projected matrix rates and CPU/simulator outputs are not
generated-output throughput. Include real scheduling, backpressure, observation
and token-feedback transfers inside the serving window. The physical data path
must produce correct complete sentences through the complete model. Device-only
cycle estimates cannot replace host-observed output throughput.

Report concurrently: request/context capacity, actual active concurrency over
time, prompt and generated lengths, TTFT distribution, per-request ITL distribution,
steady-state aggregate rate, full-run aggregate rate, prefill, initialization,
pipeline fill/drain and any request replacement/reset overhead. With onlyC live
decode requests, target throughput entails mean request feedback cycle <=C/2000s;
this consistency check is not a prediction. A slow stage, feedback, head or memory
capacity limit must be visible rather than hidden by summing per-layer counts.

Freeze meaningful independent numerical tolerances for changed operators and
reduction schedules before viewing new complete-model outputs. Preserve BF16 and
FP8 scale boundaries; fluency alone does not qualify numerics. Capture original
weight/source/artifact identity, all layer/state epochs, complete vocabulary
selection, warm reset/replay and isolation between two nonidentical requests.
The historical failed strict CPU comparison remains a failed historical result.

For components, report original shape, input/output timing boundary, payload,
PE rectangle, iterations, same-PE48-bit cycle differences, host wall time and
compiled ELF+stack. Do not compare different PE clocks without calibration or
convert cycles using an assumed frequency. Connected stage timing must include
actual upstream production, redistribution and successor consumption.

Every hardware experiment uses a fresh source manifest, shared hardware lock,
account-active gate, bounded runtime and invocation-correlated cleanup. Record
owned jobs/system assignments as resource evidence, distinguish compilation and
queue time, and verify no remaining owned assignment after use. SSH disconnection
is not release; provider billing totals remain unknown unless actually available.
