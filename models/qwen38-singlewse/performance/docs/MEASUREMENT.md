# Current measurement contract: stateful multi-turn dialogue

The September28 user clarification supersedes the former independent-request
aggregate target. The full contract is [DIALOGUE-ACCEPTANCE.md](DIALOGUE-ACCEPTANCE.md).
Run the complete original Qwen/Qwen3.8-27B-FP8,
revision017b9c7af6b5689d5dd426a76e0bc077eb5ca20a, all64 layers and the full248320-row
head on one physical WSE-3. One conversation must retain actual context across
successive user prompts and generate correct dependent replies.

Primary acceptance is at least2000 actual assistant content tokens/s, measured
as total emitted output tokens divided by the sum of all measured turn response
durations. Start each duration when the runtime accepts that turn's full new
input; end when its complete reply is available. Include appended-input prefill,
TTFT, dependent decoding, scheduling, state synchronization and output transfer.
Exclude user think time and initial model loading. Report those exclusions,
per-turn TTFT/ITL, pure decode speed, prompt processing, complete response time,
context lengths and the complete all-turn count/duration separately. Concurrent
independent conversations and prompt/control/internal tokens do not count toward
this target. A pure decode-only2000tps result is a separate intermediate result.

Freeze the tokenized multi-turn workload, template, generation settings, output
count convention, context bounds and measurement boundaries before observations.
Use the actual previous assistant reply in subsequent context. Preserve raw
per-token IDs/timestamps and every layer/state position. No favorable post-hoc
window selection, arithmetic averaging of per-turn rates, or component/simulator
rate can establish this physical multi-turn result.

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
