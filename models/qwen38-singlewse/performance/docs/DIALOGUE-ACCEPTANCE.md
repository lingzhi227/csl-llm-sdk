# Stateful multi-turn dialogue acceptance

The user's September28 clarification defines the real workload: a user supplies
text, the WSE-3 generates a reply, then later user text receives a new reply based
on the actual preceding conversation. The target is an average of at least2000
output tokens/s for this continuing conversation. More turns must be supported
as context capacity is developed. This supersedes the September27 aggregate
independent-request performance interpretation. Frozen experiments and previous
publications keep their original scopes; none is retroactively requalified.

## Primary measurement

Use one real conversation with serial user/assistant turns and original dependent
autoregressive feedback. The engineering acceptance adopts the conservative
response-inclusive interpretation: for each turn i, t_start_i is when the runtime
accepts the complete new user input and t_end_i is when the complete reply is
available to the client. The primary rate is:

    sum(actual assistant output tokens_i) / sum(t_end_i - t_start_i) >= 2000/s

Include appended prompt/template processing, TTFT, actual decoding, scheduling,
backpressure, output transfer and necessary state synchronization. Exclude human
think time and initial model loading, but report initialization separately.
Count actual emitted assistant content tokens with a declared tokenizer; exclude
prompt tokens, chat-template/EOS control markers, internal layer completions and
unemitted reasoning. Record every generated ID and classify counts before the
experiment. Do not average per-turn rates arithmetically or pick the fastest
subwindow. Independent conversations cannot contribute tokens to this metric.

Also report per-turn TTFT, first-to-last output-token decode rate, token ITL,
prompt ingestion time, complete response time, history length, peak context and
all-turn output totals. Short replies must not obtain an inflated rate from a
one-token/zero-time denominator. A separate multi-user capacity experiment is
allowed, but cannot pass this single-conversation target.

## Persistent state and turn boundaries

Preserve each layer's actual convolution history, GDN recurrent state and full
attention KV cache across user turns. Maintain absolute token positions and the
original chat template. A completed reply releases temporary work buffers and
communication leases, not the conversation's persistent state. A new conversation
or explicit reset has a distinct drained-state clear and reset acknowledgement.

Append the new user/template suffix against a verified token prefix. Track two
separate positions: tokens emitted/selected and tokens already committed through
all64 layers. A final sampled token or EOS may not yet have been consumed by the
model when generation stops; advance it exactly once when required before the
next suffix. Do not duplicate it, drop it or treat host output as a KV commit.
Changed earlier history requires an explicit supported rebuild, not reuse of
incompatible state. Any interruption must drain accepted work and preserve a
well-defined committed prefix before another turn is admitted.

Continue with the same loaded original weights and same conversation state.
Host tokenization, scheduling and output rendering are permitted; host neural
intermediates and per-layer weight streaming are not. Increasing turn count must
account for actual growing attention storage and positions. The current two-slot,
context96 capacity estimate is historical candidate capacity, not a qualified
multi-turn implementation. Do not silently truncate history, wrap positions or
reset GDN state to make a longer conversation fit.

## Qualification before a speed claim

Freeze a transcript with at least three dependent turns, generation settings,
context/output limits, tokenizer/template identities and timing/counting rules
before device output. Follow-up prompts must require earlier user facts and/or
actual preceding assistant output. Compare the full accumulated conversation to
an independent same-checkpoint reference and qualify changed arithmetic under
predeclared criteria; plausible prose alone is insufficient. Include zero/reset
replay, a fresh independent conversation after reset, stop/continue boundaries,
and declared-capacity overflow rejection. A longer growing-history workload must
be declared for subsequent capacity milestones.

The complete original Qwen3.8-27B-FP8 and all64 resident spatial stages remain
required. Cross-kernel fusion must shorten the latency of one dependent path:
consume produced chunks directly, retain state near its updates, overlap legal
communication and computation, and preserve numerical/lease boundaries. Filling
stages with unrelated requests does not satisfy this target. In steady decoding,
2000 tokens/s implies no more than500us mean feedback time for one conversation;
prompt/turn overhead makes the primary response-inclusive target stricter.
This is a budget to measure and optimize, not an attained performance result.

## Current capacity planning observation

`evidence/dialogue-state-capacity-001.json` checks the exact historical two-request
plan: a second independent context accounts for160,235,520 persistent-state bytes
(2,949,120 convolution,150,994,944 recurrent and6,291,456 KV at context96).
Specializing for one continuing conversation can make those bytes available for
longer history, consumer-local state and fused code/scratch. This does not remove
any state belonging to the retained conversation or change original weights.
Two temporary work slots are a separate communication-overlap decision; they are
not removed by this accounting. No longer context, changed placement, SRAM fit
or faster execution is admitted by the byte count. Keep the existing physical
qualification snapshot fixed; implement and requalify a new placement separately.
