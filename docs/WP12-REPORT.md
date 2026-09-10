# WP12 report — accepted candidate002

Scope: one synthetic original Q/K/V/gate path through ordinary RMS256, device
partial RoPE64 positions0–7, device operand copy and persistent cache8 attention.
No projection, GQA, output projection, complete layer/model, long-context or hardware
claim. WP09 remains unresolved and is not borrowed as a lifecycle qualification.

## Original-input CPU reference001

Passed10calls/2560official final BF16 outputs with zero nominal differences and
zero causal-last-row differences. RMS/RoPE source uncertainty is propagated through
current Q and every retained K interval, two score casts, softmax, probability,
V reduction and gate/output casts. Max coupled score-gap upper3.640625; old WP10
absQ/K<=0.5policy is unchanged and not applied to this composition.

Final interval spans and singleton counts are documented in the design. The largest
absolute radius is.0078125, a conservative bounded-source check rather than a
bitwise guarantee. Controller source review independently checked nonzero cases
against zero/misaligned/no-gate references and multi-token cases against current-KV
only. These counterexamples do not replace strict stage/copy/cache/cast gates.

CPU5.19295000seconds/229363712peak bytes under2GiB/60s;14source and4output hashes
unchanged, owned unit absent. No further CPU reference rerun was needed.

## SDK attempts

Candidate001 failed in parsing before simulation/SRAM admission: the wrapper token
function lacked one closing brace.3.13993770seconds/309657600peak bytes;33frozen
source hashes unchanged. The failed owned unit was quiescent then exactly cleared.
Candidate002 adds only that missing brace in device source. All four reused
arithmetic kernels are byte-identical to accepted standalone artifacts. Preparation
now also checks CSL delimiter balance and Python AST before an SDK job; this is a
cheap preflight, not a replacement for compiler syntax/type checking.

Candidate002 compiled. Actual ordinary section end40848+4096stack=44944<=49152,
leaving4208bytes under the ceiling. It uses one PE without changing dimensions,
capacity or topology. The full297copy/390188native-byte/13launch simulation passed in173.92221845seconds
with204943360peak bytes, below the195-second estimate and240-second hard limit.
Compilation took4.15777590seconds/422588416peak bytes. All15checks passed on every
successful token, including actual operand copy, complete physical cache and
independent source K intervals, all Q/K/attention stage and BF16/zero/tail rules.
All2560final outputs match the official and nominal BF16 values exactly for this
fixture; this does not turn conservative source bounds into a general bitwise claim.

Overflow left all21checked data/stage/cache/QK-event arrays unchanged, with only
separate error/rejection state and attention control events changing. Both resets
preserved physical stale cache bytes while correctly invalidating metadata; normal
zero and changed calls followed. All594journal entries form297complete transfers.
Normal stop,33source/11compiled hashes unchanged and owned-unit cleanup passed.
Compile/sim units are absent/inactive, MainPID0 and empty cgroups. No third candidate
or repeated full run is needed. The controller independently audited and accepted this bounded composition. The
public snapshot passes57host tests and retains the initial parse failure. Complete
observations are split into two bounded text files referenced by the evidence
summary; no intermediate arrays are omitted.
