# WP12: original-input Q/K through persistent attention on one PE

Compose the accepted Q/K RMS256/partialRoPE64 kernel and attention256/cache8 kernel
sequentially on one PE. All arithmetic kernels remain byte-identical to the scoped
accepted versions. The wrapper owns generation, position, capacity, buffer handoff
and successful completion. Original synthetic BF16 Q/K/V/rawgate are never replaced
with host-computed candidate intermediates. Static offset weights, official inverse
frequencies and finite probes are loaded once and checked every call.

## Owned buffers and completion

`original_input` holds immutable Q/K/V/gate[256] arrays. Q/K kernel reads its first512
words and produces `qk_output`512BF16 words. A device loop copies those into the
first512 `attention_input` words and original V/gate into its remaining512words.
These arrays are distinct and all are observed. The attention kernel then appends
K/V to its owned cache and computes the final output. No host transfer feeds an
intermediate back to the candidate. Each source buffer survives its last consumer
use and readback; the next successful token may overwrite temporary work buffers.

Q/K kernel events indicate local production only, never independent host-command
release. Three formerly unused attention event fields record Q/K start, completion
and operand-copy completion. Shared success/append/valid counters and host unblock
occur after the entire path. There is one loaded runtime and one command per token.

The capacity check precedes preprocessing, position-domain checking and all data
writes. After eight successes, position8 requests capacity refusal; it must leave
all preprocessing/data/stage/cache arrays and Q/K events unchanged. Only shared
error/rejection metadata and attention control events change. Reset invalidates
cache metadata while retaining physical old slots; zero and changed calls then
prove exclusion of stale data. This is not arbitrary malformed-command recovery.

## Separate original-input numerical profile

Accepted standalone policies are unchanged. Post-RMS/rotary source enclosures can
reach2.875, beyond WP10's0.5component limit. This package requires both a component
enclosure<=3 and a separately computed coupled valid-score-gap<=24 for the specific
original fixtures. It does not qualify arbitrary vectors satisfying only the first
condition. The new source oracle starts from original Q/K, weights and position,
propagates current Q and every cached K uncertainty, then both BF16 score casts,
max/exp/sum/inverse/probability, BF16 probabilities, V reduction, BF16 sigmoid gate
and final product. Shared correlations are conservatively over-approximated by
triangle inequalities. Structural zero is retained when exactness is proved.

The pinned official CPU composition uses its own normalized/rotated Q/K prefix and
original V/gate, plus a separately evaluated causal last-row reference. All2560
nominal output bits agree. The maximum source score-gap upper bound is3.640625.
Accepted trig/exp/root/inverse and per-operation cast gates remain unchanged.
Conditional actual-stage validators are separate and do not call the old source
oracle with a silently widened input domain.

Before any SDK run, the final source interval numeric spans by call are
[0,.0078125,0,0,.00390625,.00390625,.01171875,.015625,0,.00390625]. Single-valued
output counts are[256,1,256,256,74,0,0,0,256,2]. Some narrow near-zero intervals
contain many BF16 values (up to30502); count alone is not numeric width. Maximum
absolute radius is.0078125. These conservative composed bounds are not bit-parity
claims; strict stage/cast/copy/cache checks remain independent acceptance gates.
The bounds were reported to the controller before SDK and are not adapted from
candidate observations.

## Complete observations and resource plan

Each call checks original/static inputs, guards, handles, Q/K stages and their
source intervals, trig/zero/signed-zero/RNE/tail rules, exact device operand copy,
complete physical cache against actual producer history and valid K against
independent source intervals, attention stages, final source output and nominal
CPU differences. Overflow snapshots every array, including preprocessor state.
Every physical transfer has durable entry/exit records. Normal stop, unchanged
frozen source/compiled hashes and exact owned-unit cleanup are required.

Fixed297copies/159134hostu32slots/390188native bytes/13launches; maximum physical
host buffer16392bytes. The additional attention-input initialization sets real
guards and finite poison, since host input uploads no longer initialize that
consumer buffer. No runtime candidate arithmetic is performed by the host.

Estimated195seconds vs240hard limit: standalone totals were167.57seconds over
316copies/376692native bytes/25launches. The composed plan has19fewer copies and
12fewer launches, while retaining full stages and adding all-stage overflow checks
and13.5KiB payload.195seconds allows additional validation/copy overhead and stays
within the210-second planning ceiling. Compile300seconds and actual ordinary end
plus4096stack<=49152 remain mandatory; adding old footprints is not admission.
One PE is preferred; any SRAM failure requires an explicit revised topology plan,
not an automatic second PE or larger memory ceiling.

One heavy workstation job,20GiB/Swap0,8GiB available RAMreserve,20GiB active cache,
32GiB diskreserve. No download/install/GPU/MPI/hardware or additional agents. Scope
remains one synthetic head for positions0–7, without projections, full GQA, full
attention layer/model, long-context or physical-system claims. WP09 remains open.
