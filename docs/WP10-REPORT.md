# WP10 report — accepted candidate001

Scope: one synthetic post-QK-normalization/rotary D256 query/KV attention head,
BF16 persistent cache capacity8, BF16 probability/V/gate/output boundaries.
This does not qualify projection, QK RMS/RoPE, complete GQA, full attention layer,
or a model. WP09 remains unaccepted despite its separate010 diagnostic passing.

## Frozen CPU reference001

Pinned eager function bodies plus unchanged output-gate arithmetic, instrumented
only to observe output tensors, passed10original-input tokens. QK matmul/scaling,
V matmul, sigmoid and final output are BF16; softmax is FP32. Explicit causal last
row and decode agree exactly in BF16 for every output. Independent FP64 source
intervals agree with official outputs/probabilities/pre-gate attention, with zero
nominal BF16 output differences for all2560elements. All final-source intervals
are single-value for this fixture. Current-token dependence, zero-query scores,
all-negative scores, poisoned invalid slots and zero after reset are included.

Runtime3.13392653seconds; peak228712448bytes under2GiB/60s. All12frozen source hashes
unchanged; owned unit absent, MainPID0, empty cgroup. Local61tests pass, including
attention cache/overflow immutability and exact-reduction counterexample tests.

## SDK candidate001

One PE. Frozen original numerical policy/fixtures remain unchanged. Actual SRAM
ordinary section end24976 plus4096stack allowance gives29072, below49152. Compile
passed before simulation. Simulation hard deadline180seconds, expected131physical
copies/104757hostu32slots/242088native bytes/13launches; max buffer16392bytes.
Candidate001 passed all ten successful tokens, the explicit overflow refusal and
both initialization checks. All2560device BF16 outputs match nominal and pinned
official outputs exactly (zero BF16 differences, zero ULP distance). Each token
passes conditional stages, original-input final interval, complete physical cache,
input immutability, guards, metadata/events and stable symbol handles. Overflow
leaves cache/input/scores/casts/stats/stages byte values unchanged; reset retains
physical stale cache while zero token output proves valid-prefix exclusion.

Compilation4.14402182seconds/452562944peak bytes. Simulation112.05364811seconds/
194535424peak bytes, below180seconds. Exactly131physical transfers have262complete
journal records;13launches and normal stop passed. All26frozen source and11compiled
hashes stayed unchanged. Both owned units are absent/inactive, MainPID0 and empty
cgroups. No second candidate or additional simulation is needed for this package.
The controller independently audited and accepted this bounded scope.


## Published import refactor

The accepted simulator ran the original frozen001 sources. The public host oracle
imports six general interval helpers from `interval_numerics.py`; their function
ASTs, numerical constants and dependency semantics are unchanged from the original
reference module. The original frozen files and hashes remain the execution record.
All ten complete source oracles/intervals and actual-stage validation records are
identical before/after extraction and to the frozen evidence. The isolated public
snapshot passes52host tests and imports without WP09 whole-chain code. No SDK
rerun is claimed for this host-only import refactor. See
[equivalence receipt](../evidence/wp10-interval-equivalence.json).
