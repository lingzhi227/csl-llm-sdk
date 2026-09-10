# WP15 accepted original projected attention

CPU reference001 and the connected ten-PE SDK candidate are independently accepted.
The selected head consumes two consecutive original-hidden inputs in one request,
with all device operations and normal shutdown completed.

CPU sequence: dense/changed/last-column-onehot atpositions0/1/2 in request1,
then zero atposition0 in request2. All1024 selected original projection rows span
5120 columns. The official pinned projection/RMS/RoPE/eager/gate source has
transparent observation hooks and the expected FP32/BF16 boundaries. Original
projection-input hidden is not an embedding or completed upstream model layer.

The first CPU candidate passed in2.129004202s with208,433,152 peak bytes, under
2GiB MemoryMax,Swap0,60s and one thread. Its17 frozen source files and6 inputs stayed
unchanged. Source intervals were serialized before CPU projection observations.
The exact owned unit is absent,inactive,MainPID0,empty cgroup; the queue is quiet.
The CPU reference used one candidate; the later SDK execution also used one candidate.

| Case | Valid KV | Max absolute shift bound | Max output width | Width L2 / output L2 | Max BF16 span count |
|---|---:|---:|---:|---:|---:|
|dense|1|0|0.00006103515625|0.0332642|28512|
|changed|2|2.954101739|0.005714416504|2.2267183|30300|
|last column onehot|3|4.093750244|0.013458251953|1.1321130|29972|
|new-request zero|1|0|0|undefined for zero norm|1|

All shifts fit the existing exp[-24,0] domain. Dense/changed projection radii and
historical K/V uncertainty are preserved through trained normalization/rotation and
attention. The policy uses outward endpoint arithmetic, conservative FP32 reduction
error, every BF16 boundary, coupled max subtraction, exp/denominator/inverse,
uncertain cached V and gate. It discards most unknown correlations, so source
probability intervals may extend above1. This is a conservative enclosure, not an
observed probability or a tight guarantee. No empirical narrowing is applied.

| Token2 counterexample | Output coordinates outside source interval |
|---|---:|
|zero|144|
|cyclic dimension permutation|148|
|no sigmoid gate|93|
|current KV only / reset each token|115|
|wrong V from rawgate|163|
|negate current query|42|
|reverse two original hidden inputs and positions|44|
|ignore QK with uniform scores|0|
|reverse aligned cached K/V pairs|0|

Current-only and per-token reset are the same counterexample. Joint K/V pair
permutation is mathematically attention-invariant, so physical cache order is checked
through exact bytes and metadata. The token2 score enclosures overlap; only changing
the denominator bound cannot guarantee detection of uniform attention. Conditional
actual-operand stages remain essential. Both zero-dot ignore-QK witnesses are
rejected by actual CPU Q/K dot checks; actual device witnesses are separately
required if SDK execution is later admitted. See [protocol](WP15-PROTOCOL.md).

The control task independently checked22,508 intervals,18,996 observed source-stage
values,4,622 exact BF16 casts,1,536 unchanged tail values and3,584 exact CPU cached
values. All1,024 final CPU BF16 values matched the source nominal. This accepts only
the frozen CPU reference and its stated sensitivity limits, not the device chain.


## Accepted connected SDK execution

Eight producers cover all1024 selected Q/rawgate/K/V rows over5120 columns. PE8
performs trained Q/K normalization and partialRoPE; PE9 consumes the transferred
operands and retains cache8. Dense and changed original module inputs run at
positions0/1,request1,contractions1/2. All2048 projection,1024 Q/K and512 final
attention BF16 values match official CPU results. The input boundary remains the
projection-input hidden vector; upstream layer operations remain
separate integration work.

Compile8.330233700s,peak464,474,112bytes. Simulation1187.187747108s,
peak280,358,912bytes,normalstop,12.812252892seconds below the1200second hard limit.
All10 actual ordinary ELF ends plus4096 stack allowance passed49152:
44944,44928,44944,44944,44944,44944,44944,44944,28992,33280bytes.
Both exact owned units are absent/inactive withMainPID0 and empty cgroups.
The77 frozen source and37 compiled file identities remain unchanged.

Independent audit checked6144 FP32 projection values,2048 projection casts,
5127 actual-operand conditional stage values,9714 original-hidden source values,
3465 consumer casts,96 wire endpoint frames,20528 partial destination words,
20490 full physical cache words,229408 resident-weight words and4160 retained
producer output words. All16 retention summaries,736 timing pairs,3201 physical
operations(3050copies/151launches) and6409 journal records are complete.
Only command-first joins were observed in24 receiver transfers. Token1position0
exercised128 exact zero-product signs; zero sums were not observed.

Actual device Q/K conditional checks reject both token2 ignore-QK zero dots with
the centers and bounds described in the protocol. This complements the broad
original-source enclosure and does not reduce its radii. New requests,overflow,
malformed-command recovery,additional connected tokens,all heads/GQA/output
projection,complete layers/models/generation and physical systems remain separate.

The original device archive SHA256 is
`a445216aca164ab1e69fd4886f138876517c566d6f1398643889cb6071becfaa`.
Its public evidence, accepted CPU observations and source intervals retain884 typed
arrays losslessly.37 complete learned-parameter or direct static-transform arrays
are represented by dtype/shape/hash only. The6409 journal records are split into
three bounded files whose exact byte concatenation retains original SHA256
`19d4909fd2996c0e15240f32b91d7ccc05fbac1f4f70bc5a7a8169b1fe2def18`.
See [evidence](../evidence/wp15.json) and [journal index](../evidence/wp15-journal-index.json).

Host output D2H durations total724.725889794seconds and complete weight D2H
281.276801990seconds; both include queued device computation and do not isolate
transfer or kernel speed. The small hard-deadline margin argues for a reviewed
bounded diagnostic before increasing model workload.
