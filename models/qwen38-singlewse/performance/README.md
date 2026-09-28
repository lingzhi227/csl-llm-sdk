# Qwen3.8 WSE-3 spatial performance work

Current target: complete original Qwen3.8-27B-FP8 on one physical WSE-3,
64 resident spatial stages, and **at least2000 average output tokens/s for one
stateful multi-turn conversation**. Each new prompt must receive a correct reply
based on the actual preceding conversation. The September28 clarification
supersedes aggregate independent-request throughput. The primary measurement
includes appended-input processing and generation, excluding user think time and
initial loading; decode-only speed and TTFT are reported separately. See
[DIALOGUE-ACCEPTANCE.md](docs/DIALOGUE-ACCEPTANCE.md). The target remains unmet.

P45 fits the actual recurrent-port programs and all original layer00 banks on
11,388 PEs/8,240 ELF images: maximum48,112B including4,096B stack, with16B minimum
margin. Joint MLP output placement reserves753 state workers holding all786,432
FP32 recurrent elements. Original-bank reference003 verifies all97,025,344 source
words,1,044,480 MLP tiles and7,085 nonrecurrent pages; banks occupy388,094,976B.
Compile026's five112B overflows are resolved by moving ten auxiliary pages.
Forty-five selected source regressions, seven targeted refinement checks and
46 final publication regressions pass. All seven workstation services release;
no new WSE job is submitted. Actual GDN broadcast/returns, changed numerical
execution, full-layer inference and multi-turn model speed remain unqualified.
See [Recurrent co-placement](docs/GDN-CO-PLACEMENT.md).

P44 admits exact original MLP scale sharing on all11,388 layer00 PEs with
8,428 ELF images, maximum48,032B including4,096B stack and minimum96B margin.
Bank savings are3,912,128B; the measured whole-stage SRAM reduction is3,362,352B.
Remote reference002 verifies every98,003,376 original source word, all1,044,480
MLP tiles and28,605 retained pages. All routes, arithmetic order, original weight
codes and both work slots remain. Compile023's timeout is preserved;024 passes
and all services release. No new WSE job is submitted. Actual GDN transport,
changed numerical execution, complete-layer inference and model speed remain
unqualified. See [Compact MLP](docs/COMPACT-MLP.md).

P43 physically qualifies all16 original frontend groups/48 heads over six
continuous positions and two reset-replay positions:148224 FP32 packet values,
49152 gated BF16 outputs,245760 exact history samples,259512 native packets and
all43104 retained parameter words pass. Shared Q/K and reset replay are exact.
This standalone32-PE test injects independent original projection values and
frozen recurrent results; actual GDN, a complete layer and model speed remain
unqualified.40 selected source and40 publication tests pass. Both cluster jobs
and all seven workstation attempts release normally. See
[Frontend numerical qualification](docs/FRONTEND-NUMERICS.md).

P42 admits the complete original layer00 banks with native packet fusion and
all16 shared-Q/K frontends: full compile022 covers11388PE/7919ELF, maximum48112
bytes including4096stack. Remote reference001 verifies all454400 mixer tiles,
28605 retained pages and every original source word;433136 identical scale
aliases save1732544bank bytes without requantization. All existing MLP/norm/control
routes remain. Joint row/network placement reduces merge actors1216->168;
this is not a measured speedup.34 selected source and34 publication tests pass. All six workstation
services release and no new WSE job is submitted. Numerical frontend execution,
recurrent connections and complete multi-turn model speed remain unfinished.
See [Compact frontend](docs/COMPACT-FRONTEND.md).

P41 implements direct fused projection packet output and16 shared-Q/K
convolution/gate frontend groups. All33 selected cohost programs compile; all16
consumer groups keep their complete original bank extents, with maximum47984bytes
including4096 stack. Routing checks cover all16480 projected values and1408 paths.
The new resource planner rejects the unchanged whole-stage placement:1275
estimated matrix-prefix conflicts require joint weight/communication placement,
not only state-page movement. New numerical execution, the recurrent core and
complete layer/model inference remain unfinished; no speed claim is made. All
six workstation services release and no new hardware job is submitted. See
[Frontend fusion](docs/FRONTEND-FUSION.md).

P40 specializes the complete layer00 banks for one continuing conversation,
retaining all original weights, the complete first context and both work slots.
Fresh compile021 admits
all11388 PEs/7910 ELF images, maximum48112bytes including4096 stack. Actual SRAM
falls by3207168bytes; remote reference001 independently verifies every retained
word in393746048bytes. A turn-admission oracle checks exact history, pending final
tokens, three continuing turns, reset/overflow and all-stage retirement. Twenty
selected source and20 publication tests pass; these are protocol/storage checks,
not device dialogue or speed results. Two failed/cancelled compiler attempts are preserved. All four
workstation services release and no hardware job is submitted. See
[Single-dialogue banks](docs/SINGLE-DIALOGUE-BANKS.md).

P39 passes176 source and176 publication tests. Physical `layer-mixer-hw-002`
executes all five complete original QKV/Z/A/B/output projections through3554
internal endpoints. All86400 stored BF16 outputs are bit-exact to the predeclared
independent reference, with zero/change/replay, all counters and20 warm drains.
Every396953216 original bank byte and every descriptor is retained; normal stop
and release are confirmed. Physical compilation covers11388 PEs/7714 images,
maximum48128bytes including4096 stack. Download admission is now scoped to the
measured102028613-byte complete-mixer artifact. Four cluster jobs (one actual
WSE runtime) and both workstation services are released; failures are preserved.
Host root read/consume and full retention timings are diagnostic, not model TPS.
Actual conv/GDN consumers, complete neural stages and the multi-turn target remain
unfinished. See [COMPLETE-MIXER-QUALIFICATION.md](docs/COMPLETE-MIXER-QUALIFICATION.md).

P38 passes170 source and170 publication regression tests. Full routed compile018
admits all11388 PEs with7714 ELF images, maximum48112bytes including4096 stack;
reference003 verifies the unchanged396953216 original bank bytes. Simulator025
passes584 commands, exact retention of60178 words across eight complete original
PE banks, eight arrival-triggered operand cases and normal stop. Backpressured
inline commands coalesce239 bank writes into four bounded H2D streams; C15
completion events replace bulk symbol polling while sharing a drained output
queue. This is initialization/diagnostic API reduction, not measured model TPS.
An independent original full-matrix reference covers86400 BF16 outputs before
candidate projection observations. All nine workstation services released,
failures020-024 retained; no new WSE job or owned hardware allocation. Complete
projection contractions, root neural consumers and the full model target remain
open. See [STREAMED-GATEWAY.md](docs/STREAMED-GATEWAY.md).

P37 passes164 source and164 publication regression tests. Full routed compile017
covers11388 PEs in7714 ELF images, at maximum48112bytes including4096 stack;
reference002 preserves every396953216 original bank byte. Simulator019 passes
572 gateway commands and exact retention of60178 words in eight complete
original selected PE banks, with19 rounds of east-side SDK transfers and normal
stop. Internal control routes use addressed arrival, an exclusive return credit,
queue-drain restoration and explicit SDK teardown rearming. All21 workstation
services released; failures/cancellation preserved, no new WSE job. The full
3554-endpoint route is compiled but not executed as a whole; complete neural
stage and model-speed acceptance remain open. See
[ROUTED-DEVICE-CONTROL.md](docs/ROUTED-DEVICE-CONTROL.md).

P36 passes160 source and160 publication regression tests and admits the complete original layer00
mixer-projection/norm/MLP bank candidate in local compilation: all11388 PEs,
5891 ELF images, maximum48112bytes including4096 stack. Full-K QKV group row
counts and all auxiliary locations are jointly lowered; every original tile,
state page and396953216-byte bank capacity remains. The independent remote
relocation verifies all454400 mixer tiles through the actual CSL descriptors,
all53661 auxiliary pages and unchanged MLP prefixes. Both workstation services
released; no new WSE job. Gateway control routing, C22 forwarding execution,
complete neural stage and full-model speed remain unqualified. See
[JOINT-BANK-PLACEMENT.md](docs/JOINT-BANK-PLACEMENT.md).

P35 passes156 source and156 publication regression tests and adds credited internal-PE fabric control with real bank/config loading,
readback and preserved cohost entrypoints. Simulator005 passes122 gateway-only
host commands, retains the complete selected8130-word original bank and stops
normally. Its gateway/internal PE occupy11024/45920bytes including4096 stack.
Final selected resource compile017 saves640–864bytes on eight matched weight
cohosts with reduced calibration banks. Complete-stage routing/SRAM and neural
execution remain unqualified; norm cohost placement deficits remain. All eight
workstation services released, failures preserved, no new WSE job. See
[DEVICE-CONTROL.md](docs/DEVICE-CONTROL.md).

P34 passes155 source and155 publication regression tests and fuses original A/B contractions into
tagged operand arrival, retaining two
FP32 partials instead of a256-byte raw-input cache. Actual CSL simulator001
passes24 original-weight cases against the frozen P33 consumer and an independent
reference, including zero/change/replay. Selected resource compile014 saves112
bytes on each of eight matched weight cohosts; the producer remains39632bytes
including4096 stack. Failed013 is preserved. Full-bank/stage SRAM and complete
mixer/model execution are not admitted. Matrix-only resource estimates still
exceed the gate on root/norm cohosts, so subsequent work must revise matrix/role
placement together with state and workspace lifetimes. All three workstation
services released; no new WSE job. See [INGRESS-MIXER-FUSION.md](docs/INGRESS-MIXER-FUSION.md).

P33 shares original BF16/E4M3 preparation in one133-word input packet and
specializes root code. All original dimensions/banks remain;155 source and publication tests
pass. Matched3440 linked PEs save720–976bytes each (3290528bytes combined),
but every one remains above the unchanged48128-byte/4096-stack gate. Full
compile015 fails and has115 missing PEs. Selected backend compile012 admits the actual
39632-byte producer and calibrates cohost programs with explicitly reduced
4096-word banks; this is not full-bank/stage admission. Both services released;
no WSE job added. Automatic stage handoff, mixer numeric execution and the full
model speed goal remain open. See [PACKED-MIXER-OPERANDS.md](docs/PACKED-MIXER-OPERANDS.md).

P32 adds original mixed-projection CSL/dataflow source to the P31 cohost stage.
All454400 original matrix tile addresses and7072 input-owner deliveries are
checked;152 source and publication tests pass. No new WSE job was submitted.
Compile013 failed
on SDK task21; corrected014 reached linking but ran out of PE memory. Its partial
ELF census covers11273PEs, of which3440 exceed48128 bytes including4096 stack;
115PEs have no final ELF. Both workstation services are released. This candidate
is rejected for SRAM admission and has no mixer numerical/speed acceptance.
Next reduce duplicated operand preparation and revise cohost code/state placement,
then connect actual conv/gate/state consumers and complete stages. See
[MIXER-PROJECTION-COMPOSITION.md](docs/MIXER-PROJECTION-COMPOSITION.md).

P31 qualifies the physical residual/RMS -> complete original layer0 MLP ->
residual/next-layer RMS graph. All81920 checked BF16 values are bit-exact across
four epochs, including zero/change/replay. Native inputs/scales, all11388 PE
and83 endpoint counters, original banks/gains/LUT retention, local drain and
normal stop pass. The host supplies only the two initial boundary operands.
Forty norm owners communicate directly with forty adjacent quantizer endpoints;
down-output chunks feed residual owners while later returns can arrive.

Physical008 corrects ten obsolete output-grant IDs exposed by007's preserved
post-initialization timeout. Its sole CSL change is the grant target table;
weights, bank placement and independent reference002 remain unchanged. All148
source/publication tests pass; a bounded post-release capture re-read checks
all81920 values and counters again. Both jobs succeeded and released, with no
owned hardware assignment. All507 baseline files remain preserved except the
two explicitly maintained active status/acceptance notices.

Controller counts are695076–731270. Four completed-output host observations
are10.420–10.811ms, including phase logging and readback; this is not sustained
model throughput or a same-workload comparison with P30. Loading430.237s and
initialization3.096s are separate. Physical SRAM admission covers4226 ELF images
and11388 PEs, at maximum48128 bytes including4096 stack.

The800-page state relocation retains original capacity. Its new logical-state
address resolver is checked over all49152 pages of two requests, ready for future
mixer lowering; no GDN execution was added to this physical graph. Complete
mixer/attention layers, automatic stage handoff and full64-layer correct sentences
at>=2000 aggregate generated tokens/s remain unqualified. See [NORM-MLP-BRIDGE.md](docs/NORM-MLP-BRIDGE.md).

P30 remains the accepted physical MLP baseline described below; its resource
release and timings refer to that completed attempt.

P30 qualifies shared full/tail native input on physical006. One tagged wire
copy now serves both original owner classes through exact/range RAMP filters.
Native broadcast words halve from49,376 to24,688;176 group packets and all
343,040 delivered operand halfwords remain. Original arithmetic and banks are
unchanged. All20,480 BF16 outputs, native inputs/scales, counters, bank/table
retention, all-PE drain and normal stop pass. Mixer/state operations are inactive.

Nonzero complete-MLP controller counts fall from613,127–613,190 to564,289–564,333,
7.965–7.972% fewer. Four completed-output host observations are2.551–2.834ms,
shorter than P29 but still longer than P28. These four diagnostic calls do not
establish a sustained wall-speed trend or model TPS. Each epoch grants176 next
responses during packet leases, but no response arrives before that lease retires.

All134 source and publication tests pass. Local full-stage compile009 has
48,112-byte maximum footprint; the actual physical compiler reaches48,128
including4,096 stack, leaving zero minimum margin. All11,388 PE gates pass.
SDK loading418.693s, initialization3.003s and full diagnostic440.328s are separate.
Both hardware jobs and all five workstation services are released, with no owned
active/assigned hardware and the workstation shared lock free.

Two native-loop alternatives are preserved: paired input is slightly slower on
the used simulator shapes; unrolling improves local counts but exceeds SRAM
on68 PEs in the complete fused layout. Neither receives a hardware trial.
Next work integrates residual/RMS, device-triggered epoch handoff and adjacent
complete layers. The full64-layer correct-sentence and>=2000 aggregate completed
generated-token/s goal remains open, including sufficient request-state capacity.

See [SHARED-NATIVE-INPUTS.md](docs/SHARED-NATIVE-INPUTS.md),
[LAYER-MLP-QUALIFICATION.md](docs/LAYER-MLP-QUALIFICATION.md) and
[NEXT-STEPS.md](docs/NEXT-STEPS.md).

The qualified shared-input lowering is selected with `--shared-inputs` on the
complete-MLP compile and physical staging tools; use a fresh attempt identity.

Generate a fresh candidate with
`python3 performance/tools/plan_pipeline.py --output <new-directory>`.
Lower its preserved resident rectangles to native-loop ownership with
`python3 performance/tools/plan_layer_schedule.py --output <new-directory>`
(currently bound to frozen stage-map002 and qualified tile calibration).
Run checks with `python3 -m unittest discover -s performance/tests`.

The functional baseline commit6c2f4f5685478ee100f167e42a9f7f22f57dca35, original
inference source, captures and failed strict numerical evidence remain preserved.
Only active status/acceptance notices are updated outside the performance subtree.
The chronology below retains historical component scopes. References there to an
old target or next step are historical; they do not override the current contract.
P21 full compile003 was subsequently cancelled by its verified owner and released;
its pending milestone snapshot is retained and no old-layout WSE trial followed.

## Historical component milestones

P12 now provides an independently audited complete physical ownership and value
lifetime atlas: all1251 tensors,498 matrices,64 layers and105,052,160 real matrix
tiles are covered. Co-resident FP8/GDN state, distributed KV and value actors have
explicit locations and data budgets. This is metadata, not compiled routes/SRAM
or model execution. See [COMPLETE-MODEL-ATLAS.md](docs/COMPLETE-MODEL-ATLAS.md).

The first executable slice is a compiler-generated 2D multicast / acknowledgement
microbenchmark, now qualified on 256 physical WSE-3 PEs. It establishes explicit routes, queues, event ownership and
same-PE cycle timing before adding model computation. It is not model inference.

Run the source checks with `python3 -m unittest discover -s performance/tests`.
Run `python3 performance/tools/build_mesh.py --output <new-directory>` to lower
a checked mesh plan into CSL. See `docs/MILESTONES.md`, `docs/ARCHITECTURE.md` and `docs/MEASUREMENT.md`.

P3 qualifies packed-vector FP8 decoding and original-weight 2x128 native dot
products on real WSE-3: 1,286 cycles versus 11,316 for the unchanged scalar tile,
with exact matched outputs for the frozen cases. This is a local operator speedup;
complete model performance is still unqualified. `spatial/banks.py` supplies a
compact all-matrix ownership/storage candidate with explicit unproven placement
and SRAM gates. Run `performance/tools/plan_banks.py --output <fresh-json>` to
reproduce the estimate without loading checkpoint payloads.

P4 compiles and simulates the largest candidate resident bank at 46,768 bytes
including its declared stack; all112 FP8 slots and retained buffers pass. Prefetch
and compute are measured separately. There is no network overlap or whole-model
SRAM admission yet; see MILESTONES.md for the unimplemented paths and timing scope.

P5 measures a complete8x8 regional dataflow component on physical WSE-3:2,599 to
2,188 cycles with matched overlapping scheduling (15.8% lower latency), including
encoding, multicast, local dots, ordered sums and completion. All four paired
original-weight fixtures pass exactly. This is a partial projection component;
full-model2,000 tokens/s remains unmet. See `docs/REGIONAL-GEMV.md` and the frozen
`regional-gemv-hw-001` evidence for what is and is not included.

P6 qualifies the direct encoder on53,725 physical inputs and complete dynamic
quantization on44 groups, with identical bytes/scales and about2.045x reduction
in summed group cycles. P7 composes resident FP8 banks and regional communication
in the simulator:24 epochs and full storage readback pass, with48,080 bytes maximum
SRAM including stack. Its48-byte margin leaves full scheduler/BF16 integration
unqualified. Both results retain their exact tested scope; the original complete
model and2,000 tokens/s acceptance target are unchanged.

P8 qualifies a32-PE spatial group128 producer followed directly by one original
2x128 FP8 consumer on physical WSE-3. All44 frozen groups pass exact code/scale,
subtree-packet and independent arithmetic checks. Producer median2,484 cycles,
combined median3,003 cycles; host arming/readiness and one-time weight predecode
are separate. Both jobs completed normally and released. See
`docs/SPATIAL-QUANTIZATION.md`; this is a component, not complete model throughput.

P9 qualifies complete40/48/136-block K contractions on224 physical PEs in an8x28
mesh, using original projection weights. A static binary tree reduces the longest
case from14,389 to2,274–2,275 cycles (about6.33x for this component). All four cases
and two separately checked summation orders pass, including intermediate results
and retention. Only two output rows per matrix are executed. The failed initial
simulator and artifact-loading attempt are preserved; the successful run reuses
the compiled artifact and releases normally. See `docs/FULL-K-CONTRACTION.md`.

P10 qualifies actual BF16 computation beside FP8 in full112/12 resident banks on
six physical PEs. All12 switching/replay epochs and complete bank retention pass;
compiled footprint47,472 bytes including stack leaves656 bytes. The explicit
4-byte typed-slot control replaces unused per-tile metadata. Root component cycles
are1,252 FP8 and1,213 BF16, with different original operands; this is not a precision
speedup or model rate. See `docs/MIXED-BANK.md`. Device-controlled epochs, full-model
physical ownership/routes and the2,000 tokens/s target remain unfinished.

P11 runs two96-epoch resident sequences entirely under device control on physical
WSE-3. Request arrival triggers arithmetic; tagged reductions and send-completion
credits advance the loop. All4,608 local/subtree checks and full retention pass,
with47,792-byte maximum SRAM including stack. Complete loops average about1,765
controller cycles/epoch, including multicast, compute, reduction, return and gaps.
Neural inputs are independent preloaded fixtures; this is not autoregressive model
throughput. See `docs/RESIDENT-EPOCHS.md`. Full-model ownership/routes and2,000
dependent tokens/s remain unfinished.


P13 qualifies the complete128x128 FP32 recurrent state on16 physical cohostPE,
each retaining111 original FP8 tiles.96 dependent updates plus8 reset replay,
13,312 output values, complete terminal states, native FP8 dot checks and full
weight retention pass. Physical component latency is6,755–6,758cycles; actual
maxSRAM46,784bytes includes4KiB stack. Both jobs release normally. This does not
include preprocessing or full-model/token feedback; see
[COHOST-RECURRENCE.md](docs/COHOST-RECURRENCE.md).

P14 lowers all 498 matrices to 306 shared-input bundles and 980 events, adding
18 memory completion edges to protect every reused arena range. An independent
audit checks all 44,766,384 entries of four complete abstract reduction forests.
A separate physical 20-PE experiment passes 72 autonomous route/color/queue
epochs with all 705,120 bytes of synthetic bank storage retained. Same-endpoint
dependent send/return latency is 1,132–1,141 cycles; this is not model throughput.
The component's 48,112-byte SRAM footprint leaves only 16 bytes, so combined
neural-kernel admission remains open. See [PROJECTION-LOWERING.md](docs/PROJECTION-LOWERING.md)
and [ROUTE-WORD-QUALIFICATION.md](docs/ROUTE-WORD-QUALIFICATION.md).

P15 adds a complete paired-column input/storage overlay: all 498 original
matrices and 105,052,160 tiles pass independent address/route audits while the
maximum bank payload remains 35,256 bytes. A two-stage hardware counter window
feeds original FP8/BF16 native dots in representative maximum resident banks;
all ten physical window/precision/replay cases and complete original-bank
retention pass, with 47,328-byte SRAM including the declared stack. Both jobs
succeed and release normally.
See [COLUMNAR-INPUT.md](docs/COLUMNAR-INPUT.md) for the tested scope and remaining
whole-matrix integration. The full-model speed target remains unmet.

P16 binds all original projections and four complete forests to those new
storage classes, independently checking 44,718,648 PE/color entries. A physical
20-PE component now composes input filtering, original FP8/BF16 native dots,
ordered reduction and controller return. All ten cases and complete bank
retention pass; the same-controller interval is 1,685–2,499 cycles. An exact
five-operation FP8 decoder passes all 65,536 packed patterns on hardware and
measures 1.920x faster than the original decoder in matched local intervals.
Compiled SRAM including stack is 48,032 bytes, leaving 96 bytes. Both hardware
jobs and workstation services release. See [COUPLED-BANK.md](docs/COUPLED-BANK.md).
This is partial-K component execution; complete matrices, all-layer feedback and
the 2,000 dependent tokens/s target remain unqualified.

P17 qualifies four equal-work native shapes and matched scalar/vector output
scaling on eight physical WSE-3 PEs. The same 16x16 FP8 tile improves 2.088x
locally, or 1.612x including per-call weight decode, with exactly matched outputs
and all original banks retained. See [NATIVE-SHAPES.md](docs/NATIVE-SHAPES.md).
Separately, the complete original48x5120 BF16 matrix passes five full-matrix
calls plus an FP8 cohost smoke case, exact replay, all original bank retention
and normal stop at its atlas
coordinates. BF16 input-to-full-result intervals are10,824physicalcycles. The
earlier two-channel host-transport stall and successful one-channel qualification
are preserved in [FULL-MATRIX.md](docs/FULL-MATRIX.md).
These are component results. Complete dependent model inference at2,000tokens/s
remains unmet.


P18 moves the complete original48x5120 matrix to40x24 compute PEs with40 parallel
input owners, using a shared stream/completion/resource representation and one
CSL epoch protocol. Physical all-case/replay/full-bank validation passes at3898
cycles versus10824 in P17:2.77681x, including complete communication and output
delivery. The weights/inputs/oracle are identical; input storage is distributed
instead of concentrated at one controller. The first physical attempt's warm
output-order failure is preserved and fixed by resetting the gather queue binding
before the next kick. Actual SRAM plus stack is48112bytes, leaving16bytes.
The inferior8x32 strip simulation is retained without its proposed physical trial.
See [COMPACT-PROJECTION.md](docs/COMPACT-PROJECTION.md) and
[RETILED-MATRIX.md](docs/RETILED-MATRIX.md). The next acceptance boundary is a real
multi-operator MLP graph with direct consumers and cross-operator lifetimes;
its semantic importer is not executable. Complete-model2000tokens/s remains unmet.

P19 lowers a full-dimensional MLP ownership/fusion candidate with136 group128
regions and40 independent down-output stripes. All1,044,480 matrix tiles and
702,880 known stream paths pass a bounded independent audit. Fused numeric
packet helpers pass1350 simulator words under the existing quantization
convention; literal-division differences and failures are retained. Complete
subgraph runtime and bank/actor SRAM are not admitted, and no new hardware
performance result is claimed. See [MLP-SPATIAL-FUSION.md](docs/MLP-SPATIAL-FUSION.md).

P20 assigns all1251 original text tensors and persistent short-context state to
an MLP-anchored residency candidate, including5800 embedding-only communication
helpers. The independent address/capacity audit passes; combined all-role SRAM
and full execution remain unqualified. Two actual helper roles then pass physical
three-epoch input relay, credited8-word ordered joins, embedding lookups and full
original-bank retention. Their maximum SRAM including4KiB stack is46128bytes.
The import-time host failure is retained; its successful artifact is reused for
the corrected run. Both submitted jobs release normally and69 source tests pass.
See [MLP-RESIDENCY-CHUNKS.md](docs/MLP-RESIDENCY-CHUNKS.md). No complete MLP timing
or new model token rate is claimed; the mainline remains a real complete MLP and
then complete dependent model inference at>=2000tokens/s.

P21 adds static paths for the connected complete-dimensional MLP and CSL for
native gate/up/down, arrival-triggered activation/quantization, credited joins,
preceding residual and actual successor RMS. The independent audit admits1059920
listed data flows without static color aliases. Selected25-role compilation
passes with47776 maximum bytes including stack. Full750x1160 layout source is
generated; whole-wafer compile003 remains pending after the preserved600s timeout
of002. No integrated numerical execution, latency or model rate is claimed.
See [MLP-STATIC-PIPELINE.md](docs/MLP-STATIC-PIPELINE.md). No new WSE job was submitted.
