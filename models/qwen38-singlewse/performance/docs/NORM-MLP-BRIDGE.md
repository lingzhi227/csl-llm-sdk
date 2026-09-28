# Residual/RMS integration around the original MLP

The P31 candidate connects the preceding residual add and full5120 RMSNorm to
the already qualified original5120->17408->5120 MLP, then adds the retained
residual and computes the next layer's input RMSNorm. No intermediate neural
values are uploaded between these operations. Original mixer/state operations
and the next layer's projection are not active yet; this is not a complete layer
or model. Physical008 passes the complete subgraph qualification below.

## Physical008 result

The actual WSE-3 run passes all four epochs: nonzero residual input, zero after
nonzero, changed input and warm replay. All20480 MLP output values and61440
preceding-norm/final-residual/successor-norm values match the frozen independent
reference bit for bit. Native operands/scales, all83 sender counters, every11388
PE audit, original396953216-byte bank retention, both original norm gains and
SiLU table retention, local drain and normal stop pass. No intermediate neural
tensor is supplied by the host between the fused operations.

A separate bounded CPU re-read of the retained device captures checks all81920
BF16 values, all PE/endpoint counters and raw timestamps again after resource
release. It submits no hardware job. `actual.npz` is7072894 bytes, SHA256
`fb61c2cf3e61d55f7a46790dd7b8a89eea2157c24a66a7337551e914e3d52e9a`.
The first capture-audit script looked for the controller coordinate in the wrong
metadata file and failed before numerical checks; its failure is retained. The
corrected read-only audit uses the frozen fixture coordinate and passes.

| Epoch | Controller counts | Completed-output host wall |
| --- | ---: | ---: |
| Nonzero | 731230 | 10.811ms |
| Zero after nonzero | 695076 | 10.757ms |
| Changed nonzero | 731270 | 10.465ms |
| Warm replay | 731260 | 10.420ms |

Host wall includes arm/start/finish, phase logging and MLP plus successor-output
readback. These four diagnostic epochs are not sustained inference throughput.
The graph and fixture differ from P30, so these observations do not establish
a speedup or slowdown of an identical workload. Counts are not converted to TPS
using an assumed timer frequency. Each epoch retires176 native packets and40
residual chunks;9 return frames arrive during another packet's buffer lease.
This demonstrates protocol overlap, not a measured arithmetic-overlap percentage.

SDK loading430.237s, bank/setup initialization3.096s, normal stop10.070s and total
diagnostic460.322s are recorded separately. Both physical compiler/runtime jobs
succeeded and released. The final account snapshot contains164 terminal owned
jobs and no owned system assignments; provider-billed node hours are unavailable
in this API. All original functionality and the007 timeout remain preserved.

This milestone adds no GDN/attention execution, neighboring complete layer,
automatic request pipeline, sentence generation or full-model TPS result.
Connecting those operations and meeting the>=2000 aggregate generated-token/s
target remain active work.

## Concrete lowering

The original resident-stage schedule still owns every native matrix tile. The
network lowering assigns forty128-value RMS owners and forty adjacent quantizer/
bus endpoints in the layer00 mixer region. Each RMS owner retains its residual
and two original gain slices. Its normalized packet travels one cardinal hop
to its quantizer. The latter feeds the existing shared native-input distribution
and returns the final successor-normalized packet under an exclusive bus grant.

Local compensated square sums feed a fixed ordered40-owner reduction, followed
by a fixed inverse broadcast. These use distinct fixed-color queues. The two
RMS rounds reuse buffers only after the local preceding send retires. The first
normalized vector is captured on the quantizer for independent qualification.
All scalar rounding boundaries and original gain identities are explicit in the
independent reference; its full-width FP64 expression bounds the ordered FP32
reference before BF16 rounding.

The MLP controller forwards each available128-value down-output chunk to its
residual owner while later down frames can still arrive. It retains the existing
diagnostic MLP output array. Successor-output grants start only after every down
chunk is issued, avoiding a dependency cycle where an early normalization fetch
could prevent delivery of the remaining MLP outputs. Host arm/start/finish RPCs
remain in this diagnostic; automatic request admission is still required for the
serving path. No sustained speed claim follows from this source.

## Actual capacity failures and page placement

Compile010 accepts CSL syntax but rejects40 combined RMS/quantizer/bus PEs:
maximum51872 bytes including4096 stack,3744 over the48128 ceiling. Compile011
separates the programs onto adjacent PEs, reducing the maximum to49264 but
still rejecting80 PEs. Both attempts are preserved and released; neither ran on
a wafer. The memory limit or declared stack reserve was not relaxed.

Admitted compile012 moves only800 verified128-byte pages of `recurrent.0`
(102400 bytes) two rows north. RMS owners release12 tail pages each; quantizer
owners release8. Destination bank tails hold those pages, with every original
matrix/gain coordinate unchanged. `state-page-remap.json` records the logical
state offset and exact source/destination addresses for every page. The full
396953216-byte resident payload is proved to appear exactly once in both source
and destination address partitions. No state capacity is removed.

Future mixer execution **must resolve these remapped state pages**. The original
schedule alone is no longer the complete state-address definition for this
candidate. Compiler admission must cover the actual cohost programs and banks,
not just the original packing estimate. The remap is layer00-specific until the
other stage geometries receive their own admission.

`spatial/state_placement.py` now resolves the original logical GDN coordinates
through this map and checks every removed/appended bank tail. Tests enumerate
all49152 pages of both original requests: exactly800 locations change, every
4x8 FP32 page remains addressable, and the requests remain disjoint. Missing,
duplicate, foreign and overlapping moves are rejected. This is a reusable
address-lowering API for the future mixer, not evidence of GDN execution inside
the P31 graph; the physical008 image does not add a mixer kernel.

Compile012 covers all11388 PEs with4226 ELF images. Maximum48112 bytes
including4096 stack leaves16 bytes minimum margin; no PE exceeds the ceiling.
This is workstation compiler admission, not the later physical compiler result.
Reference002 completes in30.960s, pins the original two gain tensors and all
MLP weights, and independently conserves every resident word. Its frozen fixture
metadata hash is `d419833ab3b9fe1f468a2e645d0bfb3fa125e6e9804088ea46510d293503366a`.
All three compiles and the reference service are released. Reference002 retains
its actual `remap_banks` deployment directory due to a deployer variable-name
collision; the corrected tools follow the immutable dispatch identity, with no
unchanged rerun or directory rename.

Physical007 compiled and admitted all11388 PEs (4226 ELF images, maximum48128
bytes including4096 stack), but its runtime reached the900s deadline after
initialization without an epoch result. Its compiler succeeded, runtime was
cancelled and all owned physical assignments were released. Worker logs did not
expose a device assertion; the exact blocked host call was not instrumented.

Source inspection found a concrete command-ownership defect: adding33 separate
fusion endpoints moves the10 down-output endpoints from IDs40..49 to73..82,
but the collection grants retained the old IDs. Those requests select fusion
actors whose protocol rejects the output-sink command. This explains a possible
stall; the old run does not itself prove the observed device stopped there.
`layer-mlp-hw-007/diagnosis.json` binds all10 mismatches and the bounded remote
worker-log archive. The failure is preserved without numerical acceptance.

Physical008 generates collection IDs from the actual output endpoints and
audits each command's endpoint role and original output ownership. Mutations
reproduce the old reachable-but-wrong recipients and are rejected. Its sole CSL
change is the10 entries of the grant target table; matrix bytes, bank extents,
state remap, arithmetic, independent reference002 and numerical criteria match
007 exactly. Its fresh physical compiler admitted all11388 PEs at maximum48128
bytes including4096 stack and released its job. All148 source and publication
tests pass, including real-child timeout/cleanup and full relocated-state
address checks.

The new host driver records arm/start/finish/capture phases. Its supervisor
permits the existing900s image-loading/total limit, then rejects30s without
progress in any observed post-load phase. The supervisor terminates only its
own client and follows the retained correlated-job release checks. Initial
functional lifecycle source is unchanged. Norm/residual snapshots are collected
before numerical assertions, preserving upstream evidence if a later result
differs. Host wall timings explicitly include diagnostic phase logging.
Physical008 passes the subgraph checks above; no full-layer/model-speed
acceptance follows from this diagnostic.

## Reproduction and remaining qualification

`stage_layer_mlp_compile.py NNN --shared-inputs --norm-bridge` freezes the full
connected region and requests only a bounded workstation compile. The optional
mode leaves the historical complete-MLP interface available; physical006 remains
the accepted physical baseline.

After full SRAM admission, `stage_layer_mlp_reference.py NNN --norm-bridge
--compiled layer-mlp-compile-NNN` freezes original gains, both initial boundary
operands, complete ordered MLP results and both norm results. The reference
independently validates the state remap and preserves the original bank fixture.
`stage_layer_mlp_hw.py NNN --shared-inputs --norm-bridge --reference
layer-mlp-reference-NNN` stages a separate physical qualification; it does not
dispatch automatically.

Acceptance requires bit-exact preceding norm, complete MLP, final residual and
successor norm; native operand/scales and counters; all bank/gain/LUT retention;
zero/change/replay; every local lease drained and normal shutdown. Sources,
fixtures and outputs remain separately bound. The complete64-layer sentence
and>=2000 aggregate generated-token/s goal remains open.
