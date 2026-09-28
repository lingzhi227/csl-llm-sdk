# Residual/RMS integration around the original MLP

The P31 candidate connects the preceding residual add and full5120 RMSNorm to
the already qualified original5120->17408->5120 MLP, then adds the retained
residual and computes the next layer's input RMSNorm. No intermediate neural
values are uploaded between these operations. Original mixer/state operations
and the next layer's projection are not active yet; this is not a complete layer
or model. Numerical/device execution is pending until separately recorded.

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

Physical007 is dispatched for separate compilation and numerical qualification;
its outcome is pending. No physical norm-chain acceptance or speed result is
claimed yet. The complete139-test source suite passes.

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
