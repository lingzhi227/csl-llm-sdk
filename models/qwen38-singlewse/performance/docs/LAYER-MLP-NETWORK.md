# Original MLP cross-kernel composition

The connected component consumes5120 BF16 boundary values, executes original
full-K gate/up projections, BF16 SiLU/multiply, group128 quantization and the
original complete down projection, and returns5120 BF16 values. All actual
native matrix and live auxiliary banks retain P23 addresses. The mixer/state
banks in the enclosing stage remain resident but inactive in this component.
This document describes implementation and admission; numerical/device execution
and complete-layer/model acceptance require separate frozen observations.

## Data path and ownership

Forty low-payload gate/up PEs quantize distinct128-value input groups. Thirty-three
also host the existing fusion actors. Their initial operand storage, private
outgoing frame and fused active/early-boundary storage are separate. Quantization
never consumes the cohosted native worker's512-byte decode arena.

A controller receives each spatially quantized frame and emits tagged32-value K
slices. A static hardware range filter selects the exact original input owner.
Full groups and the ragged final group use distinct selectors; explicit local
part indices retain original K ownership when fused groups arrive out of order.
The receiver removes tags inside the existing decode arena, then retains native
FP16 encodings and the original FP32 scale. There is no host intermediate GEMV,
activation, quantization or group scheduling.

Complete-K gate/up BF16 packets flow along P25 fixed-color routes into the actual
SiLU/multiply actor. A128-value group can leave as soon as it is quantized, while
other roots/groups continue. The separate early suffix buffer remains necessary
at root boundaries. Whole-group copying releases the local fusion buffer only
after its private fixed-OQ send; it does not fabricate a consumer credit.

The controller distributes each ready fused group as two native64-value down
slices, each with full/tail selectors. Ten low-payload down PEs receive their
respective full-K BF16 output segments. Each retains only540 or260 values, so
all roots may produce concurrently without per-row controller grants. Final
collection covers every original5120 output once.

## WSE-3 return-bus correction

The first composed attempt used simultaneous RAMP and cardinal RX under an
exclusive software grant. Real compiler002 rejects this: WSE-3 allows only one
input direction per color. That candidate is preserved and not admitted.

The current bus retains one static transit RX and a fixed output queue/color.
Only the granted source temporarily selects RAMP through the installed SDK's
`tile_config.color_config.reset_routes`. A new grant follows complete prior-frame
receipt at the controller. The local DMA callback requests `@queue_flush`; its
empty-queue handler restores the single transit RX and exits the flush state.
Changing RX on local DMA completion alone would strand data still in the OQ.
A delayed restoration can backpressure the next source until its handler runs.
It must not be interpreted as a second simultaneously enabled RX direction.

This mechanism follows the installed SDK2.10.1 WSE-3 color configuration and
queue-flush library, rather than WSE-2 color swaps. The global grant is an initial
integration mechanism with a likely service bottleneck. It does not establish
maximum inference speed. Further sharding/overlap must retain these ownership
conditions and be measured as complete stages and complete generated tokens.

## Local resources and completion

| Use | Queue | Microthread / DSR | Lifetime |
| --- | --- | --- | --- |
| Native tagged ingress | IQ2 | UT2 / DSR2 | Before native row loop |
| Ordered left/right reduction | IQ3/4 | UT3 / DSR3 | After own dot, before send |
| Native FP8 dot | memory | DSR4 | Original local K computation |
| Native output | OQ2; root prefix OQ3 | UT5 / DSR6 | Until private arena copied |
| Fusion main | IQ6 | UT7 / DSR7 | Actual projected block reception |
| Fusion early suffix | IQ7 | UT2 / DSR2 | Only after own input capture |
| Sender grant | IQ5 | UT6 / DSR1 | Filtered three-word control frame |
| Granted return | OQ3 | UT4 / DSR5 | Through OQ flush and RX restoration |
| Down segment sink | IQ6 | UT7 / DSR7 | Separate from native input/compute |

Bus senders are never contraction roots; output sinks are never fusion actors.
The emitted programs check every input/output color against its actual router
endpoint. Every static router has exactly one RX direction. All native reductions,
all tagged input owners, grant destinations and down output paths are walked.

Controller completion means all186 expected frames and all5120 output values
were received. It is not a global request/state-reset acknowledgment: local native
callbacks and final source flush handlers may still need to retire. Repeated
component execution must also check every local worker/sender is drained before
arming another epoch. Automatic successor/request ownership, residual/RMS, mixer,
attention, neighboring full layers, model feedback and measured sentences remain
separate unfinished integration requirements.

## Frozen admission results

`layer-mlp-compile-005` covers every PE in the78x146 GDN enclosing stage:
11388 PEs and4179 ELF images,236.604s compile-plus-census wall time.
`layer-mlp-compile-006` covers the78x141 attention enclosing stage:
10998 PEs and4894 images,227.499s. Both maximum low sections plus the4096-byte
stack reservation total47776bytes,352 below48128. These are offline compiler
measurements, not device inference latency.

Attempts001/003 preserve syntax/pointer failures. Attempt002 preserves the
WSE-3 multiple-RX rejection. Attempt004 hit its180s compiler bound with no
accepted ELF census. The full composition budget for005/006 is300s with a330s
service cap; both retain2GiB RAM, no swap,64 tasks,two pinned CPUs and the shared
heavy-job lock. All six services are released. Their retained remote footprint
is631864899bytes; no model payload or ELF image was copied to the Mac/GitHub.
No WSE job was added; the final account/system snapshot has no own allocation.

All115 Python source/protocol tests pass. The compact summary is
`evidence/layer-mlp-network-summary-001.json`. Neither successful compilation nor
these protocol models establish numerical correctness, liveness on hardware or
a full-model throughput improvement. The explicit next gate is independent
original-weight complete-MLP numerical qualification, followed by full layers.
