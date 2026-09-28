# Backpressured gateway streams and shared operand admission

P38 qualifies the backpressured inline gateway, arrival-triggered shared operand
preparer and device completion event. Both source and publication regressions
pass170 tests. Full-stage compiler admission and selected original-bank execution
pass; all nine workstation invocations are released, with no new WSE job.
This work improves initialization and diagnostic communication and connects real
input preparation. It does not execute a complete neural stage or prove model TPS.

## One owned frame, bounded continuous input

The gateway's existing273-word frame and148-word packet arenas remain shared
under explicit ownership. IQ4/C14 accepts four header words: selector, body word
count, response wire word count and global sequence. Selector bit31 places the
body immediately after the header. Clear bit31 preserves P37's preloaded-frame
ABI. Selector bit30 requests a batch completion event. The remaining bits name an
internal PE, or the reserved local operand producer. An internal command has its
own per-PE sequence and the existing validated segment/opcode ABI.

The receiver blocks IQ4 while a command owns the frame. That backpressures the
remaining SDK stream until the complete response has returned and its header
has passed sequence/type/count checks. Multiple initialization writes can share
one bounded SDK transfer. Only a batch's final command may retain a read reply;
the host client rejects another submission until that reply is copied. The
client stages at most262144 words; the full-bank probe uses16384-word batches.
No device buffer grows with batch size.

A reserved local target accepts67 words: invocation, K-group, family40/48, and
64 original packed BF16 pairs. Arrival invokes the actual shared original
BF16/FP8 preparer, which emits133 tagged words on C16/C17. Routed builds remove
the prototype `mixer_send_operand` SDK RPC export. Its non-routed legacy wrapper
remains available for preserved source composition. A producer completion means
its DMA retired, not that downstream projections or the neural stage drained.

## Completion is a device event

Bulk symbol polling020/021 timed out at the penultimate completion. Stopped
core022 nevertheless contains all61 completed commands and drained internal
ports;023 reproduces the same behavior even with50ms polling backoff. This
rejects backoff as a sufficient fix. The exact simulator/SDK scheduling cause
is not established; neither failed polling run is accepted.

The device now returns the batch's four-word audit record on SDK C15. This uses
OQ7, which is otherwise the C12 command output. A queue-empty handler waits for
command data to drain, changes that queue's color, emits the event, waits for the
event to drain, restores C12 and finally reopens IQ4. It uses the installed SDK's
`tile_config.output_queue_config.encode_output_queue` and shared teardown/flush
registration. No extra output queue or global tensor buffer is allocated.
Only the gateway installs the SDK C14/C15 streams; all other SDK roles keep
ordinary parameters so their neural routes remain intact. Both streaming color
IDs are registered as named compiler parameters.

The command/event path owns DSR4 and UT6/task12; command responses use IQ3,
DSR5 and UT7/task13. The operand sender borrows the idle controller's DSR3 and
uses OQ5/OQ6, UT5/task11. SDK memcpy retains its own queues and DSRs. Neural
admission is rejected while a partial command body or gateway operation owns
the arenas. No asynchronous user task blocks or unblocks SDK's command queue.

Diagnostic024 received the exact61-command event during runtime. Its original
post-stop inspection failed because the compiler named the DSD-addressed audit
array in the `$$csl_base_address$$` namespace. A separate, preserved post-read
resolves that exact symbol namespace, matches all nine source code sections to
the stopped core and verifies the completed/drained state. The original failure
is retained; it is not rewritten into a whole-bank acceptance result.

## Independent full projection reference

`layer-mixer-reference-001` reads the pinned original layer0 checkpoint directly,
without candidate tile packing or descriptors. It covers QKV10240x5120,
Z6144x5120, A/B48x5120 and output5120x6144. Four mixed/zero/changed/replay cases
produce86400 BF16 outputs. Each128-wide partial has explicit ordered FP32 fused
products, the original rounded activation/weight scales and the actual
right-to-left K reduction order. A separate FP64 sum and predeclared absolute
roundoff bound check the unrounded result before any candidate observation.
The largest nonzero relative bound norm is0.000470067, below the0.02 reference
quality ceiling. This is an independent CPU reference, not device execution.

Full contractions, root consumers, conv/gates/GDN state updates and adjacent
layer handoff remain open. The complete original64-layer sentence and2000
aggregate generated-token/s contract is unchanged.

## Complete bank and program admission

Full routed compile018 covers11388 PEs with7714 ELF images. Every application
coordinate and SRAM record passes the48128-byte limit, including4096 bytes of
declared stack. The maximum is48112. The gateway occupies42288 bytes, leaving
5840; internal mixer cohosts retain112-160 bytes of minimum margin. The gateway
changes add960 bytes to its full-layout compiled footprint. Profiles, joint
placement, stage geometry and mixer descriptors are byte-identical to017.

Reference003 is bound to018's exact source manifest and independently verifies
all396953216 original bank bytes,454400 mixed tiles,53661 auxiliary pages and
unchanged MLP prefixes. Its bank and index hashes equal reference002. This is
storage admission and original data preservation; it is not execution of all
3554 internal control endpoints or the complete projection contractions.

The gateway stream is a bounded initialization, diagnostic and boundary-input
entry. Its serialized return network must not become the serving schedule.
Projection roots retain their output lease until an explicit consumer retires
it. Connecting actual conv/gate/state consumers must preserve that ownership,
use the joint auxiliary-page map, and qualify complete original contractions
before counting any stage or model throughput. A producer DMA event cannot
release downstream request state.

## Accepted selected execution

Simulator025 completes584 commands and stops normally. It retains all60178
original32-bit words in eight full selected PE banks, exact16-bit descriptors,
high-half32-bit test patterns and three zero-work standby epochs. Sixteen
rounds of ordinary SDK transfers reach all three east-side sentinel PEs.
Eight local arrival submissions produce all133 packet words exactly against
an independent BF16/FP8 oracle, including both40/48 families and zero/change/replay.

The239 full-bank writes use four H2D stream calls with at most16384 words each.
P37's corresponding per-command ABI used two H2D calls per write (478 calls);
each new batch also receives one four-word device completion event. These are
API counts for cold loading, not inference rates or a measured hardware speedup.
Reverse-order full-bank reads validate every original word after all writes.
The529.355-second simulator host duration is explicitly not a physical latency.
Failures020-024 remain preserved alongside the accepted025 run.
