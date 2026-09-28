# Original mixer projections in the resident norm/MLP stage

This candidate adds arrival-driven CSL projections to the accepted P31 stage
programs. It preserves all original banks and the800-page state relocation.
It is not yet a complete mixer: convolution, Q/K normalization, gates, recurrent
state updates, gated output normalization and their automatic consumers remain
to be connected. No complete-layer execution or model throughput is claimed.

The five original layer0 matrices are QKV10240x5120, Z6144x5120, A48x5120,
B48x5120 and output5120x6144. QKV/Z/output keep original FP8 tiles and scales;
A/B use original BF16 weights and unquantized BF16 activation operands. The
lowering checks every454400 original tile address, all output rows and all7072
input-owner deliveries. It does not substitute a reduced matrix or model.

## Dataflow and buffer ownership

The original45x79 mixer rectangle remains two dimensional. Its serpentine
ordering makes consecutive reduction members physical neighbors. Colors10/11
connect each member to its predecessor; both40- and48-part contractions reuse
these fixed routes. Group boundaries are a compute/packet condition, with no
runtime route rewrite. This right-associated FP32 reduction order requires its
own numerical qualification; earlier MLP results do not qualify it.

Each root exposes a leased five-word result with generation, global first row,
valid row count, packed BF16 output and projection identity. The next consumer
must acknowledge that exact generation/row before reuse. There is no automatic
root acknowledgement or fictitious completion while the consumer is missing.
The root port is intended for local convolution/gate processing or direct
redistribution, not an obligatory controller or host gather.

The input wire contains three tagged header words and128 tagged BF16 values.
QKV and output receive on separate fixed colors16/17; exact K filters select
the original owners. Z/A/B reuse QKV's retained raw input in the same unique
token invocation. A reusable request-slot number is insufficient as the input
cache identity. Projection generation is separately monotonic.

Child reception, local partial and outgoing packet have separate buffers. After
starting a row send, the PE can calculate its next tile without overwriting the
leased outgoing packet. A child header checks generation, global row and exact
subtree K count before addition. These are executable overlap mechanisms, not
measured arithmetic-overlap or speedup results.

## Cofusion resources

The composed source includes actual P31 norm bridge and quantizer/sender roles,
not empty stand-ins. Norm and mixer entrypoints both reject overlapping phases
before arming shared arithmetic DSRs or microthreads. Input40 IQ7, input48 IQ5
(IQ4 on norm-sender PEs), child IQ6 and output OQ5 are distinct from live cohost
queues. Input/child/send DMA uses DSR2/3/6 and UT7/6/5; local math uses DSR4/7.

The installed SDK reserves task21 for D2H. Initial compile013 exposed that
collision and is preserved with its compiler failure and release receipt.
Source inspection now allocates mixer callbacks14..17 and root/drain25/26,
disjoint from cohost8..13,18/19 and SDK21..24,27..31. The exact inspected SDK
source hash and table are in `mixer-sdk-resources-001.json`.

## Qualification boundary

Python tests independently walk emitted input paths and reject wrong filters,
disconnected hops, duplicate routes, changed tile addresses/types/extents,
incorrect root ports and reserved task allocation. Composition tests preserve
the original matrix/state bank extents and P31 source modules.

Compile014 changes the colliding callback IDs and attempts the entire11388-PE
composition on the bounded workstation. Its immutable source contains the
original address/reduction audit. The subsequently strengthened input-path audit
does not change emitted CSL. Compile014 failed during linking with task-table/data memory overflow. Its
partial ELF census covers11273 of11388 PEs;3440 linked PEs exceed48128 bytes
including4096 stack, and115 have no final ELF. Common mixer-only cohosts use
50544 bytes; moved-state destinations reach52080. Missing images are not
accepted and are not all proven to be the particular linker-failing programs.
No full census or compile admission exists. Both services are released. The
152-test source suite passes; no WSE dispatch occurred in these compile runs.

This rejects the old code/scratch reservation for the new mixed program even
though unchanged P31 still passes its own admission. Do not shrink stack reserve
or remove original state to manufacture a pass. The next source change will
share input quantization at the producer, then revise actual program/buffer/state
placement against compiled footprints. Numerical and full-model gates remain.

Next connect the actual mixer consumers using original auxiliary weights and
`StatePlacement` for relocated recurrent pages, then integrate a complete layer
and adjacent stages. Preserve all64 resident layers, dependent output correctness
and the>=2000 aggregate completed generated tokens/s acceptance contract.
