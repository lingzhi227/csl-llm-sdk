# Shared preparation for original FP8/BF16 mixer projections

P33 moves original group128 quantization from every receiving weight PE into
one shared packet producer. The CSL producer and consumer are composed with
the actual P31 norm/MLP stage. Original matrix dimensions, every454400 tile
address, resident banks and all800 state relocations are unchanged. No complete
mixer numerical execution, full-stage SRAM admission or speedup is established.

## Wire format and arithmetic

Each body wavelet contains three independent fields:

| Bits | Meaning |
| --- | --- |
| 31..24 | Original K128 input group,0..39 or0..47 |
| 23..16 | Original finite E4M3 code from the shared group quantizer |
| 15..0 | Original unquantized BF16 input bits |

The WSE RAMP filter still examines the upper16 bits. A receiver for K group k
uses the exact interval[k*256,k*256+255], accepting every possible code byte and
no neighboring group. All emitted route intervals and paths are checked against
original K ownership; corruption of either interval boundary is rejected.

Five tagged header words carry the full32-bit token-invocation identity, complete
K family, and both16-bit halves of the original FP32 activation scale. The body
is128 words, so a full packet grows from131 to133 words, rather than duplicating
the wire for separate raw and quantized operands. A unique token invocation is
still distinct from a reusable request slot.

The producer calls the existing qualified `encoded_pair` after the same clamped
absmax and reciprocal-multiply scale calculation. Its native FP16 encodings
are losslessly converted back to8-bit E4M3 fields for transmission. The consumer
compacts the low halves into its BF16 cache and uses three integer vector
instructions to embed the code byte as FP16/256. Native dot/scaling and original
BF16 row arithmetic remain unchanged. Z/A/B reuse the same invocation's input.

The producer port borrows the idle P31 controller's existing frame and packet
arenas, with separate fixed output queues for the40/48 families. Its source
currently exposes a diagnostic entrypoint; automatic preceding-stage handoff
and a distributed serving schedule remain open. Serial diagnostic preparation
is not a high-throughput serving claim. Across the emitted graph, quantization
work is described once per88 logical groups instead of7072 receiver deliveries;
this is a static operation count, not a measured speed ratio.

Only PEs that can be a reduction root retain root-result rounding/control code.
Other PEs preserve the same fixed adjacent reduction and leased partial packets.
No root/consumer acknowledgement is forged to make the missing graph drain.

## Evidence and limits

Independent packet tests cover all254 finite E4M3 codes under all48 selectors,
all65536 original16-bit payload patterns, signed zero, extreme finite BF16,
changing input, exact scale bits and rejection of corrupted headers/body. The
full155-test source suite passes. These Python checks do not execute new CSL.

Full composition compile015 still fails linking with PE task-table/data memory
overflow. Its5978 partial ELF images cover11273 of11388 PEs. All3440 previously
over-limit mixer PEs remain over the unchanged48128-byte gate including4096
stack;115 PEs have no completed image. Per-PE comparisons and exact partial ELF
identities are retained in `mixer-shared-operand-footprint-001.json` and the
attempt directory. Unchanged MLP bodies, banks and failed014 remain preserved.

After015 was frozen, one further source guard rejects a producer command after
MLP arm but before start: `!active` alone does not prove that the controller is
idle. This additional assertion is covered by the current source test, but is
not part of015's compiled controller. No footprint/numerical acceptance is
inferred from015 for that later source edit. Compile015's missing controller image also
prevents a whole-stage SRAM saving claim.

Selected backend compile012 subsequently compiles the current guarded producer at
39632 bytes including4096 stack, with its actual full buffers and code. Eight
cohost samples use explicitly reduced4096-word banks to measure otherwise
unlinked program overhead. Their code/workspace/stack envelopes are14384–14608
bytes for projection-only roles,18688–19760 for norm bridge roles and
19248–19280 for norm sender roles. These calibration bank extents do not
represent the original full weights; extrapolated totals are labeled and cannot
pass full-bank or whole-stage admission. This short compile also released its
service and lock.

No WSE job was submitted. The bounded workstation service is terminal, with its
process group gone and shared lock released. P31 physical008 remains the latest
accepted numerical graph. Complete conv/gate/recurrent consumers, full layers,
the64-layer pipeline and>=2000 aggregate completed generated tokens/s remain
unfinished.

Next fuse the BF16 A/B input consumption with ingress where original local row
counts permit releasing the full raw cache, and account for all actual cohost
code/scratch/state lifetimes. Repartition or relocate original data only with a
complete conservation/address proof and a new full compiler gate. Do not shrink
stack reserve, drop request state or turn this partial byte saving into model
acceptance.
