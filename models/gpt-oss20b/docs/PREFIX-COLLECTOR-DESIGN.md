# Implemented attention prefix collectors

The full attention prefix and its connection to the 32-expert MoE layer passed
physical CS3 validation (`attention-prefix-hw-001`, `decoder-layer-hw-002`).
This does not establish full 24-layer inference. See ../STATUS.md.

The four-column prefix puts the attention controller at local (2,0), eight KV
workers at (3,0..7), and the MoE controller at (3,1159). Column 0 has 38 QKV
chains with endpoints 29+30r. Column 1 has two QKV endpoints at 29 and 59,
23 O endpoints at 106+43r, and the router at y=1054..1083.

Collector column 2 uses 68 nodes: the union of y=0..7, all 38 column-0 QKV
endpoints, and all 23 O endpoints. QKV and O overlap only at y=149. The generated
layout uses a constant-time rank expression, verified against all 1160 y
coordinates, to avoid a large repeated CSL interpreter loop.

## Separate local inputs and explicit vertical relays

| Producer | Local reply color |
|---|---:|
| Column-0 QKV | 6 |
| Column-1 QKV | 8 |
| Column-1 O | 16 |
| KV worker | 11 |

Separate colors prevent illegal RAMP/WEST merges where two matrix banks share
an endpoint row. Consecutive nodes alternate northbound colors 4/5. Non-node
PEs forward SOUTH -> NORTH in hardware. Each node receives a local reply or a
reply from below, then explicitly sends it north using the other color. WSE-3
permits one physical RX direction per color.

The host's fixed phase/index command identifies the selected producer. There
is no additional color-9 request broadcast in this implementation. Nodes below
the selected target return immediately; its node arms its local input; nodes
above arm their vertical input. Exactly one producer sends in the phase, and
all forwarding buffers remain leased until TX completes.

The controller gathers 40 QKV tiles in phase 4, eight KV groups in phase 6,
and 23 O tiles in phase 8. QKV/O packets contain 64 u32 words (128 BF16 values);
KV replies contain 256 u32 words (512 BF16 values). KV group 0 reaches the root
directly on color 11; groups 1..7 use the explicit collector path.

## Activation and KV paths

QKV input uses color 3 west from the controller, then south through both matrix
columns. Per-PE counter filters select 96 FP32 values. O input uses color 7 and
4128 FP32 values: the original 4096 plus 32 zeros. Tile chains reduce partial
FP32 sums, add original bias at the final owner, then round to BF16.

Each phase-6 packet has two u32 header words (position and KV-head index),
512 BF16 queries, 64 BF16 keys and 64 BF16 values. All eight KV workers consume
each finite packet; only its selected worker updates cache and computes its
eight-query-head group with the learned sink. The controller waits for both
packet TX and output RX completion before advancing.

Attention phase 9 adds the original residual and sends the 2880 BF16 hidden
values south on color 17. The MoE controller receives during the same phase,
then normalizes and routes in phase 10. Waiting to receive until phase 10 would
risk preventing phase 9 from draining.

## Evidence limits

Both real-token attention outputs were BF16 exact, with persistent K/V. K cache
was exact; one V-cache value differed by 1 BF16 ULP without a final-output
difference. The integrated decoder likewise produced exact attention and MoE
outputs across both tokens, all 31,320 PE epochs, full weight retention, normal
stop and confirmed job release. These two positions do not validate maximum
context, arbitrary prompts, or 24-layer accumulated numerical error.
