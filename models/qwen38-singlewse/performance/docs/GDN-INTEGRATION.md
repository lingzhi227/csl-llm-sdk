# Recurrent integration after the original frontend

The acceptance target remains the complete original model replying to successive
prompts in one retained conversation at>=2000 average output tokens/s. A new
prompt appends positions to the existing state. Neither reset between turns nor
an aggregate rate across unrelated requests qualifies.

P43 physically qualified the original convolution/gate frontend, but supplied
its recurrent results from an independent reference. The missing edge is actual
device state update followed by the native return and gated output projection.

P46 physically qualifies all original state slices and paired returns with
injected independent frontend inputs; see [Paired GDN](PAIRED-GDN.md).
The actual frontend-to-core routes and full resident layer remain unconnected.

P45 now admits full original banks and753 actual private state-port programs in
complete027; reference003 preserves every original word. The remaining edge is
physical frontend broadcast, paired returns and downstream retirement. Use
[GDN-CO-PLACEMENT.md](GDN-CO-PLACEMENT.md) and the new address authority; the older
calibrations below explain prior decisions and are not the current placement.

## Measured cohost constraints

`layer-backend-compile-029` keeps one128-key FP32 state column's arithmetic
reachable in four actual cohost programs. It adds384–400 bytes with existing
decoder/partial scratch. This is a reduced-bank compiler calibration, without
transport or numerical execution. Attempt028 failed at layout export declarations
and is preserved.

Attempts030/031 measured replacing MLP SDK endpoints with internal cold-control
ports. The first adaptation accidentally retained inactive LUT buffers; explicit
compile-time segment guards repair that. Corrected ordinary workers save176B,
output senders save64B, and fused actors grow576B. This alternative is **not
selected** for the complete stage. The existing SDK and neural routes remain.

The selected capacity change is exact local sharing of original MLP FP32 scale
words. Attempt032 admits20 paired calibration PEs. It changes weight addresses,
not FP8 codes, arithmetic order, output ownership or communication. The full
candidate and bitwise source proof are recorded separately; selected compilation
does not establish their acceptance.

## State layout and buffer leases

The retained recurrent tensor contains48 heads x128 keys x128 values, all FP32.
Its old128-byte pages each contain4 keys x8 values. A future full-key worker
must explicitly retile this layout; consecutive old pages are not full columns.
For a logical `(head,key,value)`:

```
old_page = recurrent_page_start + head*512 + (key//4)*16 + value//8
old_word = (key%4)*8 + value%8
new_word = worker_base + key*worker_width + value-worker_first_value
```

Require a bijection over all786432 FP32 elements, declared per-PE bank extents,
and exact reset/replay. A worker may own a variable number of neighboring value
columns. Its routing, output format and measured code/scratch cost must be
included before capacity is claimed.

The512-byte projection decoder can hold K, then Q, only after its previous
projection transfer and math have drained. MLP cohosts must complete their
previous-token leases before recurrent ingress, and recurrent return sends must
retire before current-token MLP input may reuse that memory. Mixer cohosts also
require the local QKV/Z/A/B input and result leases to retire. Queue backpressure
must enforce these dependencies on the device; a host observation of producer
DMA completion is insufficient.

The current frontend wire format returns two BF16 values in a five-word frame.
Single-column workers therefore need a qualified pair gather or a scalar-return
frontend adaptation. Do not change only the expected packet count: duplicate
checks, whole-head completion, packet retirement and next-token admission must
remain consistent. Persistent state spans new prompts; only an explicit reset
after all stage leases drain may clear it.

## Next executable boundary

Use the actual full-stage SRAM census and original bank proof to place the state
workers. Connect the physically qualified16 frontends to actual GDN workers and
returns, then to original output projection and the retained norm/MLP graph.
Keep the full original parameters and an independent numerical oracle. Complete
layer timing precedes any model-rate claim; all64 stages and real token feedback
are still required for the continuing-dialogue target.
