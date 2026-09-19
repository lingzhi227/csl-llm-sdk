# Bounded observation schema

All arrays have three PE rows in x0/origin, x1/READY, x2/responder order.
State is64 u32/PE; initialized cache is[1,x] followed by62 zero words.
Snapshot only copies observations; it neither repairs nor resumes transport.

| State index | Actual field |
|---|---|
|0..3|compute mode, column, TX completions, RX deliveries|
|4|group0 enqueue/start causal marker|
|5|harness error count|
|6|group1 enqueue/start causal marker|
|7..11|first issued, first completed, notice witness, notice count, bad notices|
|12..19|RX phase, TX phase, sending, pending, receiving, header waiting, protocol error, control scratch|
|20..24|last actual header, length, ordinary tail completions, native consumptions, ordinary data entries|
|25..30|header/body/native-tail/error/TX-frame/TX-tail callback counts|
|31..46|unchanged16-word packet diagnostics|
|47|GO received, only producer|
|48..51|origin rows delivered, peer rows fully released, peer requests received, request pending|
|52..55|current group, origin READY seen, actual rows archived (producer: started), notice sent|
|56..61|row active, row sending, row received, offset, group, packet payload-word count|
|62..63|latest TX length, pending request group|

Marker bits0..7 sample: prior TX count correct, not sending, no pending TX,
row inactive, predecessor rows archived, predecessor rows complete, row sender
free, compute begun. Bits8..11 group,12..15 actual TX completion count,
16..19 origin rows_done,20..23 peer_rows,24..27 archived_rows,28..31 peer requests
received. Each is latched immediately before REQUEST enqueue or responder source
overwrite/begin. Producer markers remain zero. The host requires every exact
marker, rather than inferring it from later final counts.

RX archive: observed[256], eight32-word slots (complete31-word retained RX bank
plus its actual network header). Origin uses seven, peer two, producer zero.
TX archive: tx_proof[384], six64-word slots (source31 + wire frame32 + proof
bits1 before SDK,2 control-tail start,4 actual tail completion before reuse).
Unused slots are zero. Source/frame proof is point-in-time evidence, not a
continuous DMA trace.

row_archive[128]: actual group0 and group1 rows, each64 packed u32 pairs, from
origin's row_received before release and peer's row_sent after full source
release before reuse. Producer is all zero. Every halfword is checked against
the independent arithmetic sequence, not only against the other archive.

Final reads also preserve current receive_buffer31, transmit_frame32 and selected
source31 on every PE. Current source is copied during snapshot while its pointer
is valid; uncompleted proofs cannot stand in for these live banks. A failed
capture does not assert that an unreturned array has any particular contents.

Budget: initial state768B + routing96B + up to8 states6144B + observed3072B +
TXproof4608B + RX372B + currentTX384B + current source372B + actual rows1536B +
final state768B =18120B in17 D2H. At most11 launches,0 H2D; per-file raw cap8192B,
17 raw files, total1MiB. One stop, durable returned-array saves and whole-state
final-cache equality are required. No SDK/physical run is admitted by this file.
