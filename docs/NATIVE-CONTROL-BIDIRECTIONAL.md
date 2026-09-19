# Physical bidirectional payload and row lifecycle

Physical RUN007 now passes all 31 independent checks after the two reviewed GO
relocation edits. It transfers 9 packets/200 exact words, assembles two synthetic
rows, releases both peer source rows, preserves full source/frame/RX banks and
suffixes, and keeps the complete 64-word state stable. Both first-TX lease
witnesses are exact (0x10107 and 0x107). Origin separately records one RX while
an issued TX remains unfinished. This is software lease-overlap evidence;
simultaneous DMA and a unique timing root cause are not established.

The change moves the existing one-shot GO after origin prepares row state and
REQUEST0, immediately before its send call. It adds no delay, acknowledgement,
source hold, host pacing or relaxed check. The original RUN006 source, its
30/31 overall failure, and all simulator failures below remain preserved.
The unchanged strict checker is shared by both physical attempts.

RUN007 saves 10 arrays/15,384 bytes via 10 D2H calls carrying 12,744 bytes, 4 launches
and zero H2D. Capture plus normal exit takes 8.169 s; the complete supervised stage
takes 200.346 s including setup. These measurements are not inference latency.
The actual owner exits 0, the job is terminal and unassigned, and the known
guard/waiter plus bounded candidate-process scan are clear. The historical
client PID was unavailable and is explicitly recorded as such. Original raw
evidence and sources are independently verified at the compute site.

Physical compile 009 retains the 19,632-byte maximum including 4,096-byte stack
allowance. Producer/peer executable sections and DSR records match 008; origin
code/task-table addresses change with the GO relocation and were independently
checked. No whole-ELF or whole-program identity claim is made.

[Exact prepared-launch source](../examples/ready_fanin/native_control/bidirectional/prepared_launch) |
[Independent acceptance evidence](../evidence/ready-fanin/native-control/bidirectional-prepared-launch.json)

## Original RUN006: accepted subscopes and strict failure

Physical RUN006 independently passes payload, full-buffer, two-row lifecycle and
stable-state checks. Its overall strict result remains failed: 30 of 31 checks
pass, with the dual-first-TX lease witness absent on the READY producer. This is
an accepted transport subscope, not a complete run pass, a repair of the 136-producer
failure or a neural epoch. Complete original neural-layer epochs remain zero.

## Application and exact observations

Origin 0 sends two 8-word REQUESTs to peer 2. Each response has 31, 31, 26 words and
reconstructs 128 synthetic halfwords; producer 1 contributes one 8-word READY.
Nine packets total 200 application words. The original packet and row modules
retain source buffers until actual release callbacks. The second request follows
the prior row release and request-source completion. A bounded pending request
at the peer waits for source release before the next row starts.

Actual TX counts are 2/1/6 and RX counts 7/0/2. READY appears first among origin's
seven received packets. Both assembled rows and both peer source releases are
exact. The checker covers every complete 31-word RX bank, the actual header,
every 31-word source and 32-word frame, short-packet suffixes, three-point source
lease proofs, current buffers, actual row archives and causal start markers.
The 64-word state snapshot at all 3 PEs is byte-identical before and after the
final read window. No protocol, harness or bad-notice error is recorded.

The physical host saved 10 arrays/15,384 stored bytes with 10 D2H calls carrying
12,744 bytes, four launches and zero H2D. One status poll was sufficient; the
budget permits 1..8 polls, 10..17 copies and 4..11 launches. The context exited
normally. Capture and exit took 7.830 s; the complete supervised stage took 203.528 s,
including service setup. These are not inference latency or device throughput.
The owner returned 1 because the strict checker failed. Independent cleanup found
the job terminal, no assigned system and no remaining owned processes. Original
source, raw arrays and receipts were backed up at the compute site.

## Preserve the strict overlap failure

Each of origin and producer emits a tagged notice only after its first frame is
issued. The receiving notice handler must sample its own first TX lease before
software release. Expected witnesses are 0x10107 and 0x107. Actual values are
0x10107 and 0x12a. Producer's value records one correct notice, one completed TX,
first completion true, sending false and first issue true. Thus its handler ran
after the first lease ended. This does not determine the physical arrival time
of the notice. The strict dual-witness check remains unchanged and failed.

Separately, origin's actual packet counter records one receive while a TX was
issued and unfinished; producer and peer record zero. This directly qualifies
that local lifecycle overlap on origin. It neither supplies the missing producer
witness nor proves simultaneous DMA or same-cycle router arbitration.

The fixture originally releases its one device GO before origin prepares the
first row receive state and REQUEST. A minimal, separately reviewed follow-up
moves that GO after this preparation. It makes no timing guarantee and adds no
host pacing, delay, tail hold, acknowledgement or weaker pass criterion. This
report and published source retain the original failed RUN006 unchanged.

## Compiled fit and preserved simulator attempts

Physical compile 008 covers 3 PEs at offset 4,1 on 762 x 1172 fabric. All four embedded
CSL files, full banks, retained scalars, task tables and every emitted vector DSR
operation are independently checked. Executable text/task/configuration bytes
and addresses match simulator compile 021; complete ELF files and fabric geometry
differ. Maximum ordinary storage plus 4,096-byte stack allowance is 19,632 under
the 48,128 gate. The allowance is not a measured dynamic stack peak.

SIM012 failed before compute: its initial 768-byte state copy returned exactly,
then the 96-byte routing copy received zero bytes. Traced SIM013 completed only
the shortened initialization prefix, 864 host bytes and normal stop. Full traced
SIM014 again failed before compute, and retaining the Python platform reference
through SIM015 did not repair that failure. These three full attempts and the
prefix-only success remain separate outcomes. Their original receipt hashes,
child exits and saved-capture counts are retained in the evidence summary.
No simulator root cause is established and no unchanged retry is claimed.

RUN006's full physical payload path succeeded through those reads and application
traffic. The different SDK/backend/deployment paths are a combined factor; this
does not uniquely isolate silicon from simulator or host behavior. The next main
integration scope is 136-producer phase-dependent dataflow. Waiting for every phase 3
READY before issuing phase 2 REQUEST would deadlock the intended producer chain
and is not a proposed fix. Fullgraph task/queue/DSR ownership and original-layer
numerics remain required before model generation.

[Own source](../examples/ready_fanin/native_control/bidirectional) |
[Scoped evidence and preserved failures](../evidence/ready-fanin/native-control/bidirectional-attempts.json) |
[Earlier matched 73-word pass](NATIVE-CONTROL-MULTISOURCE.md) |
[Unresolved physical 136-producer failure](NATIVE-CONTROL-PHYSICAL.md)
