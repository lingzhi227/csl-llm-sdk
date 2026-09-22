# Small static transport qualification

The three-producer static transport fixture completes on the workstation SDK
simulator. Its 27 processing elements deliver exactly 3 READY packets,8 REQUESTs,
24 row fragments and 1024 halfwords. Two full 50016-byte export sets are identical;
all retained source and relay records pass the original static-v1 acceptance
schema. This is a small transport qualification, not full 136-producer hardware or neural
inference. Complete original CSL neural epochs and model generations remain 0.

## Protocol and ownership

Each producer sends the complete eight-word READY identity. A separate relay
row forwards packets through independent one-hop links. Each relay owns two
retained input buffers and selects one complete packet for output. Actual output
completion releases the selected buffer. The other input can remain held, and
round-robin arbitration selects between two ready inputs. REQUEST and row data
use separate channels; starting a request does not wait for all READY packets.

The small qualification gate delays the first output-completion callback until
both input buffers are held. Captured state and both complete eight-word inputs
prove this callback/lease condition. They do not establish a physical link stall
or observe continuous DMA reads. The gate is disabled in the proposed full 136-producer
fixture, whose physical compilation and runtime remain separately unqualified.

## Evidence and preserved failure

The first runtime completed the device protocol and a full valid export set,
but hit its 45-second child deadline during the second origin-record read. It
has no normal stop and remains a strict failed run. Nothing from that partial
run is relabeled as a complete qualification.

The second runtime changes only the child and outer guard time limits to 75 and
90 seconds, retaining the 30-second progress limit, identical compiled artifacts,
CSL, driver, schema and complete double-read acceptance. It exits normally.
The child lifecycle takes 64.073 seconds, the capture result reports 63.201 seconds,
and the guard takes 64.837 seconds, including simulator and capture overhead.
These values are not inference or token latency.

It retains 37 original NPZ files totaling 129384 bytes from 119616 host bytes,
four progress polls, two launches and zero H2D. Independent raw parsing rechecks
SHA256, dtype, extent, every meaningful export and both complete schema results.
All observed owned processes are absent and the bounded unit is quiescent.

The accepted small compiler output contains 27 programs covering 27 PEs, with a
maximum 29792 bytes including 4096 stack allowance under a 48128-byte ceiling.
Separate instruction review checks actual task tables, descriptor ownership,
retained-input forwarding and the gate's scalar copies. This does not qualify
the larger physical geometry or neural compute roles.

Historical native-control full 136-producer failures remain preserved. Their cause is
not established by this alternative's small success. No raw data, binaries,
vendor implementation, credentials, admission or execution guard is published.

Independent structural review accepts the first-observed generated files:
all use fabric 16 x 5; coordinate loads cover all 16 columns and 5 rows; memcpy
traffic-map writes stay outside the application. Default payloads match the
previous accepted SDK artifact. No independent regeneration is claimed.

[Exact sources](../examples/ready_fanin/static_transport) |
[Evidence](../evidence/ready-fanin/static-transport-small.json) |
[Previous full 136-producer sender checks](FULL136-SENDER-FRAME.md)
