# Full136 static transport: physical qualification

One original physical WSE-3 run on 534 PEs passes the complete static transport
contract. All 136 concurrent producers deliver READY, alongside 8 requests,
24 row fragments and 1,024 BF16 halfwords. Both complete 1,777,600-byte export
sets match exactly and independently decode successfully. Normal runtime exit,
owned-process exit, system release and durable backup are verified.

The network uses dedicated retained packet inputs and explicit software
arbitration. Each source remains owned through its actual completion callback.
All 175 relay links have the expected provenance, counts and release records;
receiver identity, full payloads, bounds, error absence and same-PE event
causality pass. This physical run disables the small fixture's forced gate.

The compile contains 534 actual application programs covering all 534 PEs.
Independent actual instruction/task/DSR review and 4,157 named-bank overlap
checks pass. Maximum ordinary storage plus the 4,096-byte stack allowance is
30,064 bytes, leaving 18,064 bytes below the 48,128-byte ceiling. This does not
measure dynamic stack peak.

The capture contains 29 original raw files totaling 3,710,952 bytes and
3,703,296 host-transfer bytes. It uses two launches, no host-to-device payload
copies and one status poll. The independent backup rehashes all 68 selected
original files; the model, compiled artifact and raw arrays are not published.

Capture plus normal context stop took about 6.65 seconds. The full guarded job
lifecycle took about 101.75 seconds. Those include runtime/capture overhead;
they are neither neural token latency nor a measured transfer bandwidth.

This is one fixture epoch without reset. It does not qualify original Layer3,
all 64 model layers, repeated decode, cache restoration or the three-stage
checkpoint workflow. Complete original neural epochs and model generations
remain zero. The previous native message-passing failures and small simulator
capture-timeout failure remain preserved. The new transport pass does not prove
a unique cause of the earlier corruption or continuous DMA/wire immutability.

The next integration step applies this transport ownership design to the
original neural graph, including all phases, query credits, native K/V,
normalization, matrix outputs and final consumers. Its distinct program fit,
resource allocation and original numerical outputs still require validation.

[Exact source](../examples/ready_fanin/static_transport/full136) |
[Evidence](../evidence/ready-fanin/static-transport-full136.json) |
[Earlier small qualification](STATIC-TRANSPORT-SMALL.md) |
[Preserved sender-frame failure](FULL136-SENDER-FRAME.md)
