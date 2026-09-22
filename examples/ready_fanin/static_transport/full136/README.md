# Full136 static transport on physical WSE-3

One physical run on 534 PEs passes all 136 READY messages, 8 requests,
24 row fragments and 1,024 BF16 halfwords. Two complete 1,777,600-byte exports
are identical. All retained source/input, relay provenance, sink and local
causality checks pass, and the runtime exits normally. This is a transport
fixture with one epoch and no reset; complete original neural epochs remain zero.

The two files here are exact accepted physical sources. Reuse `app.csl`,
`relay.csl`, `static_input.csl`, `static_output.csl` and `schema.py` from the
[parent directory](..), plus `diagnostics.csl` and `row_packet.csl` from
[first_fault](../../native_control/first_fault). Place the CSL files together
when compiling. The explicit compiler parameters are
`--params=producer_first:40,producer_count:136`; the layout defaults disable the
small qualification gate. Application geometry is 178x3, fabric 762x1172,
offset (4,1), memcpy enabled, one channel and one compiler worker.

The captured schema remains `static-ready-row-v1`. Host resource/admission
wrappers, vendor SDK, binaries and raw arrays are excluded. These are source
and result records, not an automatically authorized execution tool.

[Report](../../../../docs/STATIC-TRANSPORT-FULL136.md) |
[Evidence](../../../../evidence/ready-fanin/static-transport-full136.json)
