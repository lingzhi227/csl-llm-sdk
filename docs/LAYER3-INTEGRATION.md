# Complete layer 3 integration: compiled graph and bounded diagnostics

The full original layer 3 graph compiles for WSE-3 and its original parameters
are prepared and independently checked. A physical diagnostic reached 15 of 24
attention READY notifications. A separate synthetic SDK graph completed the
same phase 0 Q/K/V root-to-head fanout pattern. These are separate results: no
complete original layer 3 neural epoch or full CSL text generation is accepted.

## Original layer and physical boundary

The graph occupies 750 x 45 application PEs: 33750 total, including 30576 matrix PEs.
Its seven matrices use 744488960 original BF16 bytes, padded to 751435776 bytes.
The 103-file prepared bundle totals 752335928 bytes. Original bits, padding,
coordinates, gains and reference identities were checked independently.

The published baseline compiled to 313 application programs, with a maximum
ordinary-section end plus 4096-byte stack allowance of 47984 bytes under the
48128-byte ceiling. That is a static admission check, not a measurement of peak
dynamic stack use. The source is in [examples/layer3](../examples/layer3).

A later 315-program diagnostic variant placed a separate 32-word observer at an
idle eastern PE. Original parameter uploads and bitwise readback, initialization,
and the first input preparation passed. The retained position 0 snapshot was
coherent: heads 0–13 and 18 had reported READY; heads 14–17 and 19–23 had not.
An unrelated host logging error occurred after the raw snapshot was saved.
Normal shutdown and resource release succeeded; the failed host status remains
part of the record. This is evidence about partial progress, not a successful
layer output, a permanent-deadlock proof, or a latency measurement.

## Synthetic fanout result

[examples/qkv_fanout](../examples/qkv_fanout) removes all neural arithmetic and
uses distinct finite BF16 bit patterns. It preserves 112 phase 0 root senders,
24 head collectors, sequential six-way K/V fanout, row headers, callback
ownership and READY handling. Monotone X compression reduces the application
to 37 x 10, while preserving Y coordinates and horizontal direction/turn ordering.
It changes distances, timings and matrix concurrency, so success does not
qualify the full original layer.

The seven-program fixture completed all 370 PEs, all 112 roots and all 24 heads.
Independent offline decoding checked 576 source and 576 received row packets,
192 row-fanout completions, 24576 exact received u16 channels, and origin's 24 READY
notifications with mask `0xffffff`. Every expected counter and marker matched.
Its largest ordinary-section end plus 4096 stack allowance was 14752 bytes.

The driver used three copies before compute and two launches, with 41440 host-slot
bytes. It performed no memcpy after compute. A single explicit core capture was
followed by normal shutdown and verified resource release. The fixed 85-second
wait is a diagnostic sampling choice in the simulator, not token or kernel
performance. The accepted compiled artifacts were reused for this later capture.

The earlier ten-second snapshot retained real partial progress: 164 source packet
completions, 133 received packets, 40 row-fanout completions and 5480 committed channels
with zero marker mismatches. It sampled callbacks still in progress. Its host
file check also incorrectly expected at most four core files; the actual SDK
format had an index and eight compressed shards. Both the failure and original
capture were preserved. A new runtime with corrected format checking and a later
sampling time supplied the complete result; the earlier capture was not retried
or relabeled as complete.

## Next integration work

The next diagnostic observes the first missing original head directly, including
its collector counts, Q/K normalization and rotary stages, attention completion,
error returns and READY transmission. Only accepted source and evidence are
published here; that new diagnostic is still under development. Full-layer
numerical checks, longer contexts, recurrent-layer integration, stage state
restoration and complete 64-layer CSL generation remain open.

[Compact result record](../evidence/layer3-integration.json) ·
[Offline debugging notes](QKV-CORE-DEBUGGING.md)
