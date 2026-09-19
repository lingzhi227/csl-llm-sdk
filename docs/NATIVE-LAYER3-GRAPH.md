# Complete native layer3 graph and first-error diagnosis

Two complete native layer3 graphs passed compilation, independent actual ELF
placement and original-storage checks. Both retain 33,750 application PEs,
30,576 matrix PEs, all 24 attention heads, 24 archive sinks, 136 MLP owners and
16 native KV source roots. Their 1,254 application ELFs cover the original
750 × 45 rectangle without demoted programs. Maximum ordinary storage plus a
4 KiB stack allowance is 47,552 bytes against 48,128: 576 bytes remain. This is
static fit, not a measured dynamic stack peak or neural numerical acceptance.

Fullgraph001 contains 33 CSL files. Fullgraph002 retains 32 unchanged files and
changes only the wrapper plus a new first-error module. Its origin records the
first rejection reason, actual packet length, valid-word mask and up to eight
raw application words in 44 bytes. The origin occupies 22,000 bytes including
the stack allowance. Error-only schema 5 cannot satisfy the final-origin gate.
All mathematical kernels, layout, packet transport and archive code are unchanged.

## Physical attempts and accepted evidence

Both physical attempts uploaded and independently read back all 195 original
parameter rectangles. Each saved three captures totalling 3,243,992 bytes and
996 journal records: 394 copies and three launches returned. All 24 attention
head metadata records show event 11, native masks 15/0/15/15 and zero reported
protocol errors. Those metadata do not establish attention numerical correctness
or prove that the unreturned archive arrays contain correct values.

Runtime001 stopped progressing in phase 2 with 46 MLP READY notifications and a
latched origin protocol error. Its following all-PE status read never returned.
Runtime003 retained the first rejected application packet:

```text
reason = READY_FORMAT
length = 8, valid-word mask = 255
words = [10, 1, 1, 0, 0, 3, 122, 4194568]
reserved word 7 = 0x00400108; required value = 0
```

The identity, phase and owner fields are valid. The nonzero reserved word causes
the rejection. It equals the installed SDK's eight-word, same-row network header
for the origin at physical coordinate (4, 1). This exact bit-pattern agreement
suggests a framing or buffer-lifetime investigation; it does not identify the
sender as faulty or prove an upstream cause. Later observer fields show phase 2,
offset 13,184 and 40 MLP READY notifications. Those are live values at snapshot
time, not atomic context captured with the first error.

Runtime003's following 24-sink archive-status read also never returned. The
30-second progress watchdog cancelled each failed attempt; independent checks
confirmed terminal jobs, no active system assignments, gone owned processes and
reaped waiters. Neither attempt has a normal stop, complete capture, actual
archive payload, final parameter-retention check or full-layer numerical audit.
Complete original neural-layer epochs remain **zero**. Runtime002 was an
unadmitted source draft corrected before allocation and was never executed.

## Next qualification

Keep strict READY validation. Use a bounded, concurrent same-row producer
fixture to distinguish source-buffer corruption from network-header/payload
framing, with independent diagnostic transport and explicit resource bounds.
Any SDK pass qualifies only that fixture; a physical merge failure may require
a separate physical reproduction. Future incomplete-layer captures should stop
after the durable observer instead of issuing another read to a busy sink.
Full numerical comparisons and sequential-stage integration remain ahead.

All earlier failed SDK, simulator, fit and physical attempts remain documented
in the [archive report](QK-ARCHIVE-PHYSICAL.md), [native transport report](NATIVE-KV-SDK.md)
and [FIFO report](LAYER3-FIFO-TRACE.md). No failure was replaced by a success label.

[Compile source](../examples/native_layer3) ·
[Compile evidence](../evidence/native-layer3/compile-summary.json) ·
[Runtime prefix evidence](../evidence/native-layer3/runtime-prefix-summary.json) ·
[Source provenance](../evidence/native-layer3/source-map.json)
