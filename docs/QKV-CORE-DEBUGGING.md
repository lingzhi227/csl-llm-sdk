# Reading a bounded SDK core capture

This procedure describes the successful SDK 2.10.1 offline memory path used by
the synthetic Q/K/V fixture. It does not connect to a running appliance. Keep
core and ELF files beside the capture on the execution machine; the repository
contains owned source and compact summaries only.

The explicit core basename was `corefile.cs1`. Actual output consisted of
`corefile.cs1.idx` and `corefile.cs1_0.cgz` through `corefile.cs1_7.cgz`.
Validate every file's extent and hash before treating the capture as evidence.
The complete capture totaled 1294633 bytes, but the driver allows at most 64 MiB
per file and 96 MiB combined. Larger/extra files are rejected.

Run the context selection, target selection, rectangle selection and reads in
**one csdb process**. Separate command-line invocations do not preserve the
selected context and target. Some failed CLI operations log an error and return
zero, so a zero process exit code is insufficient. Check coordinates, addresses,
exact word counts and expected data structure for every segment.

ELF symbol addresses and extents are bytes. The csdb CLI `memory read` address
and length are both 16-bit words: divide both byte quantities by two. The Python
debug helper's low-level read length uses bytes, which is a different interface.
Select only the intended rectangle and remove the default full-fabric rectangle
before reading. The successful batch checked nine disjoint role rectangles,
370 distinct PEs and 92244 u16 words. Actual ELF symbols supplied each array's offset
and extent; no address was inferred from a filename or a PE's role alone.

A representative interactive sequence is:

```text
context select out
target create --core-file=corefile.cs1
rectangle deselect 0,0 --dimension=44,12
rectangle select 4,1 --dimension=1,1
memory read --address=1916 --length=32 --stdoutput
exit
```

The last address is specific to this compiled fixture's original coordinator
state. Re-derive it from the actual ELF when source, compiler or placement changes.
The published reader checks the known intervals against actual symbol metadata
and then validates every packet counter, completion bit and committed marker.

Preserved tool limitations: using the Python debug helper in a fresh process
exited with native 139; a register-reading attempt reported an EIN/SDR architecture
initialization conflict. Neither path supplied accepted diagnostic data, and
neither was used to infer device failure. The accepted evidence came from the
single-shell memory path. A partial snapshot by itself does not establish a
permanent deadlock; the ten-second and 85-second captures demonstrate why the
sampling point and actual progress counters matter.
