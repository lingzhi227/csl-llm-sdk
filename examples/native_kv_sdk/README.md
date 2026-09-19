# Accepted native KV and Q ownership fixture

This is the byte-exact device and host source of the accepted 45-PE SDK
transport fixture. Three operations cover a consecutive position and a reset,
four independently delayed native streams, retained Q packets, source/receiver
ownership and observer coexistence. A synthetic consumer checks each head's
original-sized buffers once. No neural arithmetic or model weights are used.

The 14 CSL files, compiler inspector and runtime/checker/persistence sources
match their frozen inputs. `compiled.json` and `parent-manifest.json` are small
historical identity metadata; compiled ELF files and vendor SDK code are omitted.
`reuse_gate.py` intentionally pins the accepted binaries and three generated
SDK auxiliary files. `entry.py` is runtime-only and never compiles.

For reproduction, use SDK 2.10.1 in a fresh execution directory under a hard
external resource guard. `compile.py` builds and inspects the actual 45-PE layout.
Keep the published historical `compiled.json` outside that fresh output
directory. A new compilation requires its own checked receipt and pins before
the runtime-only reuse gate can admit it; historical hashes are not permission
to execute arbitrary new binaries. Supply the matching original source manifest
and review any new SDK auxiliary identities. Site installation and cgroup
launch wrappers are deliberately not distributed here.

The accepted runtime profile was CPU 0, one simulation thread, 1 GiB memory,
zero swap, 128 processes, 300-second driver and 330-second outer deadline.
Use 8 MiB per file, 2 MiB logs, 32 MiB candidate storage, 8 GiB available RAM
reserve and 32 GiB free on both system and execution disks. Preserve failures,
stop/reap the owned process group and verify release. Do not retry automatically.

The driver performs eight launches and thirteen D2H calls, totaling 209,664 host
bytes. Each completed raw capture is fsynced and published without overwrite
before the next SDK call. The incomplete progress receipt survives interruption;
only a passing final result with normal stop establishes successful completion.
`capture_store_tests.py` and `reuse_auxiliary_tests.py` are SDK-free fault checks.
`host_gate_tests.py` constructs synthetic examples for the numerical-bit and
ownership checker; these constructed examples are not device evidence.

The published [raw result](../../evidence/native-kv-sdk/result.json),
[individual captures](../../evidence/native-kv-sdk/captures),
[journal](../../evidence/native-kv-sdk/journal.jsonl) and
[source mapping](../../evidence/native-kv-sdk/source-map.json) retain the accepted
observations. See the [report](../../docs/NATIVE-KV-SDK.md) for all three failed
attempts and the exact limitations. Full neural-layer and model execution remain open.
