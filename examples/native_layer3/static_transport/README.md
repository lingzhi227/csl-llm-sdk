# Original Layer3 with static transport

This directory contains the six changed CSL files from the complete original
Layer3 static-transport graph. The other 31 CSL files are byte-identical to the
published `layer3-native-fullgraph-002` sources. The [source map](source-map.json)
binds every accepted file by size and SHA-256; paths are repository-relative.
Gather all 37 files by their original basenames in one compile directory.

The actual graph has 67,500 PEs in a 750x90 application. The 33,750 original neural
PEs map to odd rows; spacer rows and a dedicated spine carry explicit static
packet routes. It retains the original weights, kernels, normalization, 24
attention heads, 136 MLP owners and native K/V paths. Compiler options are:

```
--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out
```

The accepted compile contains 1,299 application programs. Its largest ordinary
allocation plus 4,096 bytes of reserved stack is 47,936 bytes, below the 48,128-byte
ceiling by 192 bytes. Dynamic stack high-water usage was not measured.

One physical run completed position 0 and exited normally. Its original offline
operator audit reports all gates passed. The resulting hidden vector differs
from the nominal CPU BF16 reference at 346 of 5,120 values; maximum absolute
difference is 0.00390625. This is neither bitwise whole-layer equivalence nor a
propagated whole-layer error enclosure. All 64-layer inference, nonzero positions,
repeated decode/reset and stage checkpoint restoration remain unqualified.

The source fixes an observed generated-code receive-address discrepancy by
compacting the completed packet payload, one 32-bit word at a time, into the same
receive allocation before the original application callback. It preserves the
receive lease through that callback. A noinline application callback did not fix
the discrepancy; a plain shift loop emitted an unavailable memmove dependency.
The per-word helper is the variant checked in the actual accepted artifact.
Compiler-pass causality and the earlier native message-passing corruption remain
separate unresolved questions.

Host admission wrappers, SDK distributions, compiled artifacts, parameter arrays
and raw captures are not included. This is an exact device-source and evidence
bundle, not an automatically authorized hardware launcher.

[Report](../../../docs/LAYER3-STATIC-TRANSPORT.md) |
[Original numerical audit](../../../evidence/native-layer3/static-transport/audit.json)
