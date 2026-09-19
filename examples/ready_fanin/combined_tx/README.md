# Combined-frame transmission variant

This exact packet module compiled for all 356 fixture PEs, then failed the strict
physical protocol check. It is a diagnostic comparison, not a repair.

The variant owns one 32-u32 transmit frame. It encodes the same SDK network
header, copies 1..31 application words, then injects the complete 2..32-word frame
in one asynchronous operation. Both descriptor lengths are payload length+1.
The original caller lease and source-completion callback remain. This adds 128
declared bytes; actual fixture maximum including 4 KiB stack is 25,600 bytes.

To assemble its source, copy the parent example to a separate working directory
and overlay these three files. The other five CSL files, strict checker and
physical capture components are unchanged. The replacement source_tests.py
checks the exact TX delta and existing saved-record fixtures. A fresh source
manifest and site-authorized compile/runtime are still required; historical
admissions and compiled artifacts are not distributed.

The physical run again recorded `[10,1,1,0x00400108,0,10,0,1]` as a malformed
received payload. All 408 source snapshots were exact; source and peer archive
hashes matched baseline. It exited normally and all 13 captures were preserved.
The assigned physical system differed from baseline, so this is not a strict
same-system causal A/B. No unique root cause or full-model improvement is claimed.

[Report](../../../docs/READY-FANIN-REPRODUCTION.md) ·
[Provenance](../../../evidence/ready-fanin/combined-source-map.json) ·
[Capture receipts](../../../evidence/ready-fanin/combined-capture-receipts.json)
