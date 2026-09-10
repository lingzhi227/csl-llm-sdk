# Implementation status

WP00–WP04 accepted, 2026-09-10. Continuous development is
active under bounded control-node packages. Full model M0 and inference remain incomplete.

| Component | Status |
|---|---|
| Resource admission, cache/entry checks | CPU unit tests passed, including injected low-resource readings |
| Single heavy-job lock | Contention/release fixture passed |
| Three host bit codecs | CPU patterns/odd-offset/dtype/length/negative tests passed |
| cgroup bounded execution | Actual MemoryMax 20 GiB and MemorySwapMax 0 confirmed; serial execution |
| Native memcpy16 SDK bit copy | One WSE-3 PE, seven BF16 bit patterns, SDK 2.10.1 simulator; exact output and normal stop/exit |
| Packed-u32/raw-u32 SDK transfer | CPU codec tests only; no SDK qualification |
| Pinned model structure/equations | 851 metadata matches and27 extracted-function CPU checks accepted; full runtime integration remains open |
| Full checkpoint, GPU reference | Not qualified; no full-model download/execution |
| Local 128×112 BF16 GEMV | Four same-runtime SDK calls passed on an original Qwen3.8 weight slice; WP01 accepted |
| Two-PE GEMV/fabric/SUM/RMS128 | Four SDK calls with both device join orders accepted; reduced unit-gain operator fixture |
| Persistent 128×5120 contraction | Four full-width same-runtime SDK calls accepted, 46 tiles each; 128 selected output rows |
| Full generation, physical WSE-3 | Not implemented/qualified here |

Attempt wp00-001 failed in the installed wrapper's temporary-directory mount before
cslc ran; simulation did not start. Controller-approved wp00-002 corrected only the
container launch and recorded normal compiler and simulator exits. Original failure
is preserved. See [WP00 report](WP00-REPORT.md) for measurements and scope limits.
