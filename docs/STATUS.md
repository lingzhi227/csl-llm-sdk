# Implementation status

WP00 candidate, 2026-09-10. Full model M0 and all inference milestones are incomplete.

| Component | Status |
|---|---|
| Resource admission, cache/entry checks | CPU unit tests passed, including injected low-resource readings |
| Single heavy-job lock | Contention/release fixture passed |
| Three host bit codecs | CPU patterns/odd-offset/dtype/length/negative tests passed |
| cgroup bounded execution | Actual MemoryMax 20 GiB and MemorySwapMax 0 confirmed; serial execution |
| Native memcpy16 SDK bit copy | One WSE-3 PE, seven BF16 bit patterns, SDK 2.10.1 simulator; exact output and normal stop/exit |
| Packed-u32/raw-u32 SDK transfer | CPU codec tests only; no SDK qualification |
| Model semantics, full checkpoint, GPU reference | Not qualified; no full-model download/execution |
| Neural CSL kernels, full generation, physical WSE-3 | Not implemented/qualified here |

Attempt wp00-001 failed in the installed wrapper's temporary-directory mount before
cslc ran; simulation did not start. Controller-approved wp00-002 corrected only the
container launch and recorded normal compiler and simulator exits. Original failure
is preserved. See [WP00 report](WP00-REPORT.md) for measurements and scope limits.
