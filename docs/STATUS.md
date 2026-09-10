# Implementation status

WP00–WP08, WP10 and WP11 accepted, 2026-09-10. Continuous development is
active under bounded control-node packages. Full-model inference remains incomplete.

| Component | Status |
|---|---|
| Resource admission, cache/entry checks | CPU unit tests passed, including injected low-resource readings |
| Single heavy-job lock | Contention/release fixture passed |
| Three host bit codecs | CPU patterns/odd-offset/dtype/length/negative tests passed |
| cgroup bounded execution | Actual MemoryMax 20 GiB and MemorySwapMax 0 confirmed; serial execution |
| Native memcpy16 SDK bit copy | One WSE-3 PE, seven BF16 bit patterns, SDK 2.10.1 simulator; exact output and normal stop/exit |
| Packed-u32 host codec | CPU tests only; no packed BF16 SDK qualification |
| Raw32 recurrent frames | Generation/token/phase headers and128-value bodies pass two-PE SDK transfer checks |
| Pinned model structure/equations | 851 metadata matches and27 extracted-function CPU checks accepted; full runtime integration remains open |
| Full checkpoint, GPU reference | Not qualified; no full-model download/execution |
| Local 128×112 BF16 GEMV | Four same-runtime SDK calls passed on an original Qwen3.8 weight slice; WP01 accepted |
| Two-PE GEMV/fabric/SUM/RMS128 | Four SDK calls with both device join orders accepted; reduced unit-gain operator fixture |
| Persistent 128×5120 contraction | Four full-width same-runtime SDK calls accepted, 46 tiles each; 128 selected output rows |
| Ordinary RMS5120 + device BF16 RNE | Four synthetic calls accepted under48 KiB application SRAM; shared gain/output lifecycle |
| Full128×128 recurrent state | Four synthetic tokens/three generations on two PEs accepted; all device reductions and updates, normalized/scaled inputs |
| BF16 gated RMS128 and recurrence composition | Four standalone calls and four composed tokens accepted; third consumer PE, input/early/product/final BF16 casts, ACK before root completion |
| Selected-head preprocessing384 | Eight tokens accepted: width4 history, BF16 conv/SiLU, Q/K L2/scaling, beta/g/decay; pending-gates/finalize commands; standalone only |
| WP09 recurrent-head integration | Not accepted: eight-token numerical observations pass, but final weight readback/runtime lifecycle fails; investigation remains open |
| WP10 attention core and KV cache | Accepted: one256-dimensional query/KV head, capacity8; ten tokens, overflow/reset and normal stop; all2560outputs match official BF16 reference |
| WP11 Q/K preprocessing | Accepted: ordinary RMS256 and device partial RoPE64 for text positions0–7; ten calls and all5120official BF16 outputs match, with separate product casts and tail/sign checks |
| WP12 Q/K-to-attention composition | In development: original Q/K through normalization/rotation and persistent attention; coupled source-error/domain qualification required |
| Full generation, physical WSE-3 | Not implemented/qualified here |

Attempt wp00-001 failed in the installed wrapper's temporary-directory mount before
cslc ran; simulation did not start. Controller-approved wp00-002 corrected only the
container launch and recorded normal compiler and simulator exits. Original failure
is preserved. See [WP00 report](WP00-REPORT.md) for measurements and scope limits.
