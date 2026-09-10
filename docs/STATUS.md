# Implementation status

WP00–WP08 and WP10–WP15 accepted, 2026-09-10. Continuous development is
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
| WP12 Q/K-to-attention composition | Accepted: one PE, original synthetic Q/K/V/gate through normalization/rotation, device copy and persistent cache8 attention; ten successes, overflow/reset and normal stop |
| WP13 original attention projections | Accepted: all1024 selected Q/rawgate/K/V rows over5120columns, four common inputs, eight independent single-PE runtimes;4096final outputs and lifecycle/source gates pass for standalone projection coverage |
| WP14 original hidden Q/K preprocessing | Accepted: four full5120 Q/K projection PEs hand off exact operands to a trained RMS/partialRoPE consumer; one dense call atposition1, all source/stage/retention checks and normalstop; no attention/KV composition |
| WP15 original projected attention | Accepted: ten PEs, all1024 selected rows/full5120columns, two same-request original-hidden tokens, persistent KV, exact device handoff and all source/stage/cast/cache/retention gates; normal stop in1187.19s |
| WP16 original full MLP | Source, acquisition, full CPU-reference and CSL resource planning at5120→17408→5120; downloads/CPU/SDK not yet admitted |
| Full generation, physical WSE-3 | Not implemented/qualified here |

Attempt wp00-001 failed in the installed wrapper's temporary-directory mount before
cslc ran; simulation did not start. Controller-approved wp00-002 corrected only the
container launch and recorded normal compiler and simulator exits. Original failure
is preserved. See [WP00 report](WP00-REPORT.md) for measurements and scope limits.
