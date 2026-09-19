# QK diagnostic archive: physical alias equivalence

A four-PE synthetic fixture ran the original QK and attention kernels on physical
WSE-3 and passed two reset generations. Every archived diagnostic word and final
BF16 output matched a separate owner with independent diagnostic storage. The
source workspace was actually overwritten before the receiver was allowed to
commit its archive. The run stopped normally, and independent review confirmed
terminal scheduler state, no live assignments and all owned processes absent.

This qualifies the diagnostic storage lifetime and archive protocol with synthetic
inputs. It is not original-weight neural-layer acceptance, dependent decoding,
full-model inference or a performance result.

## Why an archive is needed

The selected complete native attention programs previously exceeded the 48,128-byte
per-PE ceiling: the largest measured ordinary storage plus 4 KiB stack was 51,424
bytes. Six QK diagnostic arrays occupied another 3,840 bytes. Their lifetimes allow
them to share the original 5,120-byte attention workspace, provided their contents
are sent to a separate sink before attention reuses that workspace.

| Diagnostic view | Workspace byte interval | Retained type/count |
| --- | --- | --- |
| Trigonometric values | 0–512 | 128 FP32 |
| Rotary stages | 512–2048 | 384 FP32 |
| Product casts | 2048–2560 | 256 BF16 |
| Trigonometric casts | 2560–2688 | 64 BF16 |
| Probes | 2688–2816 | 32 FP32 |
| Normalized Q/K | 4096–5120 | 512 BF16 |

The archive contains 960 raw 32-bit words. BF16 pairs use explicit low/high
halfword packing. The first block contains 704 words and the second 256 words;
a separate 32-word marker carries schema, head, generation and the five-part
request/reset/position/stage/operation identity. Source completion permits local
reuse. Receiver completion and commit are separate checks, and a late protocol
error invalidates the retained commit even when its guard words remain unchanged.

## Actual device sequence

PE 0 runs the aliased owner, PE 1 stores the archive, PE 2 joins release notices,
and PE 3 runs the baseline with independent diagnostic arrays. Both owners execute
the original arithmetic. Q is fed through the local collector, and synthetic K/V
is preloaded locally; this fixture does not test native KV transport.

After QK, the baseline retains all six diagnostic arrays. The aliased owner sends
its marker and both data blocks, runs attention, then poisons all 1,280 source
workspace words. The sink receives both blocks but defers commit. The join PE
waits for both actual source-overwrite and sink-receipt notices before releasing
commit. No host operation supplies an operand or releases this device dependency.

The host captures each completed generation before preparing the next. This is
an explicit capture-before-reuse contract, not a device acknowledgement that the
host consumed the archive. Both operations use position zero with different reset
identities and inputs. They are two resets, not two dependent decode positions.

## Accepted evidence

| Check | Result |
| --- | --- |
| Device layout | Four programs, 4×1 PEs, full fabric 762×1172, offset (4,1) |
| Ordinary storage plus 4,096-byte stack, by role | 42,688 / 18,528 / 19,712 / 41,488 bytes |
| Archive equivalence per reset | All 960 raw words equal the independent baseline |
| Final output per reset | All 256 BF16 values equal the baseline |
| Actual source overwrite per reset | All 1,280 words contain the expected poison |
| Invalid API calls | 61 actual rejections and 172 successful assertions |
| Host operations | 6 launches, 13 D2H copies, 0 H2D, 32,768 host slot bytes |
| Durable evidence | 13 JSON captures, 71,632 bytes; 43 journal events |
| Driver / guarded elapsed time | 180.265659 / 203.014262 seconds |
| Completion | Normal context exit, reaped exit 0, independently released |

The observed sink callback order was 132 in both resets. Both final-before-commit
and commit-before-final predicates were also checked through actual isolated
device API calls; that does not claim both physical wire orders were observed.
The negative phase checks all other evidence words remain unchanged.

Each raw capture is fsynced before the next SDK operation. The runtime reuses the
accepted compilation and verifies six compile receipts plus the artifact hash
before allocation. The 300-second runtime ceiling includes SDK setup and exit;
the measured elapsed time is not inference latency. Official billing was not
audited. The four-PE placement and SRAM checks do not establish fit for all native
heads or a complete layer graph.

## Separately accepted selected complete-program fit

The archive change also passed a separate compile-only check at 89 original
layer-3 coordinates: all 24 complete attention heads, all 24 archive sinks,
35 matrix PEs including all 16 native KV roots, both norms, two MLP owners,
the origin and global observer. The other tile programs were demoted to idle
code, so this layout must never execute as a neural layer.

Independent inspection checked all 1,142 actual ELFs, complete 33,750-PE
geometric coverage, the 89 selected programs, 33 embedded CSL sources, original
live bank extents and disjointness. Maximum head storage including the 4,096-byte
stack allowance is 47,552 bytes, leaving 576 bytes below 48,128. Prenorm, postnorm,
matrix and sink maxima are 46,384 / 46,016 / 44,240 / 17,248 bytes. The earlier
selected-fit002 failure at 51,424 bytes remains preserved; this successful change
includes functional archive/alias code and must not be described as outlining alone.

The compile guard completed in 257.739635 seconds, returned zero and released
all owned resources. This accepts selected complete-program storage and placement.
It does not accept fullgraph fit, dynamic stack usage or original-layer numerics.
The complete native graph is the next separately admitted compilation.

[Selected-fit source](../examples/qk_archive/layer3-native-fullfit-003) ·
[Selected-fit evidence](../evidence/qk-archive/selected-fit-summary.json)

## Preserved failures

| Attempt | Result and next change |
| --- | --- |
| SDK001 | Frontend rejected a global comptime pointer cast. Runtime accessors replaced it; no ELF or runtime was produced by this attempt. |
| SDK002 | Four programs compiled. The inspector incorrectly required an optimized-away write-only baseline workspace. The next inspector made only that baseline workspace optional. No runtime ran. |
| SIM001 | Two captures persisted before a zero-byte baseline read and native abort. No complete epoch or normal stop. |
| SIM002 | A same-value scratch-transfer control failed during the second read pair; one capture persisted. It did not qualify the workaround. No additional simulator retry was used. |
| Physical RUN001 | All thirteen raw captures passed independently, but setup and work consumed about 172.6 seconds of a 180-second limit. Context exit exceeded the remaining time; the journal ends at stop_enter. Its raw-only acceptance and failed lifecycle remain preserved. |
| Physical RUN002 | A separately admitted 300-second runtime limit completed normally using the same driver, checks and compiled artifact. Fourteen prior source files were byte-identical; no new CSL compilation was required. |

All failures were reaped and released. None was retried unchanged. The first
physical failure must not be relabeled as a fully successful runtime.

[Published source](../examples/qk_archive) ·
[Acceptance and failure summary](../evidence/qk-archive/acceptance-summary.json) ·
[Raw captures](../evidence/qk-archive/captures) ·
[Source provenance](../evidence/qk-archive/source-map.json)
