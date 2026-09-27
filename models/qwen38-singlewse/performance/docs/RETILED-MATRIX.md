# Retiling the complete original matrix

P18 compares an 8x32 decomposition against P17's 2x128 decomposition of the
same original layer-0 `linear_attn.in_proj_a.weight`, all 48x5120 elements.
The target remains complete physical Qwen3.8-27B inference at >=2000 dependent
tokens/s; this experiment alone cannot establish that target.

The PE application remains a631x2 strip;8x32 denotes the local arithmetic tile,
not a compact two-dimensional PE partition. Hardware routers pipeline/forward
packets; chain length must not be described as software execution at every PE.

The 960 worker coordinates, full resident bank capacities and other background
slots remain fixed. The selected BF16 slot6 receives new tile identities. This
is an explicit component binding; the published full-model atlas is unchanged.
All five BF16 cases use the same 5120 original input codes as P17. FP32 addition
order changes with the partition, so the candidate has a separately frozen
ordered oracle and an independent row-major FP64 matrix-vector reference.

| Property | P17 | P18 candidate |
| --- | ---: | ---: |
| Native rows x K | 2x128 | 8x32 |
| Workers / MACs | 960 / 245760 | 960 / 245760 |
| Ordered reduction trees | 24 x K40 | 6 x K160 |
| Per-worker input packet (u32) | 65 | 17 |
| Source input stream (u32) | 2600 | 2720 |
| Reduction payload (u32) | 3 | 9 |
| Ordered output stream (f32) | 48 | 48 |
| Native/gather/return word-hops | 44520 | 65886 |
| Router entries | 7586 | 7544 |
| Maximum source-input distance | 1260 hops | 1260 hops |

Word-hops measure traffic, not execution time. Faster local arithmetic may lose
its advantage to the larger/deeper reductions and the shared input path. The
controlled physical comparison is required before selecting this decomposition.

The vector join preserves local-left-right FP32 ordering. The native decoded
weight scratch is reused only after the dot completes; two independent child
buffers remain live until both receive callbacks finish. Each root forwards
8-word chunks and keeps only its last chunk. The source retains all48 outputs.
Colors2..19 are explicit; SDK20..23 and unqualified0/1 are not used. Queues,
microthreads and descriptors preserve the prior explicit lease separation.

The supplementary FP8 case now uses all K columns of the first48 original rows
of layer0 `mlp.gate_proj.weight`, with original128x128 block scales. This is a
48x5120 submatrix of the original17408x5120 projection. It is different from
P17's representative FP8 smoke fixture and must not be reported as a matched
FP8 performance comparison. Only the selected FP8 slot0 and its scale change;
remaining background slots retain their original bytes.

Every local vector, every subtree, all participant counts, every returned row,
every input packet, callback and teardown must pass. Replay, all original
physical banks including zero padding, actual compiled SRAM with a4096-byte
stack allowance and normal stop remain required. The simulator verifies all
packet and selected target words against separately uploaded originals, including
one-bit negative controls/restoration. It uploads only BF16 targets; full
background retention and the FP8 case require physical execution.

`tools/audit_retiled_matrix_plan.py` inverses physical coordinates, checks every
new row/column interval, bank capacities and scale-block boundaries, and traverses
the emitted routes independently. `tools/compare_retiled_matrix.py` compares
retained P17/P18 fixtures element by element, including every unchanged background
word. Source-copy provenance is recorded in `probes/retiled_matrix/REUSE.json`.

The bounded simulator is being retained as a shape-control result. The standalone
strip-retile physical dispatch is deferred by the architecture priority change;
no physical speedup is claimed. Mainline development moves to compact physical
2D partitioning/shared dataflow lowering, with P17 as the physical reference.

The preserved simulator001 passed normally in890.840s total. It checks
7680 local values,7680 subtree values,48 returned outputs,16320 input packet
words and122880 target-weight words. Both one-bit negative controls and restores
pass. Maximum SRAM including stack is48096 bytes. The mixed BF16 measured
interval is10242 simulated cycles versus9328 in P17 (9.80% longer).
The independent retained-fixture comparison proves all original BF16 weights and
all five BF16 inputs identical, plus8276160 unchanged background/padding words.
This does not qualify physical background retention or the new FP8 submatrix.
