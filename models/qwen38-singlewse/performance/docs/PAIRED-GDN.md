# Paired recurrent transport and numerical qualification

The target remains a complete original Qwen3.8-27B-FP8 conversation on one
physical WSE-3, with retained context across turns and at least 2000 average
assistant output tokens/s. This component does not establish that result.

The producer adapter changes only the original frontend packet stores into
`[token, decay, beta, V128, K128, Q128]`. The worker receives V, then reuses the
512-byte decoder for K and Q. Its output is the existing five-word frontend
frame `[token, global_value_row, 2, packed_BF16, 5]`. Even starting rows and widths
are checked at compile time. Two values per frame halve the scalar packet count;
this is a traffic count, not a measured speedup. The adapter alone does not
install production routes or resolve the launch/return retirement handshake.

The diagnostic preserves all 753 P45 value slices: 748 workers own eight columns
and five own 32. Together they cover all 48 heads and 786,432 FP32 state elements.
Complete simulator compilation covers 1,506 source/worker PEs in 758 ELF images,
with maximum 26,992 bytes including 4 KiB stack. Diagnostic state banks are not
the full resident model banks; P45 remains the complete-bank admission.

Attempt001's reserved SDK color conflict is preserved. Attempt002 compiles the
complete diagnostic but its simulator was intentionally stopped after several
minutes before the first position. Its artifact and linked census remain.
Attempt003 uses the identical worker/math sources on three original 8/32-column
slices: all 6,144 state elements, 24 paired frames, numerical intervals and exact
transport counts pass, with normal stop. Physical001 subsequently covers every original head and value slice.

## Numerical contract

The frozen input derives from the independent original-parameter frontend
reference used by P43. It is rounded to FP32 before recurrence. The actual
frontend is not running in this harness. The reference calculates the original
FP64 equations, including V minus the completed K/state dot product. A priori
FP32 error bounds cover decay, the candidate's reassociated 128-term residual,
beta multiplication, state correction and the Q/state reduction. State error
propagates across all preceding positions. No candidate output tunes a bound.

Every FP32 state element must fall within its interval, and every BF16 output
must lie between the independently rounded interval endpoints. The state-bound
norm ratio is capped at 0.002; the frozen six-position fixture's maximum is
0.000065025. Six continuous positions and two explicit reset-replay positions
are required on physical hardware. Full input send and complete downstream
return receive both finish before a source exchange retires; worker local-send
idle is checked separately. These do not substitute for a whole-stage fence.

## Physical result

Physical001 passes all six continuous and two reset-replay positions: 6,291,456
FP32 state observations, 49,152 BF16 outputs and 24,576 native paired frames.
A fresh bounded capture audit verifies bitwise FP32 and BF16 reset replay.
Maximum state relative L2 error is 3.201416e-7 and the largest per-element error
is 0.164656 of its frozen bound. The complete 1,506-PE/758-image physical census
has maximum 26,992 bytes including 4 KiB stack.

Both invocation-correlated cluster jobs succeeded and released; all three
workstation services released. The final accounting snapshot has no owned
active jobs or hardware assignments. The 236.494s initialization and 240.282s
complete diagnostic driver duration include cluster preparation, host transfer,
full state capture and verification; they do not measure neural device latency.
Twenty-seven selected source regressions and 27 publication regressions pass,
with four targeted checks after separating pure source construction from local
dispatch metadata. The failed publication test remains recorded. See
`evidence/gdn-paired-summary-001.json` and the physical capture audit.

## Next connected graph

Use the qualified paired frame at the original frontend, then connect actual
frontend broadcast, state-worker admission, return merging and downstream
retirement. Do not treat final producer DMA completion as consumer completion.
The frontend currently ties its consumed bit and source-buffer release together;
launch eligibility and packet reuse must be ordered explicitly when callbacks
and early returns can interleave. After that, qualify the original output
projection and changed norm/MLP graph, then all 64 layers and real token feedback.
