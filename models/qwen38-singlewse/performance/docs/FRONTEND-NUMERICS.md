# P43: original frontend physical numerical qualification

This work exercises the P42 frontend's actual native32 arrival task and exact
arithmetic with original layer00 parameters. It is a component qualification,
not a complete layer, recurrent core, conversation or model-speed result.
Projection values are supplied by the independently qualified original-matrix
reference. Recurrent core values are frozen FP64-reference injections solely to
isolate post-core gated RMS normalization. Neither injection is permitted in
complete-model acceptance.

The fixture includes all16 shared-Q/K groups and48 V heads. Every original
10240-channel four-tap convolution weight, all48 A_log/dt_bias pairs, and the
original128-element gated-normalization gain are read from the pinned checkpoint.
The six-position input sequence uses mixed/change/zero/replay cases, with two
positions repeated after an explicit reset. Three history samples per channel
persist between positions; the independent oracle verifies their exact BF16 bits.
It freezes FP64/BF16 error intervals before candidate observations. Maximum
packet bound/reference norm ratio is0.002322; maximum gated interval width/midpoint
norm ratio is0.000575, both below the predeclared quality gates. These bounds do
not permit arbitrary token divergence in full-model qualification.

The standalone32-PE harness comprises16 senders and16 consumers. It preserves
all multicast copies from the complete network, including foreign rows, and
alternates producer order to exercise gates/Z before QKV. It verifies shared
Q/K packet bits, three separate packet/output leases, exact traffic counts,
parameter retention and reset replay. IQ1 becomes IQ2 only in the SDK harness,
because SDK memcpy owns IQ1. The production full-stage receiver still uses IQ1.
A harness-only observer releases the SDK command stream on actual downstream
frame completion. The decoder and neural arithmetic are unchanged.

## Physical result

`mixer-frontend-hw-001` passes the full six-position plus two-position reset
replay profile on one physical WSE-3. All16 groups/48 heads satisfy the frozen
packet and gated-normalization intervals. It checks148224 FP32 packet values,
49152 BF16 gated outputs and245760 exact BF16 history samples, and retains all
43104 parameter-cache BF16 words. Shared Q/K packet bits and reset replay are
exact. All259512 native packets are counted through complete downstream drains.
Maximum packet absolute error is1.279003e-7; acceptance is per-element against
the frozen intervals, not against this observed maximum.

Physical compilation covers32 PEs with19 ELF images, maximum33888bytes including
4096 stack. This is the standalone harness, not a new full-bank SRAM admission;
the P42 whole-stage census remains the full-bank evidence. The original frontend
module and packet decoder arithmetic are unchanged. Both cluster jobs succeed
and release normally; the closing accounting snapshot has no owned active job or
system assignment. The71.48-second supervised runtime includes loading, SDK
transfers and validation. It is not model inference latency or a TPS measurement.

## Preserved attempts

| Attempt | Observation |
| --- | --- |
| simulator001 | Compiler rejected SDK/native IQ1 collision; no runtime. |
| simulator002/003 | Compiled, but host polling did not observe the complete projected count before its deadline. No numerical acceptance. Source audit captured later showed all sends completed; that is not a simultaneous downstream observation. |
| simulator004 | Assertion after removing the consumer initialization read barrier. No numerical acceptance. |
| simulator005 | With consumer readiness and paced batches, all16 consumers receive exactly256 frames; source audit agrees. Normal stop and resource release. Transport diagnostic only. |
| simulator006 | Added downstream completion observer. Hit the300-second simulator bound without a numerical acceptance record; all processes released. |
| simulator007 | Single-position numerical smoke passes original packet/gated intervals, exact history/shared QK/drain and normal stop. Omits initial bulk-zero and final parameter-retention copies; these pass in physical001. Same2GiB/noSwap/two-CPU cap;600-second runtime bound. |
| physical001 | Full six positions plus two reset-replay positions pass all numerical/history/retention/lease/drain checks. Both jobs succeed and release normally. |

The standalone full physical profile remains six positions plus two reset-replay
positions. It cannot silently select smoke mode on hardware. Hardware dispatch
requires the shared site lock, no active account job, exact fixture/source
identity, a bounded32-PE artifact and fresh SRAM admission. Final provider billing
is not inferred from SSH connection state; release is checked against actual
owned jobs and system assignments.

Forty selected source tests pass, including six new temporal/reference and
pre-allocation geometry checks. All40 selected publication tests also pass; this is not a full regression-suite run. Original inference source and all prior evidence remain
preserved.


All seven workstation attempts are terminal with empty process groups and the
shared heavy-work lock free. Two new cluster jobs include one actual WSE runtime.
Raw arrays/ELFs stay on remote storage; capture-identity.json binds their sizes
and hashes. Source snapshots, original fixture identity and every failed attempt
are retained. See frontend-numeric-summary-001.json, frontend-numeric-tests-002.json,
frontend-numeric-publication-tests-001.json and resource-audit-p43-close.json under
evidence/.

Next, connect actual recurrent state workers and return traffic to these
qualified frontends, then gated output projection and the retained norm/MLP
graph. Keep the P42 complete bank/route placement as the integration base. Do not
repeat standalone frontend validation or treat injected core results as a GDN
pass. Full64-layer context-preserving dialogue and2000 average output tokens/s
remain unachieved.
