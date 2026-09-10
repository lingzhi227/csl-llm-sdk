# Full-head DeltaNet recurrence

This example implements one synthetic value head with key and value dimensions
128. It accepts already normalized Q/K, Q already scaled by 1/sqrt(128), V,
beta and decay. Convolution, normalization, gate generation, gated output RMS,
48-head composition and original-weight model execution remain outside this
qualification boundary.

For each token, the device computes D=decay*S, p=k^T D,
delta=beta*(v-p), S=D+k*delta^T, and y=q^T S. The output uses the updated
state; prediction uses the decayed state. Each of two adjacent PEs owns 64 key
rows and all 128 value columns, storing 32 KiB of persistent FP32 state.

## Arithmetic and evidence

Decay is a separate FP32 multiply. Prediction and output each use 64 ordered
FP32 FMAs per PE followed by one FP32 addition on PE0. Delta uses subtraction
then multiplication; state update uses one FMA. The independent FP64 oracle
propagates previous-state error through decay, both reductions and the update.
Per-element bounds are frozen before execution; they include gamma(64)
reduction bounds and FP32 rounding at the declared intermediate stages.

The CPU fixture preparation also runs the unchanged extracted official recurrent
function body at pinned Transformers commit
`4815a0a6a064214f2d8208c094464a5a6b76ca8d`. Hub decorators alone are removed;
this is not a complete Transformers runtime. The official recurrence has
normalization disabled at this boundary. Its actual FP32 Q scaling and
exp(log(decay)) results become candidate inputs; any discrepancy from requested
decay is recorded. The independent FP64 recurrence remains the CSL acceptance
oracle. Official-body agreement is a separate comparison.

Four full-state observations cover two dependent tokens from nonzero initial
state, a new generation reset to zero followed by nonzero input, and another
generation reset followed by zero Q/K/V. Token two uses beta=0 with decay=0.5;
the first reset uses decay=1 so forgetting cannot hide a broken reset. The host
uploads initial state once and thereafter only observes device-owned state.

## Serialized transport

Each message has 131 raw 32-bit words: generation, token and message phase,
then 128 FP32 body words. PE1 sends prediction on color2; PE0 sums prediction
and sends delta on color3; both update local state and compute local output;
PE1 sends its output contribution on color2. PE0 computes global output.
Every receive consumes and checks the header only after the entire packet
arrives. The packet identity, phase counts, commit counts, event order, guards
and immutable inputs are read back after every token.

Each PE permits one in-flight fabric operation. Transport DSR5 and microthread2
remain leased until the local completion callback. Synchronous arithmetic uses
DSR3/4. PE0 reuses its send body for local output only after the delta send has
completed and all local state updates have consumed delta. The global delta
observation therefore comes from PE1's retained phase2 receive buffer. Root
prediction and final output have separate guarded storage. Full output packets
are compared bit-for-bit between sender and receiver.

The host waits for both command completions before starting another token or
reset. This is a host-observed successful completion boundary, not a distributed
atomic transaction. Wrong identities and phases are rejected, but coordinated
error draining, retransmission and replay recovery are not qualified.

## Resource admission

The compile wrapper inventories both application ELF files and requires each
highest application section end plus 4096 bytes of declared stack allowance to
fit 49,152 bytes. Known high-address configuration sections add no application
SRAM capacity. Simulation starts only after both checks pass. The allowance is
a conservative admission rule, not a measured dynamic stack peak.

Execution is serial, limited to 20 GiB RAM, zero swap, one simulator thread,
300 seconds compilation and 300 seconds simulation. The guard also enforces
8 GiB available RAM, 32 GiB free disk and 20 GiB project cache reserves/limits.
Four mandatory complete state readbacks dominate estimated host traffic below
110,000 words. A 220-second planning allowance is deliberately conservative
relative to the prior RMS example; it is not a bandwidth guarantee. No model
checkpoint, new environment, GPU workload or SDK image download is needed.
