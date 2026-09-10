# WP02 reduced chain and ownership contract

Two adjacent WSE-3 PEs share the already downloaded original 128×112 tile by
contraction columns: PE0 owns 0:56, PE1 owns 56:112. Each uses C01 BF16 expansion
and 56 FP32 FMA updates. PE1 sends 128 FP32 partials to PE0 over a single static
color-2 route. PE0 adds its local partial, then computes RMS normalization of
those 128 values using epsilon rounded to FP32 from 1e-6 and unit gain. This is
an operator fixture, not Qwen's full learned hidden5120 normalization.

## Resource and completion boundaries

The frozen resource-ledger.json is generated from core/qwen38/resource_ledger.py.
The inspected default SDK2.10.1 memcpy provider reserves colors20–23, queues0/1
and its listed local/control tasks. The project reserves color2, rank0 input
queue2, rank1 output queue2, local completion tasks10/11 and microthread2. Bank
and index both matter: GEMV uses dest/src0 index3 and dest/src0/src1 index4;
communication owns dest/src1 index5. Compiler-generated temporaries and imported
math/memcpy code are not declared free by the explicit ledger.

UT2 is explicitly attached to the fabric operand's DSR load, following the
[SDK microthread selection rules](https://sdk.cerebras.ai/csl/language/microthreads_wse3).
This fixture has one asynchronous stream per PE and no overlapping neural operation
while its communication microthread is active. Runtime framing is fixed at 128
words and only one invocation is in flight. No inter-wafer route is implied.

PE1 keeps its partial stable through the send-complete callback. PE0 owns its
receive buffer until the sum has consumed it, and keeps the sum through normalization
and readback. Sender host-unblock means local send completion only. Root host-unblock
is after both partials, the sum and normalization. The host waits for the command
on both PEs before submitting the next input, so the next call cannot overwrite
in-flight data. The fixture does not implement overlapping request slots.

Mode0 computes the local GEMV before arming receive. Mode1 first arms receive and
starts local GEMV only inside the actual receive-complete callback. Thus a mode1
arm event is never mislabeled remote completion. The exported sequence slots are:

| Slot | Meaning |
|---:|---|
| 0 | Persistent call counter |
| 1 | Per-call event sequence counter |
| 2 / 3 | Local GEMV completion / remote receive completion |
| 4 / 5 | SUM commit / normalization commit |
| 6 / 7 | Host unblock point / sender completion |
| 8 / 9 / 10 | Receive / commit / unblock counts |
| 11 | Error code |
| 12 / 13 | Receive arm / send issue |

Both root completion orders must precede SUM, then RMS, then exactly one root
unblock. Sender completion must precede its unblock. Every new call reloads
fabric extents and GEMV descriptors, zeroes per-call flags/events and local
GEMV scratch/results. Static guard bytes and exported handles remain stable.
This is bounded warm-call reinitialization, not full inference-state reset.

## Predeclared numerical propagation

Let u=2^-24, gamma(n)=n*u/(1-n*u). Each independent 56-term FP64 reference partial
has bound b=gamma(56)*sum(abs(w*x))+56*2^-126. For the sum s=p0+p1, propagate
B=b0+b1+u*(abs(p0)+abs(p1)+b0+b1)+2^-126.

For S=sum(s_i²), first bound input perturbation by sum(2*abs(s_i)*B_i+B_i²).
Add gamma(256)*sum((abs(s_i)+B_i)²)+256*2^-126 for FP32 product/accumulation.
Divide this bound by128 and account for the epsilon addition to bound Q=S/128+eps.
Use max(eps*(1-u), Q-B_Q) as a positive lower bound for the denominator argument.

The actual SDK math.sqrt_f32 result is independently compared to sqrt(actual_Q)
with a frozen relative budget 2^-20. The actual reciprocal is independently checked
against 1/actual_sqrt using the same 2^-20 budget. These are qualification gates,
not undocumented vendor accuracy promises. The final normalized bound combines
projection error divided by sqrt(Q_low), inverse-square-root perturbation
abs(s)*B_Q/[sqrt(Q_low)*sqrt(Q)*(sqrt(Q_low)+sqrt(Q))], those two independently
checked approximation budgets, and the final FP32 multiply. The zero-after-nonzero
case additionally requires exact numerical zeros at both partials, sum and output.

The host exports and checks both partials, received bits, sum, normalized output
and all four scalar diagnostics. A final-output pass cannot override a failed
intermediate, protocol, approximation or lifecycle gate. None of these thresholds
may be changed in response to candidate errors.
