# WP08 candidate001 diagnosis and candidate002 hypothesis

Candidate001 completed all eight tokens and normal SDK shutdown, then failed
numerical acceptance. History, reset, counters, immutable parameters/weights,
guards and BF16 conversion checks passed. Two independent problems remain in
that preserved candidate.

First, beta matched sigmoid of the last projected input x[383] rather than b.
For example, token1 requested b=-8 but beta's early BF16 store was0.5; token3
requested b=0 but stored0.3203125, matching the BF16 result for x[383]=-0.75.
The later FP32 diagnostics agreed with that early BF16 store. This rules out
an explanation limited to final diagnostic values being overwritten: beta
itself was already wrong before the later gate helper calls. Input/parameter
readbacks were correct. Symbol inspection places input at0x3528 and parameters
at0x382c, with no overlapping global arrays. The cause within the compiled
computation/input-source path has not been established.

The original run did not save SSA or instruction traces. GNU objdump cannot
disassemble this architecture, and the available cs-readelf supplies symbols
and memory information. No extra compiler/simulator attempt was made merely
to obtain diagnostics. This evidence does not establish a compiler bug.

Candidate002 separates vector/history work and scalar gates into two serial
device commands. The vector command enters a pending-gates phase; finalize
reads four fixed global BF16 offsets into observable FP32 slots, verifies the
same generation/token and performs the scalar gates before committing. The
host only launches and observes. Parameters are read back between commands
and after completion; the four consumed FP32 values must exactly match input.
The hypothesis is that eliminating gate-source lifetimes across the long vector
path and materializing a new command's inputs prevents the observed wrong-source
behavior. Alternate transitions cannot advance history or commit twice.

Second, the installed exp function slightly exceeded the already-frozen2^-20
relative allowance at several convolution arguments: the largest observed ratio
was1.09384. The acceptance threshold and all eight fixtures remain unchanged.
Candidate002 replaces exponential evaluation within the actual domain[-24,1].

The replacement selects integer n, recomputes r=x-n*ln2 using high/low split
constants, evaluates the degree8 Taylor polynomial by FP32 Horner arithmetic,
and scales by exact normal2^n. The selection loop's rounding margin allows
abs(r)<=0.3467 and -35<=n<=1 in the declared domain. The high-part product is
exact for every such integer. Remaining reduction error is bounded by about
2.068e-8. Polynomial truncation is bounded by exp(R)*R^9/9!, approximately
2.821e-10 for R=0.3467. Conservative propagation of coefficient, multiplication
and addition rounding gives about1.723e-7 absolute polynomial error and a
combined relative bound below2.440e-7, versus the unchanged2^-20 allowance.
Normal power-of-two scaling does not introduce a new mantissa rounding step.

The independent accounting is in `test_exp_bounded.py`. A1025-point FP32 grid,
extremes and neighboring values around reduction boundaries provide additional
test coverage, not a proof of all device behavior. Candidate observations also
check that every actual exponent argument remains in[-24,1]. Device acceptance
still requires the original independent actual-argument gates.
