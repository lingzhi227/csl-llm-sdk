# Physical Stage0 execution and exactly 1,000 comparisons

September 24, 2026 UTC. Original Qwen3.8-27B layers 0–19 completed positions 0–4
on one physical CS3 with original BF16 parameters and dimensions. Each layer
consumed the preceding layer's actual output. The prompt token IDs were 760,
6511, 314, 9338 and 369. This is prefill; no dependent output token is claimed.

## Exactly 1,000 values

The user explicitly replaced the exhaustive audit with a limit of 1,000
comparisons. The earlier audit was stopped through its original owner and its
partial evidence was preserved. The replacement compares exactly 1,000 distinct
(layer, position, dimension) coordinates against the pinned independent reference.
No other matrix, attention, DeltaNet, convolution, RMS or control values were
numerically rechecked by this comparison job.

Layer 19 contributes 100 values per position, for 500 final-output values.
Each preceding layer contributes five values per position; layers 0, 3, 7, 11
and 15 contribute one more per position, for another 500. Final-output rows
retain 20 fixed uniformly spaced dimensions including both endpoints; the rest
are filled by descending reference magnitude. Intermediate rows retain both
endpoints and fill the rest by reference magnitude. Ties use ascending dimension.
Reference NaNs rank first, then infinities. The complete selection was sealed
before any actual hidden payload was opened; it does not depend on actual error.

Of these 1,000 values, **480 match exactly in BF16 bits**. The maximum finite
absolute difference is **0.5**; **0 pairs contain a nonfinite value**.
The [full comparison](../evidence/sequential-stages/stage0-1000-comparisons.json)
lists every selected value and difference. Relative error is absolute error
divided by the absolute reference when both values are finite and the reference
is nonzero. Two zeros give zero relative error. A nonzero actual value with a
zero reference, or any nonfinite operand, gives null with an explicit flag.

No new error threshold was imposed or fitted to the results. This report makes
**no overall numerical-pass claim**, complete operator-coverage claim or propagated
whole-model error bound. The [selection record](../evidence/sequential-stages/stage0-selection.json)
and source hashes make the limited coverage explicit.

## The 500 values after all 20 layers

The user additionally requested a separate summary of the already selected
500 final-output values. This summary reuses the existing records and adds
zero actual-versus-reference comparisons. Each of the five positions has
100 selected values.

| Metric | Final-output values |
| --- | ---: |
| Exact BF16 matches | 182 / 500 |
| Relative difference at most 1% | 398 / 500 |
| Mean absolute difference | 0.01276074219 |
| Median relative difference | 0.5090% |
| Maximum absolute difference | 0.5 |
| Nonfinite pairs | 0 |

The two largest absolute differences occur at dimension 3994, positions 1
and 2: reference 59.75 versus actual 60.25, and reference 58.5 versus actual
59.0. Relative differences are about 0.84% and 0.85%. The largest relative
difference is 100%, at position 2, dimension 3502: the small reference value
0.00634765625 versus actual zero. The 1% count is descriptive, not an
acceptance threshold. Selection emphasizes large reference magnitudes and
fixed dimensions; it is not a random sample or a whole-output error bound.

## Physical artifact and preserved failures

The corrected artifact covers 24,743 programs and 690,200 PEs, including
626,100 matrix PEs. Minimum static SRAM margin is 48 bytes including the declared
4,096-byte stack. All 149 earlier family-ceiling failures remain recorded;
dynamic stack peaks are not measured. Actual host-copy binding checked 45,314
planned transfers and 1,598,438 program-bank associations before the run.

The previous full-stage attempt stopped at a legacy observer publication
boundary during the fourth position. Its evidence is retained. Corrected
context-eight limits are 8,457 attention and 29,849 linear publications, each
including one sticky error; corresponding host wire limits are 16,914 and
59,698. This run executes only five positions without resetting counters.
A separate observer comparison and its timeout remain scoped observations,
not proof of a hardware assertion trap. The first offline audit's process-limit
failure and the user-stopped exhaustive audit also remain preserved. A final
preservation validation first stopped on an incorrectly supplied metadata
receipt pin; the original failure was retained, and the corrected validation
bound the actual destination copy receipt without recopying data.

## Saved checkpoint and pause

The completed physical capture contains a checkpoint at next position 5,
generation 1: 192 native bank files / 71,893,760 bytes, including hidden, KV,
DeltaNet, convolution and semantic control state. Five actual final output
rows contain 5 × 5,120 BF16 values / 51,200 bytes. The recovery copy preserves
204 files / 72,764,883 bytes on Mass1 as independent read-only files.
Source/destination checksums, file identity and byte extents were verified.
Preservation performed zero neural-value comparisons and did not invoke Restore.

The complete 217,762-file / 6,149,555,380-byte raw capture, full journal and
compiler evidence remain on persistent ALCF storage. They are not duplicated
in the smaller recovery bundle or included in this public delta. All physical,
comparison and preservation owners were reaped and independently checked
released; no owned device assignment remains. Development pauses here.

## Timing scope and remaining work

The physical owner took 4941.660 seconds and instrumented capture
took 3223.184 seconds. The run used 253,897 copies,
52,474,566,040 host bytes and 11 launches. Matrix weight upload used 4,110 copies
in 1,030 batches with 16 channels and at most four tasks. These figures include
initialization, readback and diagnostics; they do not measure pure inference
latency, pure link bandwidth or a matched performance improvement.

Original layers 20–63, final normalization, the complete 248,320-word head,
dependent token generation, full numerical qualification and a physical restore
of this complete Stage0 checkpoint remain open. Earlier four-layer results retain
their own scope. Selected source components are byte-identical archives, not a
portable launch bundle; weights, raw arrays and vendor SDK binaries are excluded.
