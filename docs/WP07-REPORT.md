# WP07 — BF16 gated RMS and device recurrence composition

Both authorized SDK candidates passed: standalone direct-gain gated RMS128,
then the same kernel consuming the full128×128 recurrent head's output on a
third simulated WSE-3 PE. The composed path keeps recurrent state in FP32 and
performs input BF16 conversion, normalization, gain, SiLU and final BF16
conversion entirely on device before root token completion.

## Standalone qualification

Four warm calls passed all independent mathematical and exact-conversion gates.
They cover zero input, gain0, signed gains, gate values from-16 to16 and an
early-versus-late-cast counterexample. All normalized-value, gain-product and
final-output BF16 conversions matched the independent RNE oracle applied to
actual preceding FP32 bits. Sixteen signed conversion probes passed each call.

The largest exp error consumed0.52671 of its2^-20 relative allowance; SiLU
consumed0.13340 of its2^-18 allowance. Normalized-value mathematical error
consumed at most0.006869 of its propagated bound. All direct product, sqrt,
inverse, zero-annihilation, input immutability, guard and call-count checks
passed. Mathematical-oracle BF16 output mismatch counts were zero for all four
fixtures; parity is reported separately from the exact conversion gates.

The extracted official CPU class also passed four independent reference
comparisons and verified BF16 gain-product/FP32 final-product dtype promotion.
Reference001 failed before arithmetic because of the optional hub-dispatch
decorator. Reference002 removed that decorator alone, retained class/method
bodies, and passed in2.077336 seconds with observed206,712,832 bytes. Its9 frozen
files remained unchanged, and the owned unit was absent afterward. No packages
were installed. See the [precision contract](WP07-DESIGN.md).

## Three-PE composition

Four token updates across three request generations passed all complete
prediction, delta,16384-element state and128-element output gates from WP06.
The consumer's BF16 input bits matched RNE of the actual received FP32 y, and
the complete received frame matched root's retained global output bit-for-bit.
All gated RMS stage/conversion checks passed. The last reset/zero-input token
produced exact numerical zero state and output.

Composition reuses the standalone gain/gate vectors; its input is always the
device recurrence output, replacing the standalone fixture's input vector.
The normalization oracle checks the actual BF16 boundary value; recurrence
error is independently checked before that boundary. Mathematical BF16 output
parity again had zero mismatches in these cases, without being an acceptance
gate. Maximum normalized-value error consumed0.03951 of its propagated bound.

Every token passed all three PE event orders, generation/token/commit counts,
the consumer-arm preparation barrier, ACK-before-root-commit ordering and exactly
one inference unblock per PE. Root's final receive storage was correctly
observed as an ACK header plus a retained older output body and checked in two
parts. PE1's transit routes stayed active across its earlier local completion
and every request reset. All original inputs, stage guards and handles passed.
Only the observed successful arrival order is simulator-qualified; the alternate
command/completion arrival order has not been separately exercised on device.
This is a successful serialized protocol; error draining/replay recovery and
distributed atomic commit remain unqualified.

## Resource receipts

| Candidate | Compile seconds | Simulation seconds | Observed compile / sim bytes | Run allocated bytes |
|---|---:|---:|---:|---:|
| 001 standalone | 4.108657 | 10.241344 | 461,828,096 / 185,135,104 | 499,712 |
| 002 composed | 5.126167 | 96.114617 | 463,114,240 / 212,348,928 | 8,810,496 |

Memory values are observed live-cgroup maxima, not guaranteed lifetime peaks.
The unchanged limits were20 GiB RAM, zero swap, one heavy task,8 GiB available
RAM and32 GiB free-disk reserves. Compilation deadlines were300 seconds;
standalone/composed simulation deadlines were180/300 seconds respectively.

| Candidate / PE | Application section end +4096 stack allowance | Ceiling | Margin |
|---|---:|---:|---:|
| 001 / 0 | 14,320 | 49,152 | 34,832 |
| 002 / 0 | 47,216 | 49,152 | 1,936 |
| 002 / 1 | 46,432 | 49,152 | 2,720 |
| 002 / 2 | 16,320 | 49,152 | 32,832 |

Every SRAM gate passed before simulation. Both runtimes stopped normally, and
all owned units were absent/inactive with PID0 and empty cgroups. Candidate001
retained18 source/input and11 compiled hashes; candidate002 retained30 and17.
All matched after execution. Forty-four lightweight host tests pass. Re-running
the preparation tools in temporary directories reproduced the executed frozen
files except location-dependent launch paths, without re-running experiments.
Complete development receipts retain stage arrays; public
[evidence](../evidence/wp07.json) contains sanitized metrics and identities.

## Reproduction and next work

Use existing qualified CPU PyTorch and SDK installations and pinned WP04 sources:

```sh
python3 tools/prepare_wp07_reference.py --run /path/to/cache/wp07-reference-new \
  --source-dir /path/to/pinned-wp04-sources --python /path/to/existing-torch-python
python3 /path/to/cache/wp07-reference-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp07-reference-new \
  --spec /path/to/cache/wp07-reference-new/steps.json --profile cpu-reference
python3 tools/prepare_wp07.py --run /path/to/cache/wp07-standalone-new \
  --reference /path/to/cache/wp07-reference-new --image /path/to/installed-sdk.sif
python3 /path/to/cache/wp07-standalone-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp07-standalone-new \
  --spec /path/to/cache/wp07-standalone-new/steps.json
python3 tools/prepare_wp07_composed.py --run /path/to/cache/wp07-composed-new \
  --reference /path/to/completed-wp06-reference \
  --norm-reference /path/to/cache/wp07-reference-new \
  --standalone /path/to/cache/wp07-standalone-new --image /path/to/installed-sdk.sif
python3 /path/to/cache/wp07-composed-new/tools/guarded_run.py \
  --cache /path/to/cache --work /path/to/cache/wp07-composed-new \
  --spec /path/to/cache/wp07-composed-new/steps.json
```

The next boundaries are input convolution, Q/K normalization and gate generation,
followed by their composition with recurrence. All48 heads, original-weight full
layers, full attention/KV and model generation remain separate work. Neither
candidate is a physical-wafer performance measurement or a complete model layer.
