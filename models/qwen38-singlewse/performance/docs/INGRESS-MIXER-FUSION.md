# Original A/B projection fusion at operand ingress

P34 changes when the original local A/B contractions run. Their BF16 inputs
are consumed directly from the low halves of the arriving133-word tagged packet,
before the packet arena becomes the FP8 weight decoder. Each original PE owns
at most one128-column tile for each of A and B. The lowering audits this bound
and identical QKV/A/B input-group ownership over all3840 original BF16 tiles.
Only two FP32 local partials remain live for the later contraction reductions.
The256-byte persistent raw-input array is removed; the FP8 native input cache,
original matrix addresses, FP32 FMA order and complete-K reduction are preserved.

## Actual arithmetic evidence

`mixer-ingress-sim-001` executes the actual new CSL producer and native consumer,
compares them with the frozen P33 native consumer, and checks a separate NumPy
nearest-level quantizer and ordered-FP32 reference. It uses original pinned
layer0 QKV/Z/A/B/output weight slices from three actual owners: both A/B present,
A only and B only. All24 cases pass, including zero after nonzero, changed input,
replay and both40/48-group input families. Packet words, native encodings,
outputs, banks, inputs, descriptors and counters pass; the simulator stops
normally after32.671295314s including preparation/compile/runtime.

The test deliberately runs QKV and Z before retrieving the cached A/B partials,
so the shared packet/weight-decoder arena has already been overwritten. This
checks the intended shortened input lifetime. It is a one-PE arithmetic test,
not execution of fabric broadcasts, complete reductions, a full mixer or model.
The weight payload is an explicitly documented original sample, not full banks.
No local timing result is used as a model-throughput claim.

## Compiled storage and remaining placement work

Selected resource compile013 failed on a scalar-pointer type mismatch in the
new A/B call. Its source/log/failure/release evidence remains frozen. Corrected
014 compiles all nine selected actual cohost programs with the same4096-word
calibration banks and parameters as012. The eight weight-bearing programs each
shrink112bytes; the shared producer/controller remains39632bytes including the
unchanged4096-byte stack reserve. This is less than the removed248 net data bytes
because the new scheduling/control code also occupies SRAM.

The matched measurements are in`mixer-ingress-calibration-001.json`. Replacing
the calibration bank size arithmetically with only the original matrix bytes
leaves a60-byte estimated margin on the selected nonroot standby, but exceeds
the48128-byte gate by164bytes on the selected root and3724–4796bytes on selected
norm cohosts. These are estimates, not full-bank compilation results. They show
why moving only request-state pages is not a sufficient planning strategy for
these cohosts: matrix ownership, program cohosting and scratch lifetimes also
need to be revised together. Do not reduce stack reserve or discard weights,
state pages, requests or original operations to force admission.

A full-stage compile was not repeated while the selected footprints still
show placement deficits. Prior015 remains the latest complete-layout attempt;
P34 has no complete
SRAM admission. The next lowering must use measured program demands to allocate
matrix work, norm roles and original state together, regenerate their paths,
and pass complete SRAM plus connected numerical tests before hardware use.
Conv/gates/GDN, full attention, automatic stage handoff and all64-layer sentence
and>=2000 aggregate completed-token/s measurement remain unfinished.

## Resource and publication scope

Two bounded selected compilations and one bounded simulator service were used.
All three are terminal; their cgroups/processes are empty and the shared heavy
lock is free. No WSE job was submitted. The fresh ALCF accounting snapshot has
no owned active job or assigned system. It does not expose provider billed hours.
Original functional and P31 physical baselines are preserved.

All155 source regression tests and all155 site-adapted publication regression
tests pass. Their receipts bind the tested source/export identities separately.
