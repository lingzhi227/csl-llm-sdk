# Four original layers: causal execution and fresh full-state restoration

September 23, 2026 UTC. Original BF16 layers 0/1/2/3 retain their full dimensions.
A 119×1160 application executes positions 0 and 1, followed by device reset and
position 0 replay. Only Layer 0 receives the original external hidden input.
Each downstream layer consumes the actual preceding layer's device output;
all 5120 BF16 words are compared with the consumer's retained pre-normalization
input. There is no host substitution between these four layers.

## Physical graph and resources

The complete artifact contains 138040 application PEs,125220 matrix PEs and
5107 ELF programs. Inspection covers 1791410 coordinate banks. Maximum ordinary
storage plus the unchanged 4096-byte declared stack is 49040 bytes, leaving 112
bytes against actual 49152-byte physical capacity. Twenty-nine violations of
earlier 48128/48640 policy thresholds remain disclosed. The physical-capacity
decision preceded this compile; this is not a dynamic-stack proof.

Original layer parameters occupy 3081379968 prepared-file bytes across 650 arrays;
648 execution arrays account for 3081215872 bytes. Uploads retain at most four
tasks and 64 MiB globally, with 16 MiB per host copy and 16 channels. The actual
largest weight batch is 49668096 bytes. Worker memory is requested at 8 GiB;
actual worker peak and maximum useful IO concurrency are not established.

## Conditional numerical checks

The baseline passes 48084480 matrix-row checks,375660 predetermined exact FMA
samples, all ordered recurrence checks, normalization, attention/KV, MLP,
residual, persistent-state and transport checks. Every operator gate is
conditioned on its actual captured upstream operands. These checks do not
propagate one error enclosure through the whole chain.

| Layer | Nominal BF16 differences at 0 /1 /reset 0 | Maximum absolute difference |
| --- | ---: | ---: |
| 0 | 0 /438 /0 | 0.0009765625 |
| 1 | 204 /2083 /204 | 0.001953125 |
| 2 | 1126 /2553 /1126 | 0.00390625 |
| 3 | 1582 /3054 /1582 | 0.0078125 |

Each vector contains 5120 values. Reset replay is exact. The pinned nominal CPU
reference uses its original chunk-prefill path, whereas these device positions
execute sequential recurrence. Neither nominal bitwise parity nor a whole-model
error bound is claimed.

The original first offline audit stops at Layer 1 because the older preprocessing
domain ends at 1. A separate census checks 288 actual cases:177 are domain-only
failures, zero quantitative gate failures, and the actual range is
[-10.2734375,4.09375]. A new explicit domain version[-48,5] retains the exact
device kernel and 2^-20 error allowance; its conservative bound is below 2.24e-7.
The unfinished layers 1/2/3 then pass. That resumed process fails its final
consumption aggregation because Layer 0 was consumed in the earlier process.
A separate finalizer binds all four exact reports and original parameter files,
without repeating arithmetic or weakening the runtime consumption checker.
Both failed audit attempts remain preserved.

## Fresh checkpoint and independent state comparison

The completed position 0 checkpoint contains 39 semantic banks:
three convolution histories, three DeltaNet recurrent matrices, eight hidden
vectors,24 KV banks and one all-PE semantic control bank. The total native
payload is 14378752 bytes. A distinct physical runtime reloads all original
weights and every checkpoint bank, initializes the fresh local runtime and
computes logical position 1. Full journal, control, handoff, original parameter
and raw-file validation runs only after all physical owners are released.

Every byte of all 39 banks equals uninterrupted position 1. A separate controller
implementation independently performs the same complete byte comparison.
Local runtime serials differ(2 versus 1); transport counters are deliberately
not semantic checkpoint state. This proves finite physical fresh-context
restoration, not full-stage or full-model output equivalence.

## Evidence, monitoring and preserved failures

Baseline 004 has 27364 raw files/783237644 bytes; restored 007 has 9425 raw files/
276357780 bytes. Seven plus three bounded direct streams preserve all 36789
files/1059595424 bytes exactly once on the evidence disk, with source hashes,
destination rereads, read-only files and durable filesystem synchronization.
Small partition inventories bind the exact union. Large captures, original
weights, compiler products and logs are not published in this repository.

The previous restored 006 run completes device work but hits a host capture
progress timeout and does not produce a successful final capture. All 4067
partial raw files and original failure metadata remain preserved. The precise
historical blocking stack was not recorded. Restored 007 separates advisory
progress from durable evidence, batches ordered journal syncs with explicit
durable-prefix/final receipts, and adds anonymous-memory progress plus bounded
stack capture. Injected child stalls and sync failures are checked separately;
this does not prove the historical failure's exact cause.

Restored 007 completes normally, yet its maximum parent-monitor poll gap is
60.4868669 seconds. The 15-second progress policy and 0.1-second polling target
are not hard cancellation guarantees. Future full-stage parent deadline checks
must avoid blocking shared-filesystem operations and still disclose OS
scheduling limits. No arbitrary-filesystem-fault tolerance is claimed.

Earlier 600-second compile timeout, startup failures, two offline audit failures,
restored 006 failure and two pre-payload SSH preservation failures remain
separate immutable attempts. Successful baseline and restored physical jobs,
offline workers and outer owners are reaped; system assignments for those completed jobs are cleared.
Host capture windows are 327.140799 and 242.331481 seconds respectively; they
include host transfers and diagnostics and are not wafer/token latency, speedup,
matched performance or official billing measurements.

## Reproduction scope and remaining work

The published programs are exact selected historical sources, accompanied by
source maps and evidence pins. Original machine paths, live admissions, weights,
raw data and the 8 MB generated runtime host-plan are excluded. A new deployment
requires regenerated geometry/bindings, qualified original parameters and
inputs, a newly frozen manifest and its own resource admission. Do not remove
historical gates to turn these archives into an unreviewed launcher.

The next target is all 64 original layers with final normalization and all 248320
logits, context 8 five-token prefill and dependent decoding across 20/24/20 stages
on one physical CS3. Those stages and the full head are not yet compiled,
physically qualified or accepted. Full-model CSL neural epochs remain zero.

[Exact selected source](../examples/four_layer_chain) ·
[Scoped evidence](../evidence/four-layer-chain/qualification.json).
