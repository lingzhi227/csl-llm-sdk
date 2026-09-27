# Frozen measurement scope v1

Model: Qwen/Qwen3.8-27B-FP8, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, all 64 text layers, original untied
248,320-row head, original weights and state update semantics, batch one,
greedy selection with lowest token ID on exact ties. No speculative decoding,
layer omission, host neural computation or reduced vocabulary is included.

Report initialization separately (artifact load, checkpoint transfer, validation),
then prefill / time to first output and dependent decode. Count the N-1 dependent
intervals after the first output, not N divided by those intervals. Report EOS
inclusion explicitly. Report host-observed output rate and device cycle rate
separately; the final target requires at least 2,000 tokens/s of complete dependent
decode at batch one, and an actual host-observed complete sentence run. Do not
subtract communication or host work from an end-to-end observation.

For full-model acceptance use fixed prompts selected before timing, generate
through EOS with a finite context / token cap, capture actual token IDs and text,
verify reset/replay, all operator/state epochs, full weight/source provenance and
numerical qualification appropriate to changed operators. Fluency alone is not
numerical qualification. Bitwise equivalence to a different floating-point
schedule is not required; changed reduction orders require explicit independent
error bounds and output checks. Preserve the failed historical strict contract.

For microbenchmarks report precise scope, PE rectangle, payload, iterations,
same-PE 48-bit start/end cycle difference, host wall time, correctness checks,
actual ELF SRAM, clock-source evidence (or cycles only), job IDs and release.
Do not compare timestamps from different PEs without synchronization calibration.
No assumed nominal frequency is silently converted to measured microseconds.

Each hardware experiment has a fresh immutable source manifest, single-owner
hardware lock, account-active-job gate, runtime deadline and cleanup in finally.
Queue wait and compilation must be distinguished from WSE allocation time.
Record actual system assignment and job lifetime as resource evidence; these are
not a claim to know the provider's final billing calculation. Closing SSH is not
resource release. Verify terminal owned jobs and no remaining owned assignment.
