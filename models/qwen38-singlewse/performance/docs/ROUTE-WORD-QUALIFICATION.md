# Physical compact route-word and queue transition qualification

P14's `route-epoch-hw-001` qualifies the local route-word backend and a restricted
autonomous transition protocol on 20 physical WSE-3 PEs. It is a communication
component with synthetic retained storage, not original-weight arithmetic,
complete matrix execution or Qwen inference.

## What executes

The 4 by 5 region has a serpentine neighbor path. A 257-word packet travels from
one endpoint to the other, then becomes the next packet in the reverse direction.
Its first word advances the epoch identity; all remaining 256 words are retained
through every feedback step. Colors 3 and 4 alternate. One input queue and one
output queue are rebound to the active color after local work and teardown have
both completed. Input and output use explicit distinct microthreads 2 and 3.

The sender transmits its teardown control wavelet after the complete data packet,
through the same output queue and microthread. A send callback only releases the
source buffer; it is not permission to change a route. Each PE advances only
after both its local completion and the per-color teardown callback. Forwarding
PEs have no local arithmetic, but still participate in teardown and reconfiguration.

The strict path reversal is essential: the next source was the previous
destination and waits for the entire packet and teardown before sending again.
Other next-path routers are either ready or held in teardown on the alternating
color. This argument does **not** establish readiness for arbitrary transitions
between the four complete-model forests. Those need a separate dependency and
readiness protocol; a send callback or endpoint receipt alone is insufficient.

`csl/route_word.csl` translates six-bit abstract port words using the installed
SDK's `target.fabric_io` values. It preserves non-route color fields, resets
switch position/state, rejects malformed words and refuses modification outside
teardown. Zero explicitly disables RX and TX. Queue rebinding uses the installed
input/output queue APIs. The probe checks route readback and the two rejection
cases in addition to end-to-end traffic.

The older official [stencil example](https://cerebras-sdk-docs-130.netlify.app/csl/code-examples/benchmark-7pt-stencil-spmv)
motivates waiting for operation completion and teardown together. Installed SDK
2.10.1 has a shared dispatcher with per-color `@set_teardown_handler` callbacks;
the probe uses those callbacks so it can coexist with memcpy. It never reads the
pending register again inside a callback. Exact installed source hashes are in
`route-api-inventory-002.json`.

## Physical result and timing scope

| Check | Result |
|---|---:|
| Main sequence / reset replay | 64 / 8 device-controlled epochs |
| PE transition/completion counter pairs | 1,440, all exact |
| Retained synthetic bank per PE | 35,256 bytes |
| Full bank readback | 705,120 bytes, all exact |
| Same-endpoint dependent timing samples | 70 |
| Dependent send/return cycles, min / median / max | 1,132 / 1,139 / 1,141 |
| Maximum compiled SRAM plus declared 4 KiB stack | 48,112 bytes |
| SRAM margin under the 48,128-byte ceiling | 16 bytes |
| Application coordinate coverage | All 20 PEs, exactly once |

Each cycle sample starts on an endpoint when sending epoch e and ends on the
same endpoint after receiving the full packet in epoch e+1. It includes both
19-hop traversals, remote completion, route/queue transition and feedback. It is
not isolated register-write latency. Samples overlap and are not summed to
claim total loop time. Host initialization, launch and readback are outside this
cycle interval; no cross-PE clock subtraction or assumed device frequency is used.
These cycles cannot be reported as model tokens/s.

The 35,256-byte sentinel matches the maximum P12 mixed-bank **data capacity**.
It is not a replacement for neural weights and does not exercise FP8/BF16 compute
code. The 16-byte margin cannot accommodate the full forest tables and neural
kernels. Combined code, scratch and stack admission remains a substantial
integration constraint; the component pass does not solve it.

## Provenance and released resources

- `route-epoch-sim-001` stopped at compilation: directly binding task 29 conflicts
  with SDK 2.10.1's existing memcpy teardown dispatcher. Source and diagnostics
  are retained. `002` uses per-color handlers without changing packet or storage
  gates and passes 16+4 epochs and full retention in 200.591 seconds total.
- The physical run uses exactly the simulator-qualified CSL, plan and driver.
  Artifact SHA-256 is
  `8b163a70b8f55e1b6fb579664730084381e95e8daeaf5100d549bf2bf853dede`.
- Compile job `wsjob-kqdk6bprvggyeux6jpcjqd` succeeded and released; lifecycle
  stage wall time was 41.449 seconds. Runtime job `wsjob-8avu52hmvgpjb9nrqmvx6f`
  succeeded and released; stage wall time was 61.457 seconds. These are lifecycle
  times, not device inference latency or provider-billed node-hours.
- The independent job/system audit at 2026-09-27 16:14:49 UTC found no own active
  job or device assignment. Other users' two assignments were left untouched.
  All bounded workstation units and the shared heavy-job lock were released.

Next integration must combine complete matrix operand distribution, reduction
and output return with general readiness/quiescence and actual cohost SRAM.
Complete 64-layer sentence generation at 2,000 dependent tokens/s remains unmet.
