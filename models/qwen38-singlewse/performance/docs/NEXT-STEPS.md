# Next work: selected whole-model architecture

The active implementation plan is [DIALOGUE-ARCHITECTURE.md](DIALOGUE-ARCHITECTURE.md).
The user explicitly requested complete-model layout design before further
coordinate-specific integration. P22–P49 remain preserved comparison baselines.

1. Lower the selected variable-width 7/9-layer macro partition from logical
   ownership through complete physical routes and resource leases. The frozen
   candidate is `evidence/dialogue-architecture-001/selected-plan.json`: all64
   resident layers, one conversation,512-position state reservation and66
   explicitly budgeted controllers. Compiled SRAM and runtime are unqualified.
2. Resolve GDN head-local state placement, distributed normalization/fusion and
   all producer-to-consumer paths. Audit whole physical-link load and liveness;
  63 boundary handoffs alone do not qualify the complete network.
3. Test the decisive exact-FP8 decode/row-loop performance hypothesis under the
   selected memory budget. Existing primitive cycle reuse misses the target by
   a large margin; a routing-only plan is insufficient.
4. Qualify complete GDN→attention→GDN layer transitions2→3→4 and macro-boundary
  6→7, including persistent state, actual successor consumption, local callback
   retirement and independent numerical checks. Then expand64 layers/head/
   feedback and the appended-input/output serving contract.
5. Measure complete dependent dialogue responses at>=2000 average content tokens/s,
   separately report TTFT/decode/prefill and initialization, and publish evidence.

No further bridge/cohost/full-instance sequence is scheduled merely to retain
old snake coordinates. A small experiment needs an explicit architecture choice
it can resolve. Preserve all old results, including unsuccessful attempts;
release task-owned workstation and WSE allocations after each bounded run.
The former integration plan is retained as HISTORICAL-NEXT-STEPS-P49.md.
