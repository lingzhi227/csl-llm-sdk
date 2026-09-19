# Native control termination: scoped two-PE qualification

The transport keeps every application word intact, including values whose high
bits resemble network headers. A finite header/body/tail receive sequence is
followed by a native control task that consumes the termination wavelet. The
receiver delivers the packet while its receive lease is held, then releases and
rearms the header. The sender sends the immutable header-plus-body frame, then the control wavelet.
It retains both the source and frame leases until control-tail completion; only
then does the completion callback release the preceding packet and permit reuse.

The accepted packet source binds data task6 and control task41, with control
payload0x00290000. Task41 is distinct from the SDK control task reservations in
the pinned2.10.1 installation. During asynchronous header/body/tail receives the
data task is blocked. Tail completion enables native task selection; native
control consumption blocks it again before validation and delivery. An ordinary
wavelet in that phase is counted and rejected, never silently discarded.

## Accepted scope and preserved failures

All eight isolated cases have separate evidence: valid lengths1,8,26,31; early
control; extra body data; duplicate control; and absent control. Valid lengths1
and8 passed within attempts whose later suite checks failed. Length26 required
a separate observation-layout change after a runtime crash; its failed attempt
remains failed. Length31 and all four negative cases passed in a later suite.
No single all-eight successful run is claimed.

The consecutive fixture sends31 words and then8 distinct words immediately from
the first device completion callback. It checks all39 delivered words, both PEs'
complete source/frame/receive banks, retained and untouched suffixes, per-packet
headers, ordering and leases, exactly two native control consumptions, no ordinary
control-phase wavelets, final rearm and stable counters. The first consecutive
attempt failed because a host poll selected an incomplete cached counter snapshot.
The accepted host gate requires both live completion and the full cached trailer;
all original strict checks remain. Its second snapshot equals the final snapshot.
This is finite observation evidence, not a proof about all future schedules.

The same two-packet case also passes with actual SDK message routing enabled.
The static color16 route is removed, mp.init configures routing, and both PEs
report bankA=4. All eight non-state captures match the accepted static run byte
for byte; state differs only in the two routing-enable fields.

These are two-PE SDK simulator results. They do not qualify
136 concurrent producers, physical transport repair, complete neural layers or
model inference. Earlier physical READY corruption remains an unresolved gate.

## Files and reproduction

`isolated/` contains the exact aggregate-snapshot device and host source used by
accepted valid26 and the remaining five cases. Earlier valid1 and valid8 evidence
belongs to the earlier build, as recorded in the source map. `consecutive/`
contains the exact accepted static two-packet device, host and checker source.
`message_passing/` contains the separately accepted actual routing variant.

Run the lightweight checker tests within either directory without SDK imports:
`checker_tests.py`, `native_checker_tests.py` and `snapshot_checker_tests.py` in
isolated; `consecutive_checker_tests.py` and `settled_gate_tests.py` in consecutive and message_passing.
The latter uses saved scalar evidence from the failed and completed observations
to exercise the actual host predicate; it is not another device run.

A fresh site-authorized build uses SDK2.10.1, WSE3, a9x3 fabric with offset4,1,
memcpy, one channel and one compile worker. Drivers require CPU affinity0 and a
fresh output directory per case, one simulator thread and normal stop. Our
accepted runs used an external30-second guard,20-second child deadline,4GiB
address/cgroup limit and zero swap. These published drivers alone do not enforce
the complete site supervisor contract. Historical binaries, admissions, vendor
source, credentials and raw arrays are not distributed.

[Report](../../../docs/NATIVE-CONTROL-QUALIFICATION.md) ·
[Source map](../../../evidence/ready-fanin/native-control/source-map.json) ·
[Attempt evidence](../../../evidence/ready-fanin/native-control/attempts.json) ·
[Official task IDs](https://sdk.cerebras.ai/csl/language/task-ids) ·
[Official control termination](https://sdk.cerebras.ai/csl/language/dsds#dynamic-completion-based-on-control-wavelets)
