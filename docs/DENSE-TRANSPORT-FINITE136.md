# Dense transport with arithmetic: finite physical qualification

September 23, 2026 UTC. A synthetic 136-PE fixture is accepted on physical WSE-3
for two positions followed by reset and exact position-0 replay in one runtime
context. Matrix collectives, routed row packets, acknowledgements and retained
operand ownership pass together. Independent reconstruction from actual captured
operands finds zero mismatches in 1,179,648 ordered FMAs, 9,216 collective additions
and 3,072 BF16 outputs. This qualifies a communication and arithmetic component.
It uses no original model weights and does not qualify a complete layer, model
stage, vocabulary head, generated token, or token performance.

[Source snapshot](../examples/dense_transport/finite136) ·
[Evidence summary](../evidence/dense-transport/finite136/summary.json) ·
[Source hashes](../examples/dense_transport/finite136/source-map.json).

## What ran

The 34 × 4 application contains 32 matrix PEs, 16 software routers, four support
PEs and 84 idle PEs. Eight four-lane matrix stripes use both row parities. Each
lane retains a 128 × 96 BF16 matrix; two phases consume separate 384-value FP32
inputs. Local ordered accumulation and right-to-left four-lane collective sums
produce eight 128-value BF16 rows per serial. Each row travels as 23/23/18 packed
word fragments, followed by acknowledgement. Matrix PEs between routers forward
the dedicated packet routes in fabric.

The serial/generation/position schedule is `(1,1,0)`, `(2,1,1)`, `(3,2,0)`.
The last serial follows a device reset. All three serials have two identical full
boundary snapshots. Checks cover all 136 coordinate identities, errors, retire
counters, router leases, source callbacks, acknowledgements, eight complete rows
and 30 directed router edges. Captured weights and descriptors at every matrix
PE match their original host transfers for every serial. Reset replay matches
the first output bytes exactly.

## Independent arithmetic and storage checks

The numerical domain is deliberately finite and exactly representable. BF16
weight numerators are integers divided by 128; FP32 input numerators are integers
divided by 32. The independent audit decodes the actual captured operands,
reconstructs ordered FP32 FMA/addition bits and BF16 round-to-nearest-even output,
and checks destination rows. It imports neither the executor's fixture/numeric
modules nor the SDK. It also verifies all 309 raw hashes, 590,400 retained
parameter words, three stable snapshot pairs and reset replay. Exactness here
does not extend to arbitrary model inputs.

The actual appliance artifact has 22 application ELF programs across 136 PEs
and 1,005 named coordinate SRAM banks. The largest ordinary allocation ends at
40,080 bytes; with a 4,096-byte stack reserve the total is 44,176 bytes, below the
48,128-byte ceiling. This is a static allocation check, not a measured dynamic
stack peak. Actual emitted code and per-coordinate storage were independently
reviewed. Matrix arithmetic uses DSRs 2/3/4, collective and packet transmission
share DSR1 behind lifetime fences, input uses DSR5, packet receive DSR6 and host
memcpy DSR0. Router receive uses DSRs 2/3/4 and transmit 5/6/7. This does not
establish an optimal overlap schedule.

## Held acknowledgement and timing scope

The held case captures a complete row at support PE `(11,0)` while its ACK is
unsent and source PE `(1,3)` still holds the output lease. The interval from just before prepare
to release RPC return is 0.023111348 seconds, within the five-second host deadline. Source
ACK and the complete quiescent boundary are checked subsequently. RPC return
does not prove the device received the ACK at that instant, and the sampled
parent monitor does not provide a hard device deadline guarantee.

The preliminary small-call screen takes 0.020717468 seconds, with maximum
individual call 0.006095015 seconds. Runtime readiness takes 93.654614 seconds,
the capture envelope 1.983401 seconds and normal stop 4.455739 seconds. These are
diagnostic wall times including runtime, host and capture work. None is token
latency, throughput or an official billing measurement. The artifact uses one
host memcpy channel and blocking calls; no multichannel bandwidth is claimed.

## Evidence and preserved failures

The successful capture contains 309 host copies, 26 launches, 6,115,056 transferred
host bytes and 309 raw files totaling 6,083,792 bytes. Its 1,320 journal events
occupy 290,812 bytes. Runtime, executor audit and independent audit all exit
normally; every owner is reaped and the physical allocation is released.
All original runtime evidence and the independent audit are retained in separate
immutable remote backups with independently verified hashes. Private raw data,
vendor SDK files, credentials and appliance binaries are not published here.

Earlier failures remain part of the result:

- SDK001 rejected duplicate route-color configuration. SDK002 rejected an
  unsigned frame offset; SDK003 adds the bounded signed conversion.
- HW001 reached its 180-second startup deadline before capture and was cancelled
  and released. A subsequent bounded worker-log export failed because the server
  rejected its argument combination; no archive was returned or retry performed.
- HW002 entered the SDK successfully, then failed in host diagnostic logging:
  `seconds` collided with a reserved event field. Normal stop and all original
  failure evidence are retained. HW003 changes that field to `elapsed_seconds`,
  with tests using the actual event store; the device program and arithmetic,
  copy contract, capture budgets and held-ACK deadline are unchanged from HW002.

The byte-exact executor audit record was written before controller acceptance
and therefore retains `controller_acceptance: false`. The later independent
acceptance is bound in the summary by its canonical hash; the original record
has not been rewritten. The independent result published here is an explicitly
labelled extraction from that acceptance, with the original result hash.

## Use in the full-layer work

The accepted finite packet/math coexistence and retained-buffer rules can now
support the complete original-weight Layer0 integration alongside the accepted
nine-PE arithmetic component. Full graph placement, actual compiled storage,
operator contracts, complete recurrence, hidden flow into Layers 1/2/3 and stage
checkpoint restoration still require their own evidence. This result does not
establish all-stage routing capacity or remove those integration obligations.
