# Native control: physical concurrent comparison

This directory contains the exact six CSL sources, strict record checker and
bounded capture components from physical run004. Compilation passed for 356
programs, but the physical protocol check failed. This is an archived failure
reproduction component. No execution admission or compiled artifact is included.

The application retains 178 x 2 PEs, 136 simultaneous READY producers, eight
requests and 24 row fragments (three per request). The success condition remains 650
exact records. Header bit 5 selects control termination, and a native control
task consumes each expected tail before the receiver releases its payload.

The physical run saved 13 captures and stopped normally. All 408 full producer
records were exact, but the origin stopped after 22 valid records with ordinary
value 3 at the expected control-tail position. This did not repair the fixture.
Only initial and producer raw files match run003 byte for byte; receive ordering
and the first fault changed. A unique root cause has not been established.

For source inspection, start with layout.csl, pe.csl and packets.csl, then follow
the bounded sink capture into check_capture.py. production-packets.csl is the
unimported historical baseline. capture() requires an already authorized runtime
adapter; hw00/store.py is its own durable metadata helper. Site artifact binding,
allocation and execution guards are not packaged as a ready-to-submit job.

Actual full-role inspection found 356 programs, a maximum 27,872 bytes including
the 4,096-byte stack allowance, and separate RX, frame, copy and diagnostic
registers. These are static fit/allocation findings. They do not establish a
dynamic exclusion trace or a physical protocol pass.

[Report](../../../../docs/NATIVE-CONTROL-PHYSICAL.md) |
[Source provenance](../../../../evidence/ready-fanin/native-control/physical-source-map.json) |
[Attempt evidence](../../../../evidence/ready-fanin/native-control/physical-attempts.json)
