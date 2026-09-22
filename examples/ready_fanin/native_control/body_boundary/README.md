# Full136 body-completion boundary

Physical RUN009 still fails the original 650-record criterion. Its new format
check detects the same bad prefix as RUN008 before the current packet's tail
receive is armed. Independent review accepts that boundary observation, normal
exit, resource release and original evidence backup. No repair is claimed.

The two CSL files here are the exact changed sources compiled in COMPILE011
and used in RUN009. SOURCE-DELTA.patch records the complete change from RUN008.
The other four CSL files, nine qualified host files and runtime context adapter
are byte-identical to [first_fault](../first_fault); the evidence source map pins
all 16 active source/context files. Historical production-packets.csl is not
the selected transport module. No SDK distribution, compiled artifact, raw
capture, submission guard or execution admission is supplied here.

The scalar predicate checks immutable prefix format after body completion.
It does not deliver a packet or change application state; only the original
native tail path can do that. Invalid format uses existing first-fault storage
with reason 22 and receive phase 2. Full success still requires all original
650 checks plus two complete, identical and zero fault banks.

[Report](../../../../docs/FULL136-BODY-BOUNDARY.md) |
[Evidence and exact source map](../../../../evidence/ready-fanin/native-control/full136-body-boundary.json)
