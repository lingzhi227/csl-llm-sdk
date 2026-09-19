# Bidirectional native-control payload and row lifecycle

Three physical PEs execute two REQUESTs, one independent READY and six row
fragments: 9 packets and 200 application words. Origin sends requests and receives
two exact 128-halfword synthetic rows; the peer releases each source before reuse.
The READY producer shares the incoming route. This is the exact RUN006 source.

Independent checks accept all payloads, complete source/frame/RX banks and short
packet suffixes, causal row starts and final state stability. Overall RUN006
remains failed: only 30 of 31 checks pass. Producer's first-start notification was
handled after its own first TX completed, so both required first-lease overlap
witnesses were not obtained. Origin separately records one RX during an issued,
unfinished TX. None of these observations proves simultaneous DMA.

The complete strict checker is unchanged and still rejects that missing witness.
SIM012/014/015 failed before compute; traced SIM013 passed only a shortened
initialization prefix. The 136-producer corruption and original neural execution
remain open. A subsequent GO-relocation proposal does not replace this source.

capture.py accepts a factory returning an already entered physical context with
an explicit closed flag. Site admission, artifact bindings and allocation guards
are not included as an immediately submit-ready application. run_hw.py and
physical-context-original.py preserve the actual reviewed wrapper source for
comparison; running them as jobs requires those separately qualified site guards.
simulator-capture-original.py and simulator-entry-original.py are exact historical
references. PHYSICAL-HOST-DELTA.json records the four capture adaptations.

checker_tests.py retains seven legal interleavings and 7,637 rejected contradictions.
capture_tests.py is a separately bounded Linux/CPU 0 SDK-free fixture using the
actual wrapper class AST and a fake SDK context. It checks maximum and earliest
completion, copy failure and stop failure, then rereads the original saved arrays
for both successes. It writes synthetic host-mock-results and refuses reuse of
that directory. These are host tests, not device or model qualification.

[Report](../../../../docs/NATIVE-CONTROL-BIDIRECTIONAL.md) |
[Evidence](../../../../evidence/ready-fanin/native-control/bidirectional-attempts.json) |
[Exact source map](../../../../evidence/ready-fanin/native-control/bidirectional-source-map.json)
