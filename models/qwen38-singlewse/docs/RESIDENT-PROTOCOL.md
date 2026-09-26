# Candidate device control protocol

This architecture is under development. The complete model is not implemented
or compiled. `placement-resident-96.json` reserves an east result spine and a
bottom broadcast spine before allocating other weights and persistent state.
The 1,909 reserved cells leave 17,594 cells for nonlinear/vector/control roles.
The original weight strip coordinates remain unchanged.

One controller at the southeast corner broadcasts fixed command frames west
along the bottom row and north along every column. A frame has an opcode,
matrix identity, operand-group identity and data. All endpoints drain every
frame; matching endpoints consume it. Tensor identity and local contraction
index are loaded once with the original weights, rather than unique compiler
parameters for every logical layer.

Projection input frames supply one original group of128 activations. Matching
weight endpoints compute and retain their local partial result. They do not
start row reduction until a separate reduction command. This is necessary:
an early sender that blocks waiting for a later operand would also stop the
global multicast and could prevent that operand from arriving.

A reduction command arms the entire selected strip. Its left endpoint starts
the existing alternating-color reduction; other endpoints receive and add in
ascending contraction order. The right endpoint retains the complete result.
The next command may backpressure at an endpoint until this operation finishes.

Result collection is serialized by output strip. A read command identifies the
physical row and source column. Endpoints to its east forward a fixed result
extent on alternating horizontal colors. The east spine forwards it south to
the controller on different alternating colors. All other endpoints remain
ready for the next command. The controller issues the next read only after the
complete expected extent arrives. Result transfers carry no host-computed
activations, and the controller must validate command/request/position identity.

This provides a correctness-first path through interleaved strips without
allocating a dedicated full-wafer software routing tree. It can be slow; only
actual complete-model measurements can establish its latency. Projection
endpoint code plus communication must pass the48,128-byte SRAM gate including
the4,096-byte stack allowance before the route is accepted.

Still missing: qualified endpoint implementation, bounded command sequencing,
all nonlinear/vector role commands, original embedding lookup, full output
head and device argmax/feedback, reset/error behavior, whole-fabric routing and
full-model numerical qualification. Neither this document nor the coordinate
reservation closes any of those gates.


## Runtime resource leases

The prototype uses DSR2 and queue2 for command frames, DSR4 for synchronous
FP8 arithmetic, DSR5 and queue3 for row reduction, and DSR6 with queues4/5 for
result return. The installed WSE3 memcpy implementation defaults to DSR0;
input queues0/1 and output queue0 remain reserved. No permanent receive aliases
SDK descriptors. With no explicit microthread ID, the SDK language rule selects
the first fabric operand queue ID, so command waits use microthread2 rather
than the SDK memcpy microthread0. Read-only inspection used the installed
SDK2.10.1 container (internal compiler path rel-sdk-2.10.0/202604012315-1813-1394a6ea).

The language rule is documented in the [SDK Microthread IDs reference](https://cerebras-sdk-docs-130.netlify.app/csl/language/microthreads_wse3).
The accepted full-matrix and recurrent-head evidence remains separate from
this prototype's still-pending compiler/SRAM/runtime qualification.
