# WSE-native lowering boundaries

The prior functional program holds the original model on 870,000 PEs, but routes
work through a serial global interpreter. The captured execution is approximately
30.08 seconds per token. The parked fast interpreter is unqualified and bound to
older sources; it is not a new validated baseline.

The new implementation separates five inspectable, serializable levels:

1. **Model semantics.** Original checkpoint revision, all operators, precision and
   rounding boundaries, recurrent/KV state and token feedback. Import the existing
   `configs/model-graph.json`; never derive semantics from a performance paper.
2. **Regions and ownership.** Adjacent layer/operator regions, matrix partitions,
   distributed norm/quantization/head reduction, explicit tensor owners and
   retained state. Manual knobs: region rectangles, row/K partition, replication,
   compatible fusion and precision-preserving weight transforms.
3. **Streams and events.** Producer/consumer ports, payload length, epoch, initial
   readiness, credit return and completion. Arrival activates work; a local
   region controller is not a full-wafer serial RPC. Manual knobs: buffer count,
   chunk size, reduction tree, pipeline depth and tile ordering.
4. **PE resources.** Per-PE routes, colors, queues, DSRs, microthreads, local tasks,
   memory lifetimes and scratch reuse. Reject conflicting simultaneous routes,
   unowned completion signals, missing consumers, and estimated SRAM overflow.
   Final admission uses compiled ELF sections plus a declared stack allowance.
5. **CSL and evidence.** Deterministic generated code, provenance hash, physical
   artifact identity, same-PE timestamps and invocation-correlated release.

Only the stream/PE/CSL slice for one multicast/reduction pattern is implemented
initially. A verifier for this restricted pattern is not a general proof of
arbitrary graph deadlock freedom. The model and region compiler will be added
incrementally; placeholder stages must not be reported as implemented.

## First event protocol

A root sends a contiguous packet down column zero and across every row through
hardware multicast. Intermediate routers forward without a software relay task.
Every endpoint consumes one packet, records its epoch, and contributes its unique
PE ID plus the epoch to a reduction. A node waits for its own receive and each
child acknowledgement before sending one sum to its parent. The root starts the
next epoch only after the complete tree acknowledges. Input/output queues are
separate; alternating reduction colors prevent RAMP route overlap. A send buffer
is not reused until its send completion callback. Repeated launches reset state.

This measures a bounded global control roundtrip, not an efficient final model
schedule. The eventual inference graph should communicate between adjacent
regions and avoid a full-wafer barrier at each operation.

## Design references and limits

- [WaferLLM, OSDI 2025](https://www.usenix.org/system/files/osdi25-he.pdf):
  topology-aware partitioning and K-tree reduction. Evaluation uses WSE-2;
  CodeLLaMA-34B and Qwen2-72B include subsets of layers. Current upstream has a
  WSE-3 migration, so code-version and hardware-version comparisons are distinct.
- [SPADA](https://github.com/spcl/spada): explicit placement, asynchronous streams
  and multiple lowering levels. Its published stencil scaling is not a Qwen
  inference rate. The local source archive already contains earlier qualification
  work; no large dependency or repository is duplicated here.
- Existing standalone matrix code in `../experiments/fp8_matrix` already exploits
  column multicast. Its host roundtrip alone does not identify decode, local
  arithmetic or reduction costs. Add same-PE timing before attributing speedups.

A 2,000 tokens/s target gives 500 microseconds per dependent token. It is a budget,
not a prediction. All 64 layers, embedding, full head, state updates, selection,
feedback and observation must be accounted for. Matrix weights reside in different
regions; aggregate wafer peak bandwidth cannot be applied to all sequential
layers at once. Scalar FP8 expansion and nearly full local SRAM are additional
constraints requiring measured kernel work.
