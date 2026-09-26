# Full resident execution under development

Neither this contract nor the coordinate/instruction tables are an executable
full-model result. They define the remaining implementation after standalone
operator and transport qualification. ACCEPTANCE.md remains the completion gate.

`forward-program.json` binds all1251 original text tensors to883 instructions,
with400 FP8 projections,96 BF16 gate projections, all64 layers and the complete
embedding/head. Each instruction contains four u16 fields. The program occupies
7064 bytes, matrix descriptors7968, and145 activation/norm coordinates580:
15612 controller-table bytes before code, buffers, SDK and stack. GDN and full
attention group coordinates have exact checked formulas, so their2304+64 group
addresses do not require large controller tables.

The intended interpreter runs on the southeast PE. Its operand/result traffic
uses the qualified bottom broadcast/east-return architecture. Large activations
stay in16 reusable BF16 banks, each17536 elements. A projection reads128 elements,
quantizes on wafer if and only if it is FP8, broadcasts operands, separately
reduces complete strips, rounds to BF16, and stores the result in resident banks.
Original BF16 embedding/head and A/B gates preserve their distinct arithmetic.

Each original full5120 norm owner gains an in-place activation buffer. The
controller copies hidden values to it, invokes the qualified norm and uses that
PE as the projection input source. Residual and SiLU-product operations read
bounded128-element fragments and execute on the wafer. GDN group controllers
receive their Q/K/V/Z/A/B slices and own their duplicated original conv/gate/norm
parameters and convolution history. Two adjacent state PEs retain each full
128x128 recurrent head. Full-attention controllers receive the correctly split
six Q/gate heads and one K/V head; four adjacent64-channel cache PEs serve the
entire causal96-token history. No host hidden values or logits enter this path.

The next integration must share descriptors and routes explicitly. Proposed
global resources are command queue2/DSR2, reduction queue3/DSR5 for matrix roles,
return queues4/5 with DSR6, and synchronous matrix arithmetic DSR3/4. Actor roles
can reuse queue3 for their separate local network and DSR7 for sequential local
receive/send. Local colors must be disjoint from global3/4/5/8/9/10/11 and SDK
reservations. Exact local task IDs and integrated SRAM still require auditing.
The controller serializes actor computations and consumes each actor result
before dispatching another, allowing dead packet/join scratch to relay global
traffic without overwriting retained state or output. This needs physical proof.

Full-wafer PE code must use runtime coordinates/tensor IDs loaded once. Only
role, route parity, matrix endpoint flags and FP8 row phase should specialize
code. The small prototypes currently use literal coordinates; directly scaling
those specializations to870000 PEs would be an avoidable compilation risk.

For observable token latency, the first device call consumes the entire prompt
and produces its first generated token. Later calls only trigger the next
dependent step: the selected token remains on wafer and is fed to embedding
there. The host reads generated IDs, decodes text and records actual monotonic
TTFT/per-token/end-to-end times. It supplies no next-token ID or neural operands.
EOS, position limits and reset are checked on device as well as at request entry.
This avoids relying on an assumed hardware clock frequency for timing.

The candidate atlas additionally reserves3183 BF16 trace banks:1794 for all64
layer outputs at96 positions,1360 for complete vocabulary logits,29 for final
norm outputs. Their coordinates also have a checked compact formula. Trace
copies contain candidate outputs only, never reference inputs. This permits an
independent full-model comparison after device execution. Capture-enabled
validation and capture-disabled measured requests must be labeled separately;
clean reset and identical dependent outputs must be verified between requests.
Even with these reservations the coordinate plan leaves8347 PEs unallocated.

The candidate interpreter, combined wrappers and full layout have now compiled.
Full compile003 covers all870000 application PEs with434 ELF programs and passes
the declared SRAM gate. Role binding002 checks every live object extent against
the initialization roles. Original parameter audit002 passed353 small tensors
and all2304 GDN/64 attention group coordinates. The400 scale aliases are loaded
with their parent FP8 strips; loaded-name coverage must equal all1251 text tensors.

The full physical runtime002 has been admitted. It has not yet produced accepted
model outputs. Frozen criteria v1, complete candidate capture/timing, reset/replay
and the offline sequential-prefix CPU evaluator are implemented. Still required:
physical all-weight initialization, full dependent generation, every-layer/full-
vocabulary numerical comparisons, sentence checks, actual timings, and normal
stop with correlated system release.
