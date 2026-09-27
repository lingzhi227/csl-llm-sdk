# Complete-dimensional static MLP pipeline candidate

The P21 mainline now has CSL bodies for the connected full-dimensional MLP,
including a preceding residual-add producer and the real following RMS consumer.
It remains a candidate until the complete layout, original parameter binding,
numerical execution and timing gates pass. It is not a full-model executable or
a new measured token rate. The complete original64-layer, single-WSE-3 target
of at least2000 dependent tokens/s remains unmet.

## Static routes remove an avoidable phase transition

The P19 plan considered rewriting horizontal native-tree colors for vertical
down gathering, which would require proving forwarding drain, changing routes,
rebinding queues and collecting row readiness acknowledgements. The new fabric
gives down a separate static local path: alternating colors2/15 gather ordered
output pairs north, while color18 broadcasts a complete group128 native packet
west and south. Existing gate/up tree colors remain unchanged throughout.

This fits the same original5120/17408 geometry. The independent fabric audit
covers1059920 flows,4843946 PE/color entries and1857370 directed links. Every
multicast has one RAMP source, connected neighboring routes, the exact expected
consumers, and no conflicting static color definition. Aggregate maximum traffic
is256 words per directed link for the listed full MLP and two RMS phases.
It is a word count, not a congestion-aware cycle prediction. External predecessor
transport and full-model epoch/termination traffic are not included.

No route word rewrite is needed for these paths. A gate worker still waits for
its own native output callback before reusing decoded scratch and input/child
queues for down. The separate static paths allow forwarding for other rows and
regions to continue. This local reuse rule does not pretend to certify arbitrary
fabric drain. Ordered two-word gather forwarding and eight-word inter-region
joins retain outgoing-send credits.

## CSL roles and real numerical boundaries

`runtime/mlp_worker.csl` consumes the shared65-word normalized packet, decodes
the original resident2x128 FP8 tile, runs the qualified native FMA/scaling path,
and joins the original K40 tree in its explicit local/left/right order. Roots
send packed BF16 gate/up pairs directly to adjacent activation actors. Gate banks
then reuse the same compute/scratch for their original down tiles and stream
ordered FP32 pairs north. All64 MLP layer slots follow P20's residency addresses.

`runtime/mlp_activation.csl` waits for both actual gate/up packets, preserves the
SiLU and multiply BF16 boundaries, computes group128 maximum and scale, encodes
directly into native input format, and gathers the65-word down packet. The group
can advance independently of other intermediate groups. Its current nearest-
neighbor maximum/gather schedule is a concrete first integration to measure;
it is not asserted to be the fastest available group protocol.

`runtime/mlp_join.csl` generalizes P20's resident chunk helper to the real region
parities, leaves and terminal header, plus the dedicated up relay. The original
embedding-only bank obligation remains present on collectors and gate headers.
It consumes actual down output streams in the candidate layout. P20's physical
result does not automatically qualify these new specializations.

`runtime/mlp_norm.csl` holds all129 nearby original norm-gain slices. It computes
the preceding residual add from its two supplied operands, distributed post-
attention RMS and the shared quantized input. Returning down chunks then round,
add the retained BF16 residual, round again, and immediately contribute to the
actual next-layer RMS. The final layer uses the final norm slot. The supplied
boundary operands do not mean the preceding GDN/attention is implemented here.
The distributed FP32 RMS sum order and the full spatial projection order still
need independent numerical qualification. No tolerance has been relaxed.

`runtime/mlp_idle.csl` retains spectator banks outside this MLP's active roles.
It supplies storage only; the neural operations associated with those other
weights are not implemented by the MLP program. Other state/attention/GDN/head
programs and complete model control remain required.

## Compile admission and compact layout

The selected25-role compile census includes native roots/interiors/leaves,
original bank scale-offset profiles, activation roots/interiors/leaves, norm
owners, both collector/header color parities and their child/leaf variants.
Attempt002 passes with47536 maximum bytes including4096 reserved stack.

Gate workers now read their physical Y coordinate once per arm to reconstruct
their pair index. This avoids64 program specializations merely for different
forwarding lengths. Fixed fabric offsets4,1 and the full geometry are required;
the selected compile census itself is not placed at execution coordinates.
Attempt004 compiles the corrected qualified enum spelling and passes at47776
maximum bytes, leaving352 bytes below48128. This checks selected role images,
not every final whole-wafer specialization or measured dynamic stack usage.

`spatial/mlp_layout.py` emits a7470-byte candidate layout containing all870000
PE roles and their static data paths. Original bank profiles and norm slot order
are checked before emission. Candidate snapshots preserve their source identities.
The full-wafer compiler result is tracked separately; generating this source is
not proof that the compiler accepts the complete composition.

## Preserved failures and scope

Integrated compile001 omitted required layout export declarations. Compile003
used an unsupported shorthand for the SDK fabric-coordinate enum. Both failed
before execution; their frozen sources and compiler diagnostics remain intact.
The original002 and corrected004 successes are separate receipts.

Full compile001 failed in its generated host launcher before calling the CSL
compiler because a newline inside a string was emitted incorrectly. The launcher
now uses a raw bytes literal and parses every staged Python source before dispatch.
The invalid original source is retained with its observed syntax-error identity;
publication only admits that exact frozen failure and rejects newly introduced
or active-source syntax errors. No physical job was submitted by these checks.

Full compile002 reached its600-second workstation compiler cap and was terminated;
the service and residual cgroup processes are released. It did not produce an
accepted whole-wafer result. Candidate003 keeps the same role bodies, reuses SDK
parameters per column and groups route dispatch by RX direction to reduce layout
evaluation overhead. The installed SDK get_params(px) depends only on px and
fixed constants, so this hoist does not change a PE's parameters. Full compile003
is pending at this milestone snapshot, bounded by4GiB/no swap, two CPU cores,
1800 compiler seconds and1850 total service seconds. The older functional full
compile took750.612 seconds, supporting a larger bounded observation window.
That historical duration is not evidence that the new compile will succeed.

All71 current source tests pass. The latest hardware account audit has151 terminal
owned jobs and no owned system allocation; this work added no WSE jobs. The pending
workstation compiler is explicitly separate from those released hardware resources.

Next: admit the complete layout, bind original weights without aliasing, freeze
an independent full MLP oracle and run numerical/replay/retention gates before
reporting same-controller complete-path latency. Initialization and fixture work
remain separate from inference timing. The all-layer controller, other neural
operators, dependent sentence generation and2000tokens/s acceptance remain open.
