# Original Layer3 vertical region

This source-only candidate occupies 29 by 1160 PEs. It preserves all 464 logical
matrix stripes and 30,576 original 128 by 96 tiles, descending logical reduction
order, all arithmetic/state kernels, 24 query heads, 136 MLP owners and context8.
Actual compilation, emitted task/resource validation, host binding, original-word
preparation and physical positions0/1/reset0 remain separate required checks.

The top packet spine and frame rail use y0 and y1. Input normalization is [0,3];
the final residual owner is [28,3]. Matrix chains run south and reduce north.
Twenty-seven matrix columns include all16 native K/V roots in column2. Each
six-head group and its four source streams occupies a disjoint y interval.
Four physical colors are reused only between these disjoint regions; every
stream preserves its original six recipients. Each diagnostic archive sink is
immediately south of its head on an independent one-edge color6 path.

Three packet destination translations and regenerated Q-credit coordinate tables
change physical coordinates without changing logical application identities.
The wider packet envelope and packet implementation are byte-identical to the
accepted vertical Layer0. All28 other accepted Layer3 CSL modules are unchanged;
matrix_lane changes only its two physical directions. Router buffers, three
ports, tasks and DSR declarations are retained. These source facts do not replace
inspection of the actual emitted program.

Host mapping contains252 role-exact weight rectangles,751435776 host bytes,
50 parameter calls and508 configuration calls. Every copy is at most16MiB.
Each weight segment refers directly to a row and ordinal of one of the95 already
qualified original packed files. No model tensor decoding or precision change
is part of the repack. The full physical KV state includes all24 duplicated head
caches,196608 native bytes. Payload export does not yet implement restoration of
private request,position,reset or archive control state.

The complete generated packet tree costs1394298 packet-hops including endpoint
links for the unchanged1223 flows. The longest path is1792 hops. These metrics
exclude frame, native KV and matrix collectives and are not measured latency.
Native KV routes contain1576 total edges per word across16 source trees.

With18 Layer0-like regions and6 Layer3-like regions, width arithmetic gives
18*30+6*29=714 of750 application columns. This is a placement motivation, not
qualification of shared control, embedding, full-vocabulary output, state restore
or an entire stage. Context remains8; full-model and token-performance acceptance
remain open. Adjacent east/west boundaries do not implement a device handoff.

The candidate keeps16 memcpy channels and an8GiB remote worker. Compile/runtime
limits are inherited from the accepted bounded workflow. Layer3's accepted SRAM
policy is ordinary allocation plus4096 declared stack at most48640 bytes, within
49152 physical bytes. Layer0's distinct48128 candidate policy is not substituted.
The declared reserve is not a measured runtime stack high-water mark.
