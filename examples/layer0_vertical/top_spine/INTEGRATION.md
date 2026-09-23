# Vertical Layer0 with a top packet spine

This source places the complete original Layer0 in a 30 by 1160 region:
31548 matrix PEs, 460 support PEs, 970 routers and 1822 idle coordinates.
The packet spine is at y=0 and the frame route is at y=1. The origin at
[0,1] shares the first spine router [0,0] through its existing port 0.
All routers retain three ports. Branch parent traffic travels north.

Input normalization is at [0,3] and the final residual output is at [29,3].
These remain west/east boundaries for adjacent vertical regions. The actual
layer-to-layer handoff and private control-state restore are not implemented
by this single-layer source. Payload placement alone does not establish either.

Every logical 128 by 96 matrix tile, packed weight word, ordinal and descending
FP32 reduction order is unchanged. Matrix chains run south, reduce north and
broadcast south. Thirty arithmetic/protocol CSL modules are byte-exact with
the accepted horizontal source. Thirty-one of all thirty-four CSL modules
are byte-exact with the first vertical source; only layout, coordinates and
router selection changed from that version.

For the same 4369 flows and 9157 fixed 128-byte packets, complete packet-tree
paths including endpoint links cost 2819067 packet-hops, or 360840576 byte-hops.
The first vertical bottom spine costs 6287604 packet-hops and the horizontal
source costs 4134058. The maximum directed physical-link load is 3832 packets
(horizontal 3847; bottom spine 3809). The longest individual path is 1576 hops
(horizontal 1544). These source metrics exclude frame, observer and matrix
collective routes and do not measure execution latency or its critical path.

The generated source checker covers all 158088 physical links, 193458 color
entries, logical tile identities, unique tree paths and packet ledgers. The
controller independently checked the actual emitted router selector, enabled
port masks, colors and configuration against every flow. Actual compiled banks,
task and DSR assignments, SRAM and peripheral routes require the new compile.
The first vertical artifact is retained only as a resource baseline.

host_mapping.py derives physical ROW_MAJOR copies: 190 matrix weight ROIs
(775323648 bytes), five original parameter ROIs (229760 host bytes), 338
configuration ROIs (1547616 host bytes) and two full persistent-state ROIs
(3309568 host bytes; 3227648 native bytes). Copies are bounded by 16 MiB.
weight_packing.py preserves accepted preparation002 u32 words and validates
every output tile independently. checkpoint.py converts all 786432 recurrent
state indices and 40960 convolution history values without changing bits.

The new compile requests 16 installed SDK 2.10 channels with physical x passed
to memcpy.get_params. IO-CHANNEL-COVERAGE.json records the actual 65-row first
band and fifteen 73-row bands. Configuration and state copies remain blocking.
The separate runtime source uses bounded weight-only asynchronous windows:
at most four tasks in disjoint bands, at most 64 MiB retained, and FIFO wait
and done checks on each public Task before releasing its source buffer.
This source intent is not evidence of server concurrency or measured speedup.

No SDK invocation, original-weight preparation, runtime or numerical acceptance
is included in this source handoff. The unused bottom-layout preparation001
must not be executed for this geometry. Actual 0-3 layer integration, complete
checkpoints and real 20/24/20 stage capacity remain full-model work.
