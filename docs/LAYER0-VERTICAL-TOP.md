# Vertical Layer0: top-spine source and actual static qualification

> Historical static milestone. Subsequent physical positions 0/1/reset/0 are
> accepted within the [operator/state scope](LAYER0-VERTICAL-PHYSICAL.md).
> Statements below about pending execution describe this earlier milestone.

September23,2026 UTC. The complete original Layer0 now compiles as a30 by1160
vertical region with west-to-east layer boundaries. Actual placement covers
34800 PEs:31548 matrix,460 support,970 routers and1822 idle coordinates.
[Generated source](../examples/layer0_vertical/top_spine) and the
[evidence summary](../evidence/layer0-vertical/top-spine-static.json) describe
static qualification; no physical Layer0 numerical epoch is claimed.

## Placement and communication

Every128-output-row by96-input-column tile preserves original BF16 packed words
and logical input ordinal. Matrix chains run south, reduce north in the same
descending logical order and broadcast south. Whole chains share28 matrix
columns. Control, convolution, head, recurrence and MLP owners occupy the west
support region. Input normalization is[0,3], final residual output[29,3].
The packet spine is y=0, frame route y=1 and origin[0,1]. The origin shares the
first spine router through an existing port; all routers retain three ports.
Adjacent geometry alone does not implement a qualified layer-to-layer handoff.

| Same4369 flows /9157 fixed128-byte packets | Horizontal | Superseded bottom spine | Current top spine |
|---|---:|---:|---:|
| Packet-hops including endpoint links |4134058|6287604|2819067|
| Maximum directed physical-link packets |3847|3809|3832|
| Longest single path in hops |1544|2347|1576|

The first vertical placement increased total packet-hops by52.1%; it was
superseded before runtime. Moving the spine and support origins to the top
removes that regression. Current packet-tree cost is360840576 byte-hops.
The longest individual path is still32 hops longer than horizontal. These
metrics exclude frame, observer and matrix collectives, and do not measure
execution critical path, pure device latency or end-to-end token performance.
Original bottom-spine source/artifact evidence remains a static resource baseline.

## Actual compiled evidence

The accepted appliance compilation completes in354.903386 seconds and produces
a23703087-byte artifact. Independent checks cover1508 application ELF programs,
34800 PEs and401469 coordinate banks. The largest ordinary allocation plus a
4096-byte declared stack reserve is47824 bytes against48128, leaving304 bytes.
This is not a measured dynamic stack peak. Actual2,702,073 emitted instructions
support15 resource models,14 task models and entry fences for941 endpoint PEs.
The15 DSR operation signatures match previously qualified emitted structures;
this is not exhaustive control-flow or dynamic lifetime proof.

All138 installed peripheral members,10919669 bytes, are byte-identical to the
first16-channel vertical compilation. Actual west/east mux parameters assign a
65-row first band and fifteen73-row bands. No separate SDK-internal decoder or
additional device experiment is claimed. The original compile owner is reaped
and released; original binary and full decode remain remote private evidence.

## Host mapping and remaining execution work

The physical map defines190 weight ROIs/775323648 bytes, five small parameter
ROIs,338 configuration ROIs and two full persistent-state ROIs. Every copy is
bounded by16MiB. Checkpoint conversion preserves786432 recurrent FP32 indices
and40960 convolution BF16 history values. Payload conversion alone does not
restore private request,position and reset state. Weight-only bounded asynchronous
upload is implemented in the private runtime candidate but has not run on the
device; sixteen configured channels do not imply sixteen-way payload concurrency
or measured speedup.

## Preserved startup failure

The first physical runtime attempt reached its fixed 240-second startup limit
before model H2D, launch or neural capture. The environment context returned at
170.13 seconds and the device start call began at 171.14 seconds, measured from
the startup trace. That call had not returned at the deadline. The original
supervisor exits 1; the owned job was cancelled, its processes were reaped, and
independent inspection confirmed no remaining system assignment. Original
failure evidence is preserved. These observations do not identify a unique root
cause or establish a required startup duration. No Layer0 numerical failure or
measured performance result follows from this attempt.

Next are actual original-weight Layer0 positions0 and1 plus reset replay,
complete numerical/state/transport validation, vertical Layer3 migration,
actual0→1→2→3 hidden flow,
qualified checkpoint control restore and real20/24/20 stage capacity. The target
is one physical CS3 reused for three sequential logical stages, including all64
layers and full-vocabulary autoregressive generation. Accepted full-model epochs
remain zero. No source-propagated whole-layer error enclosure is asserted.
