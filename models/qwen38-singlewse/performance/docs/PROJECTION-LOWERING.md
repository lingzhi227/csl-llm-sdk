# Complete projection events and reduction forests

`projection-plan-003` lowers the complete pinned P12 model atlas. It preserves
all 498 original matrices and 105,052,160 real matrix tiles. This is a graph,
storage-order and abstract route result; operand multicast, output scattering,
the general route-transition protocol and complete executable remain unqualified.

## Shared input events

Adjacent pure projections can share input preparation when their input values,
operator attributes, representation and K width agree, and their original
weight intervals are contiguous. Every output retains its own original tensor
identity, output name and row interval. No rows or layers are removed. Examples
include gate/up, adjacent Q/K/V, GDN input projections and BF16 a/b projections.

| Quantity | Original | Lowered |
|---|---:|---:|
| Graph events | 1,172 | 980 |
| FP8 projection/input preparations | 400 | 256 |
| BF16 GEMV/input preparations | 97 | 49 |
| Embedding lookup | 1 | 1 |
| Original matrices | 498 | 498 |

The 306 bundles use 428 complete-K resident-slot segments. Multiple projections
in one concurrent bundle occupy distinct bank PEs; the largest uses 696,320.
The full vocabulary head spans repeated resident slots and must execute those
segments sequentially. Embedding dispatch selects the requested row. Neither
operation is misrepresented as a simultaneous full-ring GEMV.

## Storage completion is a separate dependency

P12's closed-lifetime arena was allocated in original graph order. Following only
data edges is insufficient when independent branches reuse the same cells. The
lowering adds a completion edge from the final reader of each earlier value to
the producer that overwrites those cells. These edges require completion of the
read, not merely return of a sender buffer.

There are 192,736 overlapping-storage value pairs and 148,543 distinct direct
memory edges before reduction. Transitive reduction against existing data and
memory order retains only 18 additional memory completion edges. An independent
set-based reachability audit reconstructs lifetimes from event inputs/outputs
and proves all 192,736 orders remain present. It also checks fusion does not
extend any value into another live allocation.

`projection-plan-001` is a preserved, unadmitted draft without memory completion
edges. `002` is the safe unreduced graph; `003` is current. Their status is explicit
in `projection-plan-history.json`. Numerical precision rules are unchanged by
projection grouping itself; the reduction tree is a separate implementation
choice qualified at component scale in P9, not claimed bit-identical to the
original ascending-block reduction.

## Four bank reduction profiles

All atlas stream starts, resident ring lengths and class phases align to complete
K groups. Thus matrix identity does not change the underlying tree boundaries:
only FP8 K=40/48/136 and BF16 K=40 profiles are needed. Each bank has a compact
preorder rank; two route colors per tree depth give 13 color planes in total.
Every activated segment contains complete disjoint trees. Resident slots reuse
the same profile sequentially.

The physical embedding includes all 860,880 bank coordinates and 12 actor-row
bridge coordinates, producing a continuous 860,892-PE neighbor path. BF16 state
banks are excluded as contributors but remain routers inside a tree interval.
This forwarding distinction must survive later queue and task lowering.

| Profile | Trees | Edges | Active PE/color entries | Longest single edge, physical hops |
|---|---:|---:|---:|---:|
| FP8 K40 | 21,522 | 839,358 | 3,228,324 | 22 |
| FP8 K48 | 17,935 | 842,945 | 3,228,330 | 26 |
| FP8 K136 | 6,330 | 854,550 | 4,000,602 | 70 |
| BF16 K40 | 20,600 | 803,400 | 3,184,930 | 797 |

The BF16 maximum is a real layout cost exposed by the audit, not a measured
latency. Routing around excluded state regions deserves measurement and possibly
a different local mapping. These forests alone do not implement input multicast
or delivery of completed rows to the P12 value arena.

## Independent audit and limits

`audit_projections.py` reconstructs tree edges with an independent iterative
interval traversal. For each of the four profiles, it expands every edge using
endpoint difference arrays, rejects every PE/color collision and compares all
11,191,596 dense route words. Total coverage is 44,766,384 entries. It separately
checks physical bounds, coordinate uniqueness, neighbor continuity, actor bridges,
BF16 eligibility, all matrix/row bindings and exact resident-slot segmentation.
The latest revision also checks each published compact vertex/cut mask against
the independent oracle.

`projection-audit-002` completed the full dense audit in 2.867 seconds with
197,320 KiB peak RSS. `003` adds the compact-document comparison and passes in
2.873 seconds at 197,256 KiB peak RSS. Both run under
a shared workstation lock, 2 GiB memory, no swap, two assigned CPU cores and a
240-second deadline. Four dense `.npy` route arrays, 89,533,280 bytes combined,
remain on workstation storage; only hashes and compact receipts are published.
No hardware allocation or weight reads are involved in this audit.

The first audit stopped before execution because a pre-existing contraction
helper used Python 3.11 starred-subscript syntax on the Python 3.10 workstation.
The equivalent tuple concatenation fixed portability without changing topology.
The failed source snapshot and resource-release receipt are retained.

The abstract word encodes one RX port and one TX port in six bits; zero disables
the route. It is not an SDK register image. `csl/route_word.csl` separately lowers
these ports using installed SDK enum values, masks only route fields, rejects
malformed words and requires teardown. A small alternating-path experiment tests
that backend; it does not admit transitions between these full forests.
