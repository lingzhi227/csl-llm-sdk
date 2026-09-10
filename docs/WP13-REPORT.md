# WP13 accepted result

**Accepted September10,2026:** original layer3/head0 Q256/rawgate256/K256/V256,
all5120 input columns and four common synthetic BF16 inputs, covered by eight
independent single-PE four-call runtimes. All1024 rows appear exactly once. This
qualifies the selected-head projections across separate runs; simultaneous full-head
production, device attention consumption and complete-model inference remain open.

The [design](WP13-DESIGN.md) defines the arithmetic and precision scope, and the
[example guide](../examples/wp13/README.md) gives reproduction commands. The
[evidence index](../evidence/wp13.json) records acquisition, CPU comparison,
independent acceptance, source/compiled identities, all numerical arrays and logs.

## Independent validation

Independent checks from original row-major weights covered12288 prefix/final FP32
values,4096 source BF16 outputs and actual RNE casts,2048 exact one-hot/zero outputs,
114704 final resident-weight containers including guards,32 release states and1472
raw timestamp pairs. All4824 copy entry/exit pairs and1568 launches matched the
independently reconstructed sequence. Observed official/nominal BF16 differences
were0; maximum independent FP32 absolute error was3.6670826375484467e-09.

All eight source manifests match outside profile/steps, all four input hashes agree,
and rows0..1023 have no omissions or duplicates. The entire workload ran serially:
a fresh frozen directory per slab, actual ELF admission, all four calls in one
runtime, complete validation and exact-unit cleanup before the next slab. No failed
baseline, retry, changed input, widened source interval or extra model download was
needed. All16 owned compile/sim units were independently checked absent/inactive,
with MainPID0 and empty cgroups; all8×24 frozen and8×12 compiled files stayed unchanged.

## Batches and retained evidence

| Slab | Original selected rows | Sim seconds | Peak bytes | Numerical arrays | Transfer journal |
| --- | --- | ---: | ---: | --- | --- |
| 0 | Q global[0:128] | 207.314002 | 227254272 | [Arrays](../evidence/wp13-slab00-observations.json) | [Journal](../evidence/wp13-slab00-journal.jsonl) |
| 1 | Q global[128:256] | 207.167345 | 227225600 | [Arrays](../evidence/wp13-slab01-observations.json) | [Journal](../evidence/wp13-slab01-journal.jsonl) |
| 2 | rawgate global[256:384] | 210.655047 | 227966976 | [Arrays](../evidence/wp13-slab02-observations.json) | [Journal](../evidence/wp13-slab02-journal.jsonl) |
| 3 | rawgate global[384:512] | 207.562821 | 227168256 | [Arrays](../evidence/wp13-slab03-observations.json) | [Journal](../evidence/wp13-slab03-journal.jsonl) |
| 4 | K global[512:640] | 209.464358 | 227139584 | [Arrays](../evidence/wp13-slab04-observations.json) | [Journal](../evidence/wp13-slab04-journal.jsonl) |
| 5 | K global[640:768] | 220.160653 | 227819520 | [Arrays](../evidence/wp13-slab05-observations.json) | [Journal](../evidence/wp13-slab05-journal.jsonl) |
| 6 | V global[768:896] | 208.088571 | 227643392 | [Arrays](../evidence/wp13-slab06-observations.json) | [Journal](../evidence/wp13-slab06-journal.jsonl) |
| 7 | V global[896:1024] | 207.171100 | 227201024 | [Arrays](../evidence/wp13-slab07-observations.json) | [Journal](../evidence/wp13-slab07-journal.jsonl) |

Each batch used603copies,196launches,10,714,296host bytes and5,405,728native bytes.
Totals are4824copies,1568launches,85,714,368host bytes and43,245,824native bytes.
The largest physical host buffer was57352bytes. Simulation/validation totaled
1677.583898seconds (27.96minutes); the slowest batch was220.160653seconds and largest
simulation memory peak227,966,976bytes. All stayed below240second estimates and
300second hard limits. Actual ELF end+4096stack ranged41552–41568, below49152.

Public typed arrays preserve every input/output/prefix/cast/state observation and
per-array hashes. Original downloaded weight files and complete resident-weight
readback payloads are excluded; the latter retain their shapes, hashes and successful
comparison records. Full original observations remain archived with hashes in the
index. The [input artifact](../evidence/wp13-inputs.json) and
[CPU observations](../evidence/wp13-reference-observations.json) preserve the reference
inputs, FP64 dots, absolute-product sums, source radii and observed official results.
No SDK binary, private filesystem path or unaccepted integration code is published.

## Preceding diagnostic and preparation

Before the full batches, a four-PE/one-dense-case diagnostic covered Q/rawgate512
across all5120 columns. It passed in233.914793seconds with231,018,496bytes peak,
612copies and49launches, source/RNE/parity checks and normal stop. Its
[observations](../evidence/wp13-diagnostic-observations.json) remain supporting evidence,
not a substitute for four-case coverage. Durable identified logs and finally-stop
handling were refined during unexecuted review freezes; the arithmetic-body AST
was unchanged. The accepted persistent GEMV kernel stayed byte-identical to WP03.

Rapid host launch returns did not represent isolated compute time: the diagnostic's
output readbacks occupied161.562453seconds and four complete weight readbacks
57.829797seconds including pending work. Measured behavior justified bounded serial
batches rather than an eightfold larger simulator job. Timing scopes are host
simulation/validation and device cycle counters, not physical-wafer performance.

The original acquisition used12 exact206 ranges,10,486,784bytes and a16MiB cap;
it completed in4.158541seconds with22,532,096bytes peak. CPU reference001 completed
in3.124574seconds with384,024,576bytes peak under2GiB/Swap0/60seconds, without loading
full projection matrices, a checkpoint/model, installing dependencies or using GPU.
The portable public preparers reproduce exact hidden bytes, source closure and
formulas; path/dependency-description metadata may differ from historical freezes.
Their equivalence checks do not constitute another CPU or SDK arithmetic experiment.
