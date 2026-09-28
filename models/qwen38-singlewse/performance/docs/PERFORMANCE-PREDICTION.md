# Evidence-scoped spatial pipeline performance prediction

This tool supports layout and kernel decisions before expensive compilation or
physical execution. It reads the same emitted stage map and layer-kernel/stream
IR used by development. It never changes model weights, launches a job or turns
an estimate into physical acceptance. The September28 target is now one stateful
multi-turn conversation at>=2000 average output tokens/s, under
[DIALOGUE-ACCEPTANCE.md](DIALOGUE-ACCEPTANCE.md). The existing abstract multi-request
aggregate report is not this acceptance metric. Use a single active conversation
for dependent-path latency screening; its lack of prompt ingestion, turn-boundary
state and EOS timing prevents a multi-turn response-speed prediction. Preserve
older aggregate scenarios as historical capacity studies, not target evidence.

## Implemented now

`tools/predict_pipeline.py` produces a JSON report, a readable report, and an
exact-plan-bound template for future measured/assumed stage costs.

1. **Placement and provenance.** Reject unadmitted placement and mismatched
   kernel regions/matrices. Record SHA256 of all inputs and prediction sources.
   A service scenario must match the exact stage-map hash; changed layout cannot
   silently reuse an old timing profile or assume more resident request slots.
2. **Measured primitive reuse.** Import the physical P17 native-shape evidence.
   Match dtype and exact tile dimensions; include per-invocation FP8 decoding.
   Apply the original cyclic tile ownership/start offsets to estimate native
   busy cycles for each region's most heavily loaded PE. Report min/median/max
   measured samples and expose the tile implementation reuse assumption.
   These are partial resource costs, not whole-layer latency or formal hardware
   lower bounds. Quantization, reduction, control, state operations and overlap
   remain outside this component model. A different kernel requires recalibration.
3. **Communication screening.** Count packet/header words on the actual declared
   adjacent-stage lanes, respecting chunk-to-lane mapping and accumulating shared
   directed-link loads. This does not substitute adjacent endpoints for complete
   producer-to-port/port-to-consumer routes. Interior traffic, multicast, credits,
   queues and feedback remain visibly unmodeled until their lowering is available.
4. **Finite-request event model.** Given complete, sourced stage latency and
   initiation interval values, simulate finite stage slots, downstream backpressure,
   pipeline fill and feedback. A request's next position is admitted only after
   its previous modeled output completes and the declared feedback delay expires.
   Latency, launch interval and buffer capacity are distinct. Warmup and sample
   windows are explicit. This is a deterministic queue abstraction, not a CSL
   simulator, packet-level protocol proof, actual sentence generation or a model
   of prefill/EOS/replacement. The stage slots abstract buffer ownership; exact
   send-callback/consumer-credit lifetimes still require the production protocol.
5. **Missing data stays missing.** No clock frequency is assumed. Cycle-to-TPS
   conversion requires an explicit frequency and its provenance. Service-profile
   blanks fail rather than becoming zero-cost events. BF16 1x128 row tiles are not
   assigned the cost of the older 2x128 probe. Full-model TPS remains unavailable
   when only the partial native-cost model has been supplied.

## Run on the current frozen spatial layout

From the project root:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 performance/tools/predict_pipeline.py \
  --plan performance/evidence/pipeline-stage-map-002/stage-map.json \
  --ir performance/evidence/pipeline-stage-map-002/layer-kernel-ir.json \
  --output performance/evidence/pipeline-prediction-NEW
```

The output directory must be fresh. Use `--scalar-scales` to screen the older
scalar-scale implementation instead of the qualified vector-scale primitive.
This changes an explicit reuse scenario, not the production code automatically.

To run the queue model, copy the emitted `service-profile-template.json` to a
new scenario file. Supply positive `latency_cycles`, `ii_cycles`, the actually
placed `slots`, and a `source` for every stage. Fill `feedback_cycles` and
`feedback_source`. Keep `clock_hz` null to stay in cycles, or provide both it and
`clock_source`; label hypothetical frequencies as assumptions. Then pass
`--service-profile path/to/scenario.json`. Optional `--concurrency` may reduce
the placed request count; increasing it requires another admitted stage map.
Do not tune unspecified values until the resulting number happens to reach 2000.

For uncertainty analysis, run explicit favorable/central/adverse service profiles
with their assumptions recorded, comparing the resulting queue rates. Those are
scenario sensitivity ranges, not statistical confidence intervals. The narrow
min/max spread in the native microbenchmark is not the uncertainty of a new
full-layer implementation. Scheduling, interior traffic and precision-changing
code cannot be hidden inside an arbitrary universal correction factor.

## Interpretation and current finding

For a declared concurrency C and target X, steady state requires mean request
feedback cycle <= C/X seconds, as well as adequate service capacity everywhere.
With the current two-request map and X=2000, this necessary budget is 1 ms. It
does not assert a measured 1 ms cycle or that two requests fill 64 spatial stages.
The queue model separately exposes limits from long feedback cycles, slow stages
and insufficient buffer slots.

The matching native shapes cover about 94.95% of matrix MACs in stage-map-002.
This is arithmetic coverage, not timing coverage. That layout assigns about134
2x128 FP8 invocations to a busiest gate/up or down PE; reusing the measured
decode-plus-native primitive gives about104,034 busy cycles for each such region.
That observation motivates checking layer-local loop/tiling costs early. It is
not a complete layer time, an immutable bound or a hardware-frequency estimate.

## Incremental calibration alongside layer integration

The executor should keep developing the connected layer0 -> layer1 neural path.
This predictor is decision support, not a new prerequisite framework milestone.

- Add measured new layer-local tile and BF16-tail costs with source/artifact,
  dtype, shape, bank profile, timing boundary and clock provenance.
- Feed emitted full routes and event/resource leases into a link/queue/compute
  event model. Include shared links, serialization, callback/credit stalls,
  RMS/quantization barriers and buffer residency. Total word-hops alone is not
  a latency estimate; a max-link load alone is not a complete critical path.
- Measure connected-stage latency and initiation interval under realistic
  resident weights and multiple requests. Replace assumptions with those values.
- Freeze predictions before the next physical run, hold some cases out from
  fitting, record residual/error by scope, and widen uncertainty where the model
  fails. Do not retroactively relabel fitted observations as predictions.
- Compare stage geometry, tiling, buffer/credit policy, fusion, concurrency and
  context under the same workload. Preserve physical numerical and resource gates.

Validation: `python3 -m unittest discover -s performance/tests -p test_prediction.py`.
The tests include analytically solvable queues, single-request feedback, a
two-request 64-stage pipe, latency/II separation, buffer saturation, shared-link
traffic, stale calibration/plan rejection and missing-cost rejection.
