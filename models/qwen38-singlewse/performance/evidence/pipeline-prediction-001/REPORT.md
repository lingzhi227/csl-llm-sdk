# Pipeline performance screening

Evidence-calibrated partial resource-cost screening, NOT a full-model TPS prediction

- Resident concurrency: 2; context: 96.
- Target: 2000 aggregate generated tokens/s.
- Necessary mean request cycle at that concurrency: 1 ms.
- Matching native-shape MAC coverage: 94.95%. This is not timing coverage.
- Full-model TPS prediction: unavailable (unmeasured components and missing clock calibration).

## Conditional native-work bottlenecks

| Region | PEs | Median native busy cycles |
|---|---:|---:|
| layer_00/gate_up | 5226 | 104034.25 |
| layer_00/down | 2607 | 104034.25 |
| layer_01/gate_up | 5226 | 104034.25 |
| layer_01/down | 2607 | 104034.25 |
| layer_02/gate_up | 5226 | 104034.25 |
| layer_02/down | 2607 | 104034.25 |
| layer_04/gate_up | 5226 | 104034.25 |
| layer_04/down | 2607 | 104034.25 |
| layer_05/gate_up | 5226 | 104034.25 |
| layer_05/down | 2607 | 104034.25 |

These reuse measured tile costs; they exclude communication and other operators.
Observed sample min/max are not prediction confidence intervals.

## Missing costs

- New layer-local kernel instruction schedules and composed SRAM
- BF16 row-tail compute and embedding lookup
- GDN/attention state operations, nonlinearities, normalization, quantization and full-head selection
- Interior redistribution, reductions, multicast overlap, link/queue conflicts and completion latency
- Host observation, feedback, request admission, prefill and reset
- Calibrated cycle frequency and end-to-end stage latency/initiation intervals
