# Next work: resident spatial pipeline

1. Preserve physical006 as the accepted shared-input complete-MLP baseline:
   all original numerical/retention/drain checks pass,49376->24688 broadcast words,
   nonzero counts564289–564333. Four host observations improve versus P29 but
   remain slower than P28; no sustained wall speedup is established. Physical
   maximum48128incl4096stack controls admission, despite local compile48112.
2. Preserve accepted physical008 as the residual/RMS+MLP+residual/next-RMS
   integration baseline:81920 exact BF16 values, all native operands/counters,
   resident bank/gain retention, warm replay and normal stop.007's timeout and
   obsolete output-grant IDs remain preserved;148 source/publication tests pass.
   The state-address resolver now checks all49152 original pages through the800
   relocations and must be used by mixer lowering. Keep this integrated graph;
   advance to complete mixer/layer execution rather than an open-ended isolated
   optimization sweep. Host arm/start/finish and phase logging are diagnostic;
   device-driven admission and full backpressure remain future serving work.
3. P33 shares original BF16/E4M3 input preparation:133 words preserve raw BF16,
   exact FP8 code and group scale. All3440 matched mixer PEs shrink720–976bytes,
   but full compile015 still fails SRAM. Selected backend012 measures actual producer
   plus cohost code at explicitly reduced calibration payload; it is not full
   bank admission. Both are released. Next consume original A/B inputs during
   ingress where their exact one-row ownership permits shortening raw-cache
   lifetime; use measured code/workspace demands for data-conserving placement,
   followed by complete SRAM admission. Preserve4KiB stack and request capacity.
   Connect original GDN/conv/gates, attention and4x8 FP32 state pages to the
   resident projections; validate adjacent complete layers on actual requests
   with backpressure, state isolation and warm reset. Preserve every original
   matrix dimension, operation, rounding boundary and model/token identity.
4. Measure real stage and full request service. Current packing admits only two
   context96 request states; it does not establish concurrency sufficient for
   >=2000 aggregate tokens/s. Revisit bank program/loader size and distributed
   state capacity using compiled footprints and actual request-cycle times.
   Do not infer completed tokens from intermediate activations or raw counters.
5. Instantiate embedding, all64 resident stages, the complete head and dependent
   token feedback. Predeclare request mix and steady-state window, generate
   correct complete sentences, and report aggregate completed-output throughput,
   TTFT, ITL, initialization, prefill, fill/drain and full run separately.

Native pair simulator003 is slower on the used shapes. Native unroll005 saves
local simulator counts but complete compile008 exceeds SRAM on68PE. Preserve
these observations and the syntax failure004; neither variant is selected for
hardware. The shared-input candidate retains the original native kernel.

Retain all qualified kernels and failed snapshots. Do not dispatch the retired
whole-wafer temporal overlay, unsupported WSE-3 color swap or simultaneous
RAMP/cardinal receive. Publish evidence and verify resource release after every
bounded experiment. No partial graph passes complete-model acceptance.
