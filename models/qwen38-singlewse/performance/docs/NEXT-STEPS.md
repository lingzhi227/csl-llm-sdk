# Next work: resident spatial pipeline

1. Keep accepted physical `layer-mlp-hw-004` as the complete original MLP
   baseline:40 parallel initial quantizers, vector controller copies,713625–713752
   nonzero controller ticks and2.388–2.801ms completed-output host timings. Preserve
   the frozen original4-case outputs/ingress/counter/retention/drain contract.
2. Coalesce all native slices/aliases of a128-value group into one immutable
   controller packet (max296u32), reducing864 send callbacks to176. Prefetch the
   next exclusive return frame while the current packet broadcasts, using
   distinct frame/packet ownership and a finite one-frame credit. The previous
   complete response must still arrive before granting the next bus source;
   sender OQ flush must still precede RX restoration. Recheck all SRAM/UT/DSR
   lifetimes and adversarial delays before the same full-MLP physical comparison.
   The measured middle405913-tick interval includes both neural work and transport;
   do not label all of it communication overhead. Then partition the shared
   distributor where whole-component measurements justify direct fused paths.
3. Fuse down output with residual, RMS/norm and successor input; compose original
   GDN/conv/gates, attention and4x8 FP32 state pages on disjoint resident regions.
   Admit all simultaneous code/queue/DSR/UT use and validate adjacent full layers
   on two actual requests with retained state, backpressure and warm reset.
4. Measure complete stage service times and traffic. The single granted bus is
   an initial integration mechanism, not a maximum-speed claim. Replace its
   bottleneck through measured sharding/overlap while preserving one WSE-3 RX,
   fixed queue colors, empty-queue route transitions and original arithmetic.
5. Rebalance memory/areas/concurrency and instantiate embedding, all64 stages,
   full head and token feedback. Run correct complete sentences and the declared
   >=2000 aggregate generated tokens/s contract; publish evidence and verify
   resource release after each bounded experiment.

Preserve rejected compiles and exact source snapshots. Do not dispatch the
retired whole-wafer temporal overlay, WSE-3 color-swap candidate or the rejected
simultaneous RAMP/cardinal return bus. No partial scope establishes model TPS.
