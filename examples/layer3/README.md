# Original layer 3 source graph

These CSL files and the placement plan are byte-identical to the accepted
complete-layer compilation candidate. The graph uses 750 x 45 application PEs on
WSE-3 and integrates the original attention, normalization, residual and MLP
operations. The current owner bounds context to eight positions.

Compilation and original-weight preparation are accepted. A complete original
layer 3 numerical epoch is not. Model weights and saved reference/diagnostic
arrays are not included. This folder is source for the integrated graph, not an
out-of-the-box model download or a full-model simulator workload.

The accepted physical compiler configuration used fabric 762 x 1172, offset (4, 1),
memcpy with one channel, and one compiler worker CPU with 4 GiB memory under a
600-second watchdog. It produced 313 application programs and passed actual SRAM
and full-placement checks. New builds require those checks again before any
hardware run. See [the integration report](../../docs/LAYER3-INTEGRATION.md) for
results, boundaries and the distinct later physical observer diagnostic.
