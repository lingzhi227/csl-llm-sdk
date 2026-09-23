# Finite Layer0 arithmetic source

These are the exact nine CSL sources compiled by arithmetic SDK002 and executed
by the accepted finite SIM002 experiment. They are a synthetic nine-PE fixture,
not an original-weight complete Layer0 or a physical hardware result.

The [source map](source-map.json) pins every file. Read the
[report](../../../docs/LAYER0-ARITHMETIC-FINITE9.md) and
[full audit](../../../evidence/layer0-arithmetic/finite9/audit.json) for the actual
coverage, error limits, preserved failures and unqualified scope.

With a separately installed compatible Cerebras SDK, the device compilation is:

```sh
cd device
cslc layout.csl --arch=wse3 --fabric-dims=8,11 --fabric-offsets=4,1 --memcpy --channels=1 -o=out
```

The recorded experiment used SDK 2.10.1's compile wrapper with one compiler worker.
Compilation alone does not reproduce the runtime qualification. A host runner
must bind all 29 exports to their actual role extents, retain all 452 ordered calls
and the finite inputs, enforce the published resource limits, then stop/reap the
simulator before auditing the measured arrays. This source snapshot intentionally
does not bundle the private admission launcher or vendor runtime.
