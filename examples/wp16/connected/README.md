# Connected partial MLP with generation reuse

Accepted SDK candidate `wp16-connected-sdk-003` executes dense, changed and zero
inputs through four PEs in one runtime: resident 128×112 gate/up projections,
device SiLU/product and a 128×128 partial down projection. Explicit reset,
retention and release barriers protect generation ownership.

Read the [report](../../../docs/WP16-CONNECTED-MLP-REPORT.md),
[observations](../../../evidence/wp16-connected.json) and
[source mapping](../../../evidence/wp16-connected-source-map.json).

`layout.csl` places the four roles; `tile.csl` contains the generated program.
`tile_body.csl.inc`, `lifecycle.csl.inc` and `build_sources.py` retain its source
provenance. `protocol.csl` defines the device frames. `driver.py` and `checks.py`
execute the exact accepted schedule and observations. Layout/protocol/source
helpers live in `core/qwen38/wp16_reuse_*.py`. The qualification ledger is the
original proposal; the protocol test applies the documented timing-width correction
before comparing it with the accepted schedule.

From the repository root, with Python and NumPy available:

```sh
PYTHONPATH=core python3 -B -m unittest discover -s tests -p 'test_wp16_reuse_*.py' -v
```

These 28 synthetic host test groups cover protocol, validation, resource refusal,
and the real driver/evidence loop using a mocked transport. They need no private
files, weights or SDK installation. The exact failed002 driver under `qualification/`
is a regression fixture only; the test demonstrates its premature archive flush.

The source map distinguishes 40 exact accepted source files, two unexecuted
portable configuration templates and that failed reference. The templates
`wp16_reuse_sdk_admission.py` and `wp16_reuse_source_admission.py` remove local
installation/cache paths and stamps. Their placeholders refuse normal execution.
Configure, freeze and admit a separate candidate with matching private source
inputs before attempting SDK reproduction. Importing or testing a template does
not constitute an SDK execution. Weights and vendor binaries are not distributed.

The full-width one-input [diagnostic](../diagnostic/README.md) is a separate
accepted profile. Full 5,120-column connected reuse and the whole 17,408-channel
MLP remain open; the fixed ranges here do not implement a generalized scheduler.
