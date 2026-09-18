# Full original resident MLP

This directory publishes the accepted complete layer-3 MLP device graph and its
four-input physical capture and offline numerical audit. See the
[report](../../docs/HW01-FULL-MLP.md), [acceptance](../../evidence/hw01-full-mlp.json)
and [exact source map](../../evidence/hw01-source-map.json).

`device/` contains the exact compiled CSL and metadata generator. Compilation used
the appliance SDK compiler with these flags:

```text
--arch=wse3 --fabric-dims=762,1172 --fabric-offsets=4,1 --memcpy --channels=1 --max-parallelism=1 -o out
```

`capture/` contains the exact capture driver, numerical checks, source gate and
release supervisor. Only the offline audit interpreter path is configurable via
`CSL_AUDIT_PYTHON` (default `python3`). This portability edit has not submitted a
new hardware run. The public tree contains no checkpoint payload, compiled ELF,
site credentials or original tensor dumps.

Reproduction requires a licensed copy of the pinned model revision, the private
full reference arrays and row-packed native BF16 payloads with the identities in
`inputs.template.json`. Set its four paths in a new execution directory; the
recorded identities intentionally reject arbitrary substitute inputs. A fresh
compiled artifact needs its own embedded-source, SRAM and unique-placement
inspection, with corresponding artifact and postcheck hashes. Freeze the actual
source/input configuration into a new manifest and review the exact bounded
admission before launching `supervisor.py` in an authenticated appliance Python
environment. It allocates one physical system. Do not run `run_hw.py` without the
remote release watchdog. The template is evidence, not a ready-to-submit run.

After capture, `COMPLETE.json` proves capture/release only. A fresh terminal-job
and system-assignment check must precede the separately admitted
`audit_supervisor.py`; `numerics.json` then reports mathematical acceptance.
Preserve failed runs and partial arrays. No automatic retry is implemented.

The standalone rectangle is a module-validation layout. Full layers and dense
stage placement remain subsequent integration work.
