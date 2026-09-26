# Reproducing the bounded full-model physical inference experiment

This is the functional, batch-one bring-up harness for the original GPT-OSS-20B,
not a serving API. It runs `Hello` (13225), then feeds the actual generated token
back for a second step. The model's complete 24 layers, all 32 experts per layer,
and full vocabulary matrices remain resident. Current KV allocation is 96 tokens;
the two-step experiment does not establish long-context or corpus accuracy.

First configure the published site-path templates as described in
[PUBLICATION.md](../PUBLICATION.md). The original deployment used an authenticated
ALCF session; credentials and that private session wrapper are not bundled. The supervisor holds the same
hardware lock as the Qwen project and refuses allocation while other account jobs
are active. It captures only its own job IDs; failed attempts are never overwritten.

## Remote inputs

The publisher-verified model and receipts are at:

* ALCF: `/srv/gpt-oss20b-hardware/model`
* Workstation: `/srv/model-storage/gpt-oss20b/model`
* Original CPU reference: workstation `runs/reference-001`
* Two-step diagnostic fixture: workstation `fixtures/full-model-001`

Model revision is `6cee5e81ee83917806bbde320786a8fb61efebee`. The fixture's saved
hidden states are diagnostic expectations only. They are never H2D inputs.
RoPE frequencies are initialization constants, calculated once from original
configuration; per-token rotations are computed on the device.

## Fresh attempt

Choose a new, unused `full-model-hw-NNN` directory for every changed source or
driver. From this source tree:

```sh
python3 tools/stage_full_hw.py --name full-model-hw-NNN --cluster-compile-first
python3 tools/dispatch_hw.py full-model-hw-NNN
python3 tools/inspect_hw.py full-model-hw-NNN
```

The published prerequisite receipts describe the accepted historical run.
Staging verifies the generated layout and accepted operator prerequisites.
Compilation must pass embedded-source identity and every actual ELF SRAM gate
before runtime allocation. Shared application geometry is 750 x 1160, physical
fabric 762 x 1172 at offset (4,1), one SDK memcpy channel. Code and data share each
PE's local SRAM; the checked bound includes a declared 4096-byte stack allowance.

Initialization streams all 459 tensors in 73,893 bounded transfers. Each token
then uses one west token H2D and 1990 fixed phase/layer/index commands. Neural
computation, expert selection and layer handoffs stay on wafer. An east token D2H
completes generation. Only afterwards are intermediate diagnostics read.

After two actual autoregressive steps, the driver reads back every loaded weight,
scale, bias, gain and sink, plus initialization identities/frequencies. The runtime
stops normally before the supervisor checks job termination and system release.

## Receipts and numerical review

`PHYSICAL_CAPTURE_COMPLETE.json` means both physical steps and exhaustive
retention finished, not that numerical acceptance necessarily passed.
`result.json` retains the original strict reference comparison. If rounding or
equal-score routing differs, `NUMERICAL_REVIEW_REQUIRED.json` is written instead
of a premature `COMPLETE.json`.

Use `tools/collect_hw.py NAME --capture-only` for compact capture receipts and
`tools/relay_full_capture.py NAME` to stream the two actual NPZs directly from
ALCF to workstation storage. No model, capture, ELF or SDK archive is saved on
the Mac. Run `reference/qualify_full_capture.py` on the workstation with the
original model, relayed capture directory, a fresh output directory and the
predeclared `docs/NUMERICAL-QUALIFICATION.md` contract. Use two CPU threads and
a bounded service; this is offline verification, not part of inference.

The per-attempt `numerical-contract.json` binds the contract and verifier before
observing results. `tools/accept_full_qualification.py` accepts only an all-passed
48-layer-step report, rechecks capture hashes against ALCF, verifies frozen source
identity and current job release, and writes a qualified completion receipt.
It never changes the original strict result. Collect accepted compact receipts
with `tools/collect_hw.py NAME`.

The timing field `forward_host_seconds` includes host control submission and
the east result transfer. It excludes model loading, diagnostics and exhaustive
readback. It is not a pure kernel time or a steady-state performance benchmark.
