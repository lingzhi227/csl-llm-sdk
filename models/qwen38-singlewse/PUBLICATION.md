# Qwen3.8 single-WSE source publication

This directory is the complete resident FP8 Qwen3.8 implementation. It is
separate from the older root-level three-stage BF16 Qwen development and from
`models/gpt-oss20b/`. `SOURCE_EXPORT.json` identifies the exact physical attempt
and compiled device source in a publication. The uncompiled performance
experiment remains local and is excluded from this functional release.

This initial functional release is recorded in
`evidence/functional-milestone-001/COMPLETE.json`, under the user's explicit
phase-closing instruction. Strict numerical acceptance is false and remains
pending; the compiled and captured full-model execution is documented separately.
A future complete numerical-acceptance claim must have the physical attempt's
`acceptance/COMPLETE.json`; readable output alone cannot establish that claim. See the [acceptance contract](docs/ACCEPTANCE.md) and
the frozen numerical thresholds in `configs/full-acceptance-v1.json`.

## Contents and integrity

The package contains device CSL, complete generated layout, original-weight
packing and initialization, bounded runtime, operator experiments, original
reference sources, numerical evaluators and compact evidence. All published
current device files are byte-identical to the actual successful compile's
source manifest. `SOURCE_EXPORT.json` maps original files to published hashes
and explains omitted files. The repository-wide `PUBLIC_MANIFEST.json`
identifies the public bytes.

Host deployment paths have explicit site substitutions. Those publication
edits were not physically rerun. Historical manifests and hashes continue to
identify the real execution files; do not rewrite historical manifests to
describe deployment edits. Original upstream reference source and all device
arithmetic are unchanged.

Weights, tokenizers, raw tensor arrays, compiled artifacts, ELF files, SDK
distributions, authentication material, operational logs and transient dispatch
state are excluded. Checkpoint hashes, numerical summaries and artifact/source
identities remain. Full raw captures remain on the original execution hosts;
fresh captures or access to those originals are required to repeat the complete
offline numerical comparison.

## Site configuration

Run tools from `models/qwen38-singlewse/`. Configure these placeholders before
using the retained deployment templates:

| Placeholder | Purpose |
|---|---|
| `/srv/model-storage/qwen38-singlewse` | Workstation checkpoint, fixtures and reference runs |
| `/srv/qwen38-singlewse-hardware` | Hardware-side checkpoint, compile and runtime attempts |
| `/opt/cerebras/sdk/2.10.1` | Separately installed compiler/simulator wrappers |
| `/opt/cerebras/venv/bin/python` | Hardware Python with the Cerebras SDK |
| `/srv/cerebras-hardware/hardware.lock` | Shared lock used by all physical workloads |
| `/srv/cerebras-workstation/heavy.lock` | Shared lock for bounded heavy workstation work |
| `/path/to/alcf-session.sh` | Authenticated wrapper accepting `host COMMAND` and forwarding stdin/stdout |
| SSH alias `workstation` | Preparation and independent CPU reference host |

Physical attempts retain wall-clock and client/server memory limits and only
clean up invocation-correlated owned jobs. CPU reference runs use a 26 GiB
memory cap, no swap, two CPU threads and a runtime bound. Storage guards must
refer to the actual deployment volume. Adjust site paths and fresh attempt
references without removing these controls.

See [the reproduction guide](docs/REPRODUCING-FULL-INFERENCE.md) for acquisition,
actual ELF SRAM admission, role binding, frozen requests, captured numerical
comparison, reset/replay and release checks.

## Scope and licenses

The full text network is dense with 48 Gated DeltaNet and 16 full-attention
layers. Its context capacity here is 96 positions. Vision and optional MTP
branches are outside the text workload. The reported latency belongs to this
unoptimized CSL implementation and includes host observation boundaries;
it is not a WSE-3 peak-performance or service benchmark.

Original project code follows the repository [MIT license](../../LICENSE).
Original Transformers and vLLM files retain their Apache-2.0 licenses and
notices; see [reference provenance](reference/PROVENANCE.md). Model payloads
retain the publisher's own license and are not redistributed here.
