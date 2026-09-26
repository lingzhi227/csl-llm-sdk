# Published GPT-OSS-20B source snapshot

This directory adds the GPT-OSS development line to CSL-LLM SDK without moving
the existing Qwen files. The accepted physical run is `full-model-hw-003`:
complete original model, one WSE-3, two autoregressive tokens, original weights
retained, normal stop and released jobs. Its predeclared actual-input numerical
qualification passed. Strict global CPU parity remains false.

## What is included

* Every CSL kernel, generated layout, host driver, checkpoint reader/packer and
  operator/full-model experiment from this snapshot.
* The original OpenAI Torch reference, pinned source provenance and Apache-2.0
  license; CPU reference and independent numerical qualification tools.
* Acquisition hashes, SRAM records, numerical observations, completion/resource
  audits and historical failure receipts as compact JSON.
* Complete-model design, numerical contract, accepted result and reproduction
  documentation. English source documentation and the Chinese physical result
  report retain the same scope.

Model weights, NPZ/NPY captures, binaries, ELF files, compiler/SDK distributions,
session credentials, raw operational logs, dispatch receipts and stale
progress snapshots are not distributed. Their relevant hashes and numerical
summaries remain in evidence. The absence of payload files means the public
receipts alone cannot replay the independent numerical audit; fresh captures or
access to the original retained captures are required.

## Integrity and historical hashes

[`SOURCE_EXPORT.json`](SOURCE_EXPORT.json) maps source-file hashes to published
hashes and inventories omitted files. CSL and bundled OpenAI reference files are
byte-identical to the source project. Every published current CSL file and the
full layout also match the successful physical run's source manifest.

Operational host paths were replaced by explicit site placeholders; the exporter
does not claim that those adapted launchers were physically rerun. The current
schedule's `implemented` metadata was updated after acceptance, without changing
its command sequence. Publication notes and model navigation were added.

Historical `source-manifest.json`, `numerical-contract.json`, capture hashes and
qualification hashes continue to identify the actual original execution files.
Do not rewrite them to describe edited launchers. The repository-wide
[`PUBLIC_MANIFEST.json`](../../PUBLIC_MANIFEST.json) identifies the public bytes.
Full-model completion, qualification, observations and weight receipts are copied
unchanged; the retained strict-comparison failure has not been relabeled a pass.

## Dependencies and deployment templates

Preparation/reference code needs NumPy, PyTorch, tokenizers and safetensors.
The accepted reference used PyTorch 2.4, NumPy 1.25 and tokenizers 0.22. Install
compatible versions in a separate environment; a different numerical backend
requires its own qualification. Hardware execution needs a separately installed
Cerebras SDK 2.10.1 and authorized WSE-3 access. Simulator launchers use Singularity.
No vendor package or cloud authentication is supplied by this repository.

Run commands from `models/gpt-oss20b`, since its tools resolve the model-local
`core/`, `csl/`, `experiments/`, `runtime/` and `reference/` directories.
The retained site launchers are deployment templates. Configure these locations
in the operational Python files before using them:

| Published placeholder | Purpose |
|---|---|
| `/srv/model-storage` | Mounted storage volume used by the download guard |
| `/srv/model-storage/gpt-oss20b` | Workstation model, fixture, source and run storage |
| `/srv/gpt-oss20b-hardware` | Hardware-side checkpoint and fresh attempt directories |
| `/opt/cerebras/sdk/2.10.1` | Workstation CSL compiler and simulator entry points |
| `/opt/cerebras/venv/bin/python` | Hardware-side Python with Cerebras SDK |
| `/srv/cerebras-hardware/hardware.lock` | Shared physical reservation lock; all colocated workloads must use the same lock |
| `/path/to/alcf-session.sh` | Your authenticated session wrapper, accepting `host COMMAND` and forwarding standard input/output |
| SSH alias `workstation` | Your preparation/reference host |

The downloader deliberately checks the storage mount and a 48 GiB free reserve:

```sh
python3 tools/download_model.py --destination /srv/model-storage/gpt-oss20b/model
python3 tools/audit_scales.py --model /srv/model-storage/gpt-oss20b/model \
  --output /srv/model-storage/gpt-oss20b/model/scale-audit.json
python3 reference/run_reference.py --model /srv/model-storage/gpt-oss20b/model \
  --output /srv/model-storage/gpt-oss20b/runs/reference-001 --text Hello --steps 2
python3 tools/prepare_transformer_math.py --model /srv/model-storage/gpt-oss20b/model \
  --reference /srv/model-storage/gpt-oss20b/runs/reference-001 \
  --output /srv/model-storage/gpt-oss20b/fixtures/transformer-math-001
python3 tools/prepare_full_fixture.py --reference /srv/model-storage/gpt-oss20b/runs/reference-001 \
  --transformer /srv/model-storage/gpt-oss20b/fixtures/transformer-math-001 \
  --output /srv/model-storage/gpt-oss20b/fixtures/full-model-001
```

Keep operational edits outside historical evidence and use a fresh attempt name.
Follow [the full reproduction guide](docs/REPRODUCING-FULL-INFERENCE.md) for fixture
preparation, physical compilation, every-ELF SRAM admission, inference, complete
retention/readback and post-release qualification. An old completion record does
not authorize skipping those checks for a new deployment.

## License boundaries

Original project code follows the repository [MIT license](../../LICENSE).
`reference/upstream/` is unchanged OpenAI source under its bundled
[Apache-2.0 license](reference/upstream/LICENSE), pinned in
[PROVENANCE.md](reference/PROVENANCE.md). Downloaded model files retain their own
publisher license and usage-policy files; model weights are not republished here.
