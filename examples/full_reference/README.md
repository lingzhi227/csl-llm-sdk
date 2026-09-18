# Full original text-model CPU reference

The accepted reference runs all 64 original Qwen3.8-27B text layers and the full
248320-token vocabulary head. It generates four greedy tokens from one fixed
raw-text prompt, saving every layer boundary and original cache state. This is
a nominal CPU baseline, not full CSL execution or hardware acceptance.

The four Python modules are byte-identical to the accepted frozen worker.
`pinned.py` extracts selected definitions and the complete original text-model
forward body from the hash-bound upstream files listed in `upstream-sources.json`.
Upstream source files, checkpoint weights, SDK distributions and the 800445000
bytes of saved arrays are not included. Published receipts and the array ledger
record their identities without copying those payloads into Git.

For reproduction, supply the exact upstream files under `sources/`, copy the
published tensor headers to `metadata/tensor-headers.json`, and configure a local
`checkpoint-plan.json` from the portable template. The checkpoint reader requires
all 18 complete, hash-verified original files, eight tokenizer/configuration
assets, and a checkpoint completion receipt; it checks all original identities
again before tensor use. The observed environment was Python 3.11.14,
Torch 2.4.0+cu121 and NumPy 1.25.0, using CPU tensors and one logical CPU.

These are the worker modules, not a replacement for the bounded execution
supervisor. Invoke `run_reference.run(check)` only inside an equivalent resource
supervisor whose `check` callback enforces the frozen limits: 7200 seconds,
20 GiB RAM, no swap, 128 tasks, 8 GiB available system RAM reserve, 4 GiB evidence,
2048 files, 32 MiB per evidence file, and 2 MiB logs. Source imports do not start
the model, download files or allocate an accelerator.

The [report](../../docs/FULL-REFERENCE.md) explains the source adaptations,
independent saved-artifact audit, numerical scope and remaining device work.
