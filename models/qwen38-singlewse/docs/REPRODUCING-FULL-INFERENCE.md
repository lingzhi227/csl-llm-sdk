# Reproducing full resident inference

The implementation executes the complete 64-layer text network of
`Qwen/Qwen3.8-27B-FP8`, revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`, on one physical WSE-3.
The context capacity is 96 positions. Vision and optional MTP are outside this
text-only workload. The model, tokenizer, all layer weights and complete
248,320-entry output head are original; this is not a 20-layer stage.

## Environment and source identity

Hardware execution used Cerebras SDK 2.10.1. The independent reference uses
PyTorch `2.6.0+cpu`, NumPy `2.2.6`, tokenizers, and the pinned Transformers
source in `reference/sources/`. Its extracted mathematical methods and the
FP8 checkpoint adapter are recorded in the reference qualification. Install
the vendor SDK separately and obtain authorized WSE-3 access.

Run tools from the model directory. The public host launchers are deployment
templates: configure the storage roots, authenticated session wrapper, SDK
Python, workstation SSH alias and shared lock locations listed in
`PUBLICATION.md`. Do not remove resource bounds or the active-job admission
checks. Give every run a fresh attempt directory. Historical receipts and
source manifests must remain unchanged.

The exact compiled device program is `resident/full-layout.csl` plus the four
files in `resident/generated/` and their imports in `csl/`. The composition
sources in `resident/*.cslpart` and the layout/build tools reproduce those files.
Historical comments describing a candidate remain part of the frozen source;
the completion receipt, not a code comment, determines acceptance.

## Checkpoint and compilation

1. Acquire the pinned checkpoint using `tools/acquire.py`. Verify every file
   against the publisher metadata in `configs/hub.json`, inspect the safetensors
   index with `tools/inspect_checkpoint.py`, and run `tools/audit_checkpoint.py`.
   This acquisition includes 74 files and 30,879,968,808 bytes. The text network
   uses 1,251 tensors with 29,468,003,328 payload bytes.
2. Build or verify the placement, role map, device tables and forward program
   using the `tools/build_*` sources. The runtime uses the BF16-140 atlas,
   `role-plan-96.json`, `device-matrices-bf16-140.json` and `forward-program.json`.
   Earlier 144-row proposals are historical and are not runtime coordinates.
3. Run the representative-role compile and its actual ELF SRAM checks, then
   stage the complete layout with `tools/stage_full_compile.py`. Both compiler
   and runtime source/artifact transfers use the qualified bounded upload
   adapters. A new compiler version requires fresh qualification.
4. Compile the 750-by-1160 application on the 762-by-1172 fabric at offset (4,1),
   with `--arch=wse3 --memcpy --channels=1 --max-parallelism=1`. Check every ELF,
   source identity and placement rectangle. The declared application ceiling
   is 48,128 bytes including a 4,096-byte stack allowance per PE. This allowance
   is an admission budget, not a measurement of peak runtime stack use.
5. Bind every compiled role and exported object extent with
   `runtime/role_binding.py`. Check original parameter packing with
   `tools/qualify_resident_parameters.py`. Compilation, role binding and weight
   byte packing are prerequisites; they do not establish numerical inference.

Pass the newly qualified representative attempt with `--role-compile`, and the
full compile and binding attempts with `--compiled` and `--binding` when staging
the next steps. Some historical defaults refer to earlier code snapshots and
are rejected by source identity checks after a device change. Transport
prerequisite names are also explicit. For a new deployment, set these references
to its newly qualified attempts and
preserve their source manifests. Never reuse the old receipts as proof that a
new binary or edited host driver passed.

## Physical requests

`tools/stage_resident_run.py` freezes the runtime sources and binds the exact
compiled artifact. `tools/dispatch_hw.py` starts its bounded supervisor. The
supervisor holds the shared hardware lock, rejects another active account job,
and only cleans up invocation-correlated owned jobs.

`runtime/resident_loader.py` uploads all original text tensors once, checks
sampled stored matrix tiles and the original small parameters, and checks all
duplicated GDN/attention parameter groups. This is not exhaustive post-run
weight readback. No weights, hidden states, router decisions or next-token IDs
are uploaded between dependent steps.

`runtime/resident_run.py` follows the frozen `full-acceptance-v1.json` order:
the sunny-morning request with full numerical capture, a greeting after reset,
and two sunny-morning requests with capture disabled. The first call consumes
the complete prompt and emits its first generated ID. Later calls merely
trigger the next step; embedding receives the device-selected preceding ID.

The driver saves all 64 layer outputs, final normalization and all vocabulary
logits for every processed position of the captured request. It checks all
870,000 endpoint traces after each request and preserves failures. Text/token
observations and a `CANDIDATE.json` alone do not imply numerical acceptance.

## Independent validation and timing

The sequential reference runs one position at a time on CPU. It may precompute
the fixed prompt before capture arrival. `tools/transfer_sequence_candidate.py`
streams the immutable physical capture into the reference host and checks its
hashes. Candidate token IDs supply the prefix only to this offline reference;
they never become host-provided neural operands for the physical run.

The frozen criteria check every full layer vector and final norm, the complete
vocabulary logits, finite values, deterministic device argmax and the bounded
BF16 reference tie rule. They also require EOS, two complete sentences with the
requested topics, and identical reset/replay token IDs. Thresholds are not
changed after candidate results are observed.

`tools/accept_resident.py` requires both successful numerical completion and a
normal physical stop with released jobs. It binds the checkpoint, compiled
artifact, source manifests, captured request, criteria and reference receipt.
That tool targets strict numerical acceptance, which this initial functional
release did not achieve. `tools/write_functional_report.py` instead records the
completed sentence capture, local checks, observed differences, cancellation and
release under the user-authorized initial milestone; it does not bypass or
rewrite the strict acceptance tool.

TTFT includes prompt/control upload, all prompt processing and first-ID
readback. Each dependent decode time includes the device call and ID readback.
End-to-end time runs from prompt upload to the last ID. Artifact startup and
weight initialization are separate. Headline timings use the two capture-
disabled sentence runs; captured diagnostic timing is labeled separately.
These are measurements of this unoptimized implementation, not a claim about
the maximum performance of WSE-3 or the Cerebras inference service.

## Preserved artifacts

Model payloads, compiled archives, ELF files and raw numerical arrays remain on
the execution/reference hosts. The public repository contains sources, hashes,
compact measurements, qualification summaries and historical failures. To rerun
the offline numerical comparison, obtain the original captures or make a new
physical capture; compact JSON evidence alone cannot reconstruct every value.
