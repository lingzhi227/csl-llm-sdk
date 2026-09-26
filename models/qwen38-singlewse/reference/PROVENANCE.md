# Reference source provenance

The official checkpoint is `Qwen/Qwen3.8-27B-FP8` at revision
`017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`. Its configuration names
`Qwen3_5ForConditionalGeneration`; this is the upstream implementation identifier
used by the official Qwen3.8 checkpoint, not a substitution of Qwen3.5 weights.

`sources/` contains unchanged original Transformers files from commit
`27166ea03f12c940f23176a904ab1d2ff1a3dcbb`. URLs and byte hashes are in
`sources/SOURCE.json`. `finegrained_fp8.py` is from the same commit and is pinned
in `SOURCE.json`. These files are under the bundled Apache-2.0
[license](sources/LICENSE); their upstream copyright notices are retained.

`vllm_fp8_utils.py` is unchanged vLLM reference source from commit
`ddd6fbca148a867aad1fcab7ec72f582b9977db4`, pinned in `VLLM-SOURCE.json`,
under the bundled [Apache-2.0 license](VLLM-LICENSE). It documents the dynamic
group-128 FP8 activation convention. It is not imported into physical runtime.

`full/pinned.py` extracts the pinned original model methods with their source
hashes. `full/checkpoint.py` adapts loading to the original compressed checkpoint
and the qualified FP8 arithmetic. `sequence/run.py` evaluates the actual device
prefix offline with one position at a time, retaining every layer and full-head
reference vector. CPU outputs are numerical evidence only. No CPU hidden state,
logit, selection or recurrent update enters the physical model computation.
