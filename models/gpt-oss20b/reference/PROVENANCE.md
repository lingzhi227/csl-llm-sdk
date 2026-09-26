# OpenAI reference

`upstream/gpt_oss/torch/model.py` and `weights.py` are unchanged copies from
https://github.com/openai/gpt-oss at commit
`7b583341fe16729127f6d5b94a7b09ccae97e1a1`.

`run_reference.py` maps the Hugging Face checkpoint names to those modules.
It constructs each attention/MLP block on the meta device, assigns original
parameters, and supplies lazily decoded selected experts through the original
indexing interface. The upstream forward methods are unchanged. This avoids
materializing all 20.9B parameters as BF16 in host RAM.

CPU outputs are reference evidence only; they never count as CSL or physical
model execution. The final CSL backend must keep every original weight on WSE-3.
