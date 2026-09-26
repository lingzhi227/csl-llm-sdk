# GPT-OSS-20B inference in CSL

Target: the complete, unpruned `openai/gpt-oss-20b` on one physical WSE-3,
with original MXFP4 expert weights and BF16 non-expert weights resident across
tokens. Host input enters from the west, computation advances east, and output
leaves at the east. No host expert execution or layer-weight reload is permitted
in the final resident backend.

Checkpoint revision: `6cee5e81ee83917806bbde320786a8fb61efebee`.
Reference implementation: OpenAI gpt-oss commit
`7b583341fe16729127f6d5b94a7b09ccae97e1a1`.

Status: the complete original model has passed a bounded **two-token physical
WSE-3 inference** test: `Hello` -> `,` -> ` World`. All weights were loaded once,
retained across both steps and exhaustively read back unchanged. The 870,000-PE
backend's 2009 programs passed SRAM checks; all owned jobs stopped/released.
Independent actual-input BF16 operator qualification passed all 48 layer-step
cases. Strict global bitwise/one-ULP CPU parity did not pass and remains reported.

Initial runtime KV capacity is 96 tokens; the current harness validates this
short prompt, not arbitrary-prompt quality or long-context accuracy. See
[the result report](docs/PHYSICAL-INFERENCE-RESULT.md), [current evidence](STATUS.md),
[backend design](docs/FULL-MODEL-BACKEND-DESIGN.md) and
[reproduction guide](docs/REPRODUCING-FULL-INFERENCE.md).

Model payloads and SDK artifacts are kept on remote storage, outside this tree.
Source, compact metadata, and reproducible preparation tools are kept here.

## Published source and environment

See [PUBLICATION.md](PUBLICATION.md) for exact-source provenance, dependencies,
site configuration, excluded payloads and the distinction between historical
execution hashes and public file hashes. Device CSL and original OpenAI reference
sources are unchanged. Host launchers use explicit site-path placeholders and
require configuration before a fresh admission; they are not a turnkey cloud API.

Project code follows the repository [MIT license](../../LICENSE). The unchanged
OpenAI reference retains [Apache-2.0](reference/upstream/LICENSE).
