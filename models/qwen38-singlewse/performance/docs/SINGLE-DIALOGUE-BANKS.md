# One persistent conversation: resident bank specialization

The active workload is one continuing conversation. The previous two-request
layout reserves a second independent convolution and recurrent state in each
GDN layer. The new storage lowering keeps the first complete conversation and
both temporary work slots, while removing only the second request's state.
It does not shorten history, reset retained state at a turn boundary, remove
weights, change arithmetic, or qualify a dialogue runtime.

For the complete original layer00, the candidate retains28605 auxiliary128B
pages and removes25056 pages belonging exclusively to the second context:
480 convolution-history pages and24576 recurrent-state pages. The original
396953216-byte bank allocation becomes393746048bytes, reclaiming3207168bytes.
Every matrix and original MLP bank prefix remains intact. Retained pages stay
on the same PE; only physical tail offsets are compacted. This avoids adding
communication distance while making the released bytes actually available to
future consumer code and scratch.

The map retains original logical page IDs. Removed pages are holes, not new
addresses for another object. `DialogueAuxiliaryPlacement` rejects those holes
and any GDN request other than0. Its audit independently requires all original
weights, both work slots, every retained state page, disjoint contiguous bank
tails and exact byte conservation. The compiled source includes an explicit
`bank-specialization.json` selector; `joint-stage.json` and
`joint-bank-placement.json` remain the original two-context provenance.
Consumers must use the selected dialogue resolver rather than those source maps.

The complete builder preserves every non-layout CSL file, original matrix
descriptor and fabric route from compile018/P39. It changes only bank lengths
in PE parameters. Compiler admission still needs a fresh complete-program census:
global reclaimed bytes alone do not admit any specific fused consumer.

`prepare_dialogue_banks.py` operates only on bounded remote storage. It checks
the original complete-bank hashes, classifies every original word as retained
exactly once or discarded second-context state, and compares every output word
on a fresh read. A nonzero discarded state is rejected: fixture preparation must
not silently destroy an active second conversation. All original matrix/MLP
prefixes, auxiliary weights and retained state/work buffers are preserved.

This is a prerequisite for placing conv/GDN consumers beside their data. It does
not implement those consumers, qualify longer context than the historical96
positions, demonstrate stateful multi-turn generation, or establish any TPS.
Those remain the active acceptance gates in [DIALOGUE-ACCEPTANCE.md](DIALOGUE-ACCEPTANCE.md).

## Turn admission and pending output

`spatial/dialogue_protocol.py` adds a lifecycle oracle for that same single
conversation. It keeps the selected transcript prefix separate from the number
of positions committed through the complete model. A new turn must extend the
exact prior token IDs, including the actual final assistant token or control
marker. That pending token is the first input committed before the new suffix;
it is not dropped or committed twice. Intermediate prefix positions cannot emit
assistant output. Context is reserved before accepting a turn, with no implicit
history truncation. Output limits include control tokens; content counts are
reported separately under the caller's declared classification.

Completion requires all66 distinct stage retirement acknowledgements and the
matching request/generation/absolute-position tag. Turn end retains state. Only
an explicit idle reset with all66 state-clear acknowledgements creates a fresh
generation. Adversarial tests cover three continuing turns, a pending EOS,
stop/continue after a content token, duplicate/stale completions, missing or
duplicated acknowledgements, prefix mismatch, capacity overflow and reset/replay.
These tests supply synthetic acknowledgement/token events. They establish the
admission contract only; no device event path, multi-turn model output or speed
is admitted by this protocol oracle.

## Bound evidence

Compile019 exceeded its600s limit before producing an ELF census. Compile020 was
stopped after observing the2GiB cgroup ceiling with35994 memory.max events and
no OOM. The same29 CSL files then passed compile021 in342.295s under3GiB/noSwap,
two allowed CPUs and a900s compile limit. Every11388 PE passes;7910 ELF images
cover the rectangle once, with maximum48112bytes including the4096-byte stack.
Failures and invocation-correlated release receipts are preserved.

The independent complete-census comparison finds exactly3207168bytes less SRAM
occupied. It retains all per-PE coordinates/margins for consumer placement. Only
59 non-gateway PEs have at least8192bytes free, and many still have very little
space. The minimum margin is16bytes. Consumer code, phase buffers and routes
must be admitted with their actual cohost programs; the global gain is not a
fusion admission. See `evidence/dialogue-bank-census-001/result.json`.

Remote reference001 classifies every original word exactly once, discards only
801792 zero-initialized second-context words, and freshly verifies all98436512
retained words. It preserves389796352bytes of initialized weights/scales and the
two work slots. Preparation takes6.816s under2GiB/noSwap/one CPU/180s. No model
arrays or compiled binaries are copied to the Mac client.

Twenty targeted state/turn/joint-bank/mixer-interface tests pass in both source
and publication trees; this is a
selected regression suite, not a rerun of all earlier component tests. All four
workstation services are released and no new hardware job is submitted. The
ALCF accounting snapshot reports no owned active job or system assignment.
The bound summary is `evidence/single-dialogue-summary-001.json`.
