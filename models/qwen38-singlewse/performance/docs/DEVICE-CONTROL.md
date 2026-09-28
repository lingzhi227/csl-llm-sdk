# Credited control on internal mixer PEs

P35 replaces the SDK memcpy/RPC module on selected internal mixer cohosts with
an actual fabric command port. SDK access remains on a gateway PE. The adapter
retains the existing `arm`, `start`, `finish`, `mixer_begin` and `mixer_consume`
operations; it adds bounded original-bank/configuration loading and readback.
No host-provided address is accepted: role-specific segment IDs resolve to fixed
arrays with checked extents and phase guards. This is a local implementation,
not a routed complete stage or a serving-model throughput result.

## Wire and lifetime contract

A command has six32-bit words: sequence, opcode and four arguments. Sequences
advance exactly once. Opcodes0/1 write/read a segment; arguments identify segment,
element offset and count. Count is bounded to256 and checked against the actual
array. Opcodes2–6 invoke the retained operations above. A four-word response
carries sequence, zero status, payload count and logical element width. Every
wire word is32bits, including explicitly expanded16-bit configuration elements.

The receiver blocks its command queue after a complete header. Writes consume
only the admitted number of words; reads retain the source until asynchronous
transmission completes. The response header arena is reused for four-word
16-bit expansion chunks only after its own send retires. The next command is
admitted after the complete response retires. Root output still requires the
original explicit consume acknowledgement; the adapter does not automatically
release a root or fabricate a neural result.

The local prototype owns input/output queue0, DSR0, send UT0, local tasks23/24,
data task0 and colors12/13. These are distinct from the actual cohost's neural
resources. They are **not** globally admitted routes across the complete stage.
Internal PEs have no SDK command consumer, so their unused C22 receive route
must not backpressure SDK command broadcasts. The tested endpoint discards C22;
a forwarding option exists but is untested. Production gateway placement must
explicitly qualify SDK routing/transit as well as the new control paths. Do not
apply the local discard setting blindly before an SDK endpoint farther east.

## Executed evidence

`device-control-sim-005` uses one SDK gateway and one internal original tail PE,
logically[103,145]. Its complete unchanged8130-word original bank is extracted
from the hash-verified P31 reference fixture, including the original initialized
weights/scales and zero state/value storage. The host accesses only the gateway;
all worker writes, reads and operation dispatches cross the actual fabric.

All122 commands pass: complete bank load and two full bit-exact readbacks,
40 exact16-bit descriptors, three arm/start/finish and mixer-generation/drain
rounds, and a changed tail restored before the final complete bank read. The
selected PE has no A tiles; those three mixer invocations check control/drain
only and perform no neural work. The original complete projection, contraction
paths, root consumers, norm-phase callbacks and complete model are not executed
by this probe. It stops normally after47.701835074s including fixture preparation,
compilation and simulator work. No timing here is used as inference throughput.

Both actual ELF gates pass with4096-byte stack allowance: gateway11024bytes and
internal PE45920bytes. The selected full bank is admitted; no claim extends this
to every original bank or the full11388-PE layer.

## Matched compiled storage

Resource compile017 includes the final C22 and widened-return corrections.
Against014, the eight selected weight-bearing cohosts with identical4096-word
calibration banks shrink640–864bytes. The SDK controller's32-byte difference
reflects the changed export surface and is not counted as internal-PE savings.
Both source adaptation hashes and unchanged parameters/bank sizes are checked.
The detailed measurements are in`device-control-calibration-001.json`.

Original-bank totals remain arithmetic estimates in that comparison. Even with
all nonmatrix pages omitted, selected norm cohosts still have estimated deficits
of2956–4140bytes at the unchanged48128-byte ceiling. SDK removal alone therefore
does not resolve the complete-layer placement problem. Keep every original
matrix, state page and4KiB stack allowance; jointly revise role/weight/scratch
placement and then compile the entire connected graph.

## Preserved failures and resources

- Selected compile015 failed on an unsupported comptime pointer cast. A stager
  variable also incorrectly named its remote directory/unit after a source file;
  its actual identity is preserved and release verified. The name handling is
  fixed and has a regression test.016 compiled the intermediate control port.
- Simulator001 failed before compile because a fixture path was assumed instead
  of resolved through the existing dispatch receipt. The fixture is actually in
  the preserved `remap_banks` run; subsequent attempts use that frozen identity.
- 002 had an SDK zero-length response failure without phase localization.003 adds
  observation and shows24 successful writes followed by a host reply-read failure
  at sequence25.004 changes only C22 handling in CSL and gets through all8130
  original words of write/read, then fails the16-bit return at sequence66.
- 005 keeps C22 isolation and widens each16-bit return into32-bit wire words using
  the retired response header arena. The unchanged driver then passes all122
  commands and normal stop. All failures remain frozen.

All three selected compile services and five simulator services are terminal,
with empty owned cgroups/processes and the shared workstation lock available.
No new WSE job was submitted. ALCF accounting reports no owned active job or
system assignment; provider billed node-hours are not exposed by that query.
The functional and P31 physical baselines remain unchanged. Full-stage routing,
original GDN/attention execution, complete64-layer sentences and sustained>=2000
aggregate completed generated tokens/s remain open.
