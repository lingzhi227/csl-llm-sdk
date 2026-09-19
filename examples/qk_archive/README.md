# Physical QK archive and workspace alias fixture

The fixture directories retain separate compilation and runtime sources.
`layer3-native-fullfit-003` adds the independently accepted selected complete-role
compile source; its demoted graph must never execute. All ten
CSL files, the physical driver, archive checker and durable capture code match
the accepted frozen inputs. Two compiler entry points have the same sole type of portability edit: selecting the compiler's
SDK ELF inspector interpreter through `CSL_INSPECT_PYTHON`. It must point to the
installed interpreter that provides `cerebras.elf`; it does not alter device code.

See the [report](../../docs/QK-ARCHIVE-PHYSICAL.md) for the four device roles, two
reset generations, source overwrite before sink commit, exact comparisons and
five preserved failures. [Source provenance](../../evidence/qk-archive/source-map.json)
records both original and published hashes. Historical source manifests are
evidence; private launch configuration, compiled artifacts, SDK distributions,
cluster logs and model weights are omitted.

The accepted run used SDK 2.10.1, physical WSE-3, one worker CPU and 4 GiB worker
memory. The host was pinned to CPU 0 with a 4 GiB address-space limit, 1 GiB
sampled-RSS ceiling, 128 MiB file ceiling, 8 MiB logs and 32 GiB free-disk reserve.
Compilation had a 300-second limit and 128 MiB candidate ceiling; runtime had
300 seconds and 32 MiB. The cleanup watchdog independently stops and reaps owned
processes and verifies scheduler release. There is no automatic retry.

Reproduction requires a fresh site-local execution directory with the same two
candidate directory names, a working Cerebras client/`csctl`, CPU affinity and a
qualified external guard. Freeze a manifest over the actual local source and
prepare an exact admission matching `source_gate.py`. Compile and inspect first,
then independently review actual SRAM/placement and resource release. Bind the
new artifact and all six actual compile receipts in `compiled-binding.json`,
freeze the runtime sources and separately admit that runtime. The published
historical hashes do not authorize a new compilation or arbitrary artifact.

`supervisor.py` is the guarded entry point in each directory. The compile entry
requires `CSL_INSPECT_PYTHON`; the watchdog provides `HW00_JOB_CAPTURE` to its
child. Do not invoke the physical driver outside the guard or remove the
one-attempt checks to rerun a used directory. This source projection intentionally
contains no ready-to-submit admission or historical binary.

Saved raw evidence can be inspected without the SDK. The runtime folder's
`check_capture.py` and `archive_decode.py` accept the `values` arrays in each
published JSON capture. The accepted result includes thirteen paths and hashes.
No full neural-layer, original-weight or dependent-token acceptance is implied.

Run `python3 examples/qk_archive/verify_saved.py` from the repository root for
the SDK-free saved-capture check. This reads only the published JSON evidence.
