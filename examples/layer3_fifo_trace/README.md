# Full layer 3 finite FIFO diagnostics

This publishes the device source and capture logic of the independently accepted
September 19, 2026 UTC diagnostic. It is an original-weight physical observation,
not a complete layer or model numerical result. Twenty of 24 attention heads
reported READY; heads 17, 21, 22 and 23 still lacked K/V inputs. Complete neural epochs: 0.

- `device/`: all 23 CSL files are byte-identical to full compile 002. The complete
 750 × 45 graph compiled into 563 programs; actual ordinary storage plus the 4096-byte
 stack allowance was at most 47920 bytes under 48128. The compiler, actual ELF
 inspector and release supervisor are included.
- `capture/`: the frozen 394-copy/three-launch host plan and both decoders. Its
 sole post-compute transfer reads 50 idle locations, 160 u32 each, at(748, 0, 2, 25).
 After a 20-second delay, the read has a 35-second absolute window and is followed
 by unconditional stop. No busy-source or post-compute all-PE read is performed.
- [Raw idle words](../../evidence/layer3-fifo-idle-words.json) permit host-only
 replay through `capture/fifo_observer_decode.py`. They contain diagnostic words,
 not model parameter arrays. All original NPZ files and compiled binaries remain
 outside this repository.

Reproduction requires SDK 2.10.1, an authenticated appliance Python environment,
the pinned original model and prepared arrays, and a fresh execution directory.
`inputs.template.json` retains the observed identities but leaves deployment paths
null. Fill these paths and copy it to `inputs.json`; fresh artifacts require their
own accepted source/artifact/ELF/SRAM/placement receipts. The included `prepared.json`
is a historical identity ledger, not a parameter payload or a preparation script.

For compilation, `CSL_INSPECT_PYTHON` chooses the SDK interpreter used by the ELF
inspector; default `cs_python`. This is the sole portability edit to copied Python
source and has not submitted another run. Freeze a new source manifest and review
its exact admission before using either supervisor. Run `supervisor.py` through
the site's bounded launcher; do not bypass its owned-job release watchdog.

The historical compile profile is 600s, 1 CPU/4 GiB worker, 4 GiB host address space,
1 GiB sampled RSS, 128 MiB file, 8 MiB log, 512 MiB candidate and 32 GiB disk reserve.
Physical capture uses 600s, 2 CPU/4 GiB worker, 30s journal-progress protection,
64 MiB candidate, and the same host/file/log/disk bounds. The 50-slot snapshot is
32000 bytes; all host transfers total 1508898048 bytes. Keep all large arrays,
artifacts and logs on the execution disk. Preserve failed attempts and freshly
verify terminal job, empty system assignments and all owned process exits.

The source FIFO contract preserves its pre-compilation historical status fields;
[accepted evidence](../../evidence/layer3-fifo-trace.json) records the later result.
No copy of a historical admission is valid for a new execution.

The 1024-f32 QK RMS intermediate record aliases later attention workspace. Its old
retained-intermediate numerical audit is invalid. Zero runtime error codes,
finite summaries, READY signals and successful capture do not substitute for
reference numerical acceptance. A new observation of complete progress still
needs a separately declared numerical capture and applicable mathematical gates.

[Detailed report](../../docs/LAYER3-FIFO-TRACE.md) ·
[Exact source mapping](../../evidence/fifo-trace-source-map.json)
