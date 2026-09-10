# CSL-LLM SDK

Research toward handwritten CSL inference for open LLM architectures on WSE-3
clusters, starting with an independent Qwen3.8 implementation. Early code provides
a resource-aware foundation. **Full
Qwen3.8 inference is not implemented or qualified.** Current code covers Linux
resource admission, a single-heavy-job lock, a bounded systemd runner, and explicit
bit-preserving host transfer codecs. These are foundations for handwritten CSL
kernels using Cerebras SDK compilation, layout and runtime.

Planned target: text-only Qwen3.8-27B, a one-WSE-3 BF16 streamed path, a separately
qualified W8 resident path, and two/three-WSE-3 pipeline paths. Optional ML graph
integration and direct physical inter-wafer transport are unqualified. This project
is independent of the earlier CSL-LLM implementation.

## Lightweight checks

Python 3.10+; the core and tests use the standard library. Run from this checkout:

```sh
python3 -m unittest discover -s tests -v
mkdir -p /path/to/project-cache
PYTHONPATH=core python3 -m qwen38.resources \
  --cache /path/to/project-cache --planned-write-bytes 134217728
```

Preflight requires Linux `/proc/meminfo`; unknown readings fail closed. It neither
reserves resources nor starts work. The tests run on macOS and Linux. SDK examples
require a separately installed, licensed Cerebras SDK 2.10.1 and Singularity; no
SDK distribution, model weights or private deployment settings are included.

## Resource policy

One heavy job at a time, compiler and simulator serial. Defaults: job MemoryMax
20 GiB, task swap zero, 8 GiB available RAM reserve, 20 GiB active cache and 32 GiB
free disk reserve. These are ceilings, not allocation targets. The runner enforces
cgroup memory/CPU/process/deadline limits and samples disk and MemAvailable each
second. Disk reserves are sampled guards, **not filesystem quotas**; transient
overshoot is possible. Each regular output file is limited to 64 MiB, core dumps
are disabled, and the cache traversal is bounded to 10,000 entries.

Put active small files on SSD and verified cold archives on HDD. Keep checkpoints
and vendor environments outside Git and off the review laptop. Never remove
unrelated data to pass admission. Check mount identity before archival; there is
no automatic SSD fallback or automatic retry. Existing machine swap use is not
attributed to the new job; its cgroup swap limit is zero.

`tools/guarded_run.py` accepts a JSON spec with `planned_write_bytes` and one or two
`steps`, each containing `name`, an explicit `argv` array and `seconds` (1–300).
Use a fresh work directory inside the cache. `tools/sdk_container.py` binds that
work directory before its temporary directory and explicitly sets container
`TMPDIR=/tmp`. The runner is a local research harness, not a security boundary or
a complete persistent inference lifecycle manager. Phase-specific device reset,
cancellation and recovery remain future work. Failure records stay immutable.

## Transfer contracts

- `memcpy16_containers`: one valid low u16 per u32 host container. D2H upper bits
  are ignored explicitly. Logical bytes and host container bytes are different.
- `packed_u32_stream`: low halfword first, two u16 per u32, zero odd-tail padding.
- `raw_u32`: one unsigned u32 bit value per word, little-endian serialization.

Inputs are contiguous one-dimensional unsigned bit arrays; floating-point casts,
empty inputs, incorrect units and mismatched formats are rejected. CPU round trips
are not evidence of SDK transfer or hardware execution.

See [status](docs/STATUS.md) for qualified scope and outstanding work.

## Reproduce the tiny native memcpy example

This example copies seven u16 BF16 bit patterns on one PE. It performs no neural
arithmetic. Preparation does not launch a compiler, simulator or download.

```sh
python3 tools/prepare_wp00.py --run /path/to/project-cache/my-new-run \
  --image /path/to/already-installed-sdk-2.10.1.sif
# Review source-manifest.json and steps.json in that new bundle first.
python3 /path/to/project-cache/my-new-run/tools/guarded_run.py \
  --cache /path/to/project-cache --work /path/to/project-cache/my-new-run \
  --spec /path/to/project-cache/my-new-run/steps.json
```

Use the reviewed SDK image identified in [WP00 results](docs/WP00-REPORT.md).
The runner needs a working systemd user manager with cgroup v2 memory delegation.
If a previous unit remains, inspect its ownership and processes before clearing
that specific failed unit. Do not bypass the lock or overwrite an attempted run.
