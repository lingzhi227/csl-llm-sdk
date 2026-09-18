# Original-weight fragment on physical WSE-3

HW00 executes the accepted layer-3 resident 192→128→128 MLP on eight physical
WSE-3 PEs using the appliance SDK 2.10.0. Three dense/changed/zero inputs pass
the same 125-operation numerical, handoff, epoch and complete weight-retention
checks as the earlier simulator fragment. Full MLP and full-model generation
remain unqualified. See the [report](../../docs/HW00-PHYSICAL.md) and
[independent acceptance summary](../../evidence/hw00-physical.json).

The eight CSL files and existing numerical driver are unchanged. `run_hw.py`
maps appliance context entry/exit to the accepted driver's load/run/stop contract.
`compile_hw.py` checks a newly generated physical artifact's embedded sources,
all eight SRAM footprints and actual native ELF placement before allocation.
Set `CSL_SDK_PYTHON` to your installed SDK 2.10.1 `cs_python` wrapper for this
metadata inspection. The wrapper performs no simulation.

`hw00/watchdog.py` imposes 600-second compilation and 300-second physical client
deadlines, then independently checks every captured owned job is terminal and
absent from system assignments. An installed-SDK notification hook persists the
job ID before readiness waiting. Missing identity or release proof prevents a
subsequent allocation; it never cancels another invocation based on account
changes alone. The host watchdog must remain on the remote appliance host.

Obtain the original model under its license and prepare the exact private inputs
using the existing [resident source recipe](../../core/qwen38/resident_source.py).
Input/oracle identities are checked by `accepted_inputs.py`; arbitrary replacement
arrays do not reproduce this accepted test. Model parameters and binaries are
deliberately not included. On an authenticated appliance host, prepare a new run:

```sh
python3 examples/hw00/prepare.py --prepared /path/to/private/prepared --work /path/to/new-run
```

Review the generated admission template and your deployment, then rename it to
`admission.json` to authorize the explicit physical reproduction. In the run
directory, using the appliance Python environment and the SDK wrapper configured:

```sh
export CSL_SDK_PYTHON=/path/to/sdk/cs_python
nohup python -m hw00.watchdog > supervisor.log 2>&1 < /dev/null &
```

This operation compiles and reserves one physical CS-3; it is not an offline test.
Use a fresh directory. Preserve failed attempts and investigate cleanup before
another allocation. `COMPLETE.json` is written only after numerical, source and
resource-release gates pass. Official charges require the site's accounting ledger.

The public adapter's SDK path is environment-configurable; this portability edit
has not submitted another hardware run. To exercise only the focused cleanup and
checkpoint fixtures, run `python3 examples/hw00/test_hw00.py`; no SDK is imported
and no device is allocated by those fixtures.
