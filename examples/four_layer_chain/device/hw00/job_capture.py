"""Pin-installed SDK hook: persist the job ID before waiting for job readiness.

Only the job notification is observed; submission and numerical calls are intact.
This closes the ordinary missing/truncated-log case. An RPC accepted by the server
whose response is lost remains ambiguous and must fail closed in the supervisor.
"""
import functools
import inspect
import os
from pathlib import Path
import re


def install(path):
    from cerebras.sdk.client.sdk_appliance_client import ClusterManagementClient
    original = ClusterManagementClient._record_new_job
    signature = inspect.signature(original)
    if "job_id" not in signature.parameters:
        raise RuntimeError("Installed SDK job capture interface changed")

    @functools.wraps(original)
    def captured(*args, **kwargs):
        fields = signature.bind(*args, **kwargs).arguments
        job_id = fields["job_id"].split("/")[-1]
        if not re.fullmatch(r"wsjob-[a-z0-9]+", job_id):
            raise RuntimeError("Unrecognized SDK job identity")
        with Path(path).open("a") as out:
            out.write(job_id + "\n")
            out.flush()
            os.fsync(out.fileno())
        return original(*args, **kwargs)

    ClusterManagementClient._record_new_job = captured

