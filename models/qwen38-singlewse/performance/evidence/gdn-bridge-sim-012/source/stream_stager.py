"""Run the same continuing wire checks on an exact direct-stream bridge."""
import json
from pathlib import Path
from tools.stage_gdn_bridge_sim import build as buffered
from tools.build_frontend_stage import encoded


def build(group=15):
    files=buffered(resource_compile='layer-backend-compile-044',group=group)
    plan=json.loads(files['probe-plan.json'])
    name='stream_bridge_compact_layer_projection.csl'
    old=b'fn probe_arm() void {bridge_arm();}'
    assert files[name].count(old)==1
    files[name]=files[name].replace(old,b'fn probe_arm(token:u32) void {bridge_stream_arm(token);}')
    for name in ['source.csl','sink.csl']:
        files[name]=files[name].replace(b'fn probe_arm() void',b'fn probe_arm(new_token:u32) void')
    files['layout.csl']=files['layout.csl'].replace(b'@export_name("probe_arm",fn()void)',b'@export_name("probe_arm",fn(u32)void)')
    files['run.py']=files['run.py'].replace(b"r.launch('probe_arm',nonblock=False)",b"r.launch('probe_arm',np.uint32(token),nonblock=False)")
    plan.update(direct_fabric_streams=True,explicit_invocation_argument=True,
                scope='Exact resource044 cohost and original bank extent, with three diagnostic RPC wrappers. '
                      'Synthetic bidirectional streams with interspersed original control markers; no neural execution.')
    files['probe-plan.json']=encoded(plan)
    files['stream_stager.py']=Path(__file__).read_bytes()
    return files
