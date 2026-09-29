"""Run the same continuing wire checks on an exact direct-stream bridge."""
import json,re
from pathlib import Path
from tools.stage_gdn_bridge_sim import build as buffered
from tools.build_frontend_stage import encoded


def build(group=15):
    files=buffered(resource_compile='layer-backend-compile-045',group=group)
    plan=json.loads(files['probe-plan.json'])
    name='stream_bridge_compact_layer_projection.csl'
    old=b'fn probe_arm() void {bridge_arm();}'
    assert files[name].count(old)==1
    files[name]=files[name].replace(old,b'fn probe_arm(token:u32) void {bridge_stream_arm(token);}')
    for name in ['source.csl','sink.csl']:
        files[name]=files[name].replace(b'fn probe_arm() void',b'fn probe_arm(new_token:u32) void')
    files['layout.csl']=files['layout.csl'].replace(b'@export_name("probe_arm",fn()void)',b'@export_name("probe_arm",fn(u32)void)')
    expression=re.search(rb'\.bridge_segments=(\[\d+\]u16\{[^}]+\})',files['layout.csl']).group(1)
    segments=list(map(int,expression.split(b'{')[1].rstrip(b'}').split(b',')))
    assert len(segments)==plan['workers']and sum(segments)==plan['frames']
    sink=files['sink.csl'].replace(b'param memcpy_params;',b'param memcpy_params;param segments:[total_workers]u16;')
    sink=sink.replace(b'var waiting:bool=false;',b'var next_marker:u16=0;var waiting:bool=false;')
    sink=sink.replace(b'if(@as(u32,frames)*@as(u32,total_workers)/@as(u32,total_frames)>audit[3])',b'if(frames==next_marker)')
    sink=sink.replace(b'audit[3]+=1;advance();',b'audit[3]+=1;if(audit[3]<total_workers){next_marker+=segments[@as(u16,audit[3])];}advance();')
    sink=sink.replace(b'heads=0;frames=0;',b'heads=0;frames=0;next_marker=segments[0];');files['sink.csl']=sink
    lines=files['layout.csl'].splitlines();index=next(i for i,s in enumerate(lines)if b'@set_tile_code(2,0,'in s)
    lines[index]=lines[index].replace(b'});',b',.segments='+expression+b'});');files['layout.csl']=b'\n'.join(lines)+b'\n'
    plan['segments']=segments
    files['run.py']=files['run.py'].replace(b"r.launch('probe_arm',nonblock=False)",b"r.launch('probe_arm',np.uint32(token),nonblock=False)")
    plan.update(direct_fabric_streams=True,explicit_invocation_argument=True,
                scope='Exact resource045 cohost and original bank extent, with three diagnostic RPC wrappers. '
                      'Synthetic bidirectional streams with interspersed original control markers; no neural execution.')
    files['probe-plan.json']=encoded(plan)
    files['stream_stager.py']=Path(__file__).read_bytes()
    return files
