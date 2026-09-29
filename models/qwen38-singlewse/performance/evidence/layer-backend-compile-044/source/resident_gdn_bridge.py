"""Explicit color translation with independent wire/source/control lifetimes.

Compiler admission is required on every actual cohost. The bridge is armed by
an admitted GDN phase; the serving controller must also fence subsequent native
projection entry on bridge_idle(), not only on the remote terminal marker.
"""
from pathlib import Path
from spatial.gdn_fusion import replace_one
ROOT=Path(__file__).resolve().parents[1]

def compose(base,role):
    if role=='compact_layer_projection.csl':
        idle=b'@assert(!running and !finish_requested and !native.pending);'
    elif role=='device_mixer_layer_mlp_standby.csl':
        idle=b'@assert(mixer.drained() and mixer_status[2]==0);'
    else:raise ValueError('Unadmitted bridge cohost family')
    return base+(ROOT/'runtime/gdn_bridge.cslpart').read_bytes().replace(b'BRIDGE_COHOST_IDLE',idle)

def weighted_frontend(base):
    # Keep P47's no-argument marker ABI unchanged. The weighted fence has its
    # own entrypoint; an old general control payload is never read as an arg.
    if b'task gdn_marker() void {@assert(gdn_markers>0);gdn_markers-=1;}'not in base:
        raise ValueError('Original marker ABI differs')
    return base+b'''\nconst gdn_weighted_marker_task=@get_control_task_id(41);
task gdn_weighted_marker(count:u16) void {
 @assert(count>0 and count<=gdn_markers);gdn_markers-=count;
}
comptime {@bind_control_task(gdn_weighted_marker,gdn_weighted_marker_task);}
'''
