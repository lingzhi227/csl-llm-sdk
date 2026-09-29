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
    return replace_one(base,
        b'task gdn_marker() void {@assert(gdn_markers>0);gdn_markers-=1;}',
        b'task gdn_marker(count:u16) void {const delta:u16=if(count==0) 1 else count;@assert(delta<=gdn_markers);gdn_markers-=delta;}')
