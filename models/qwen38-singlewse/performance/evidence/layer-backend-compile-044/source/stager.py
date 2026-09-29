"""Calibrate direct streams on original cohosts and one tight immutable bank."""
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.stage_gdn_bridge_cost import build as previous
from tools.build_frontend_stage import verified_frozen,encoded
from tools.stage_gdn_resource_compile import main


def build():
    files,samples=previous()
    port=(ROOT/'runtime/gdn_stream_bridge.cslpart').read_bytes()
    for role,idle in [('compact_layer_projection.csl',b'@assert(!running and !finish_requested and !native.pending);'),
                      ('device_mixer_layer_mlp_standby.csl',b'@assert(mixer.drained() and mixer_status[2]==0);')]:
        files['stream_bridge_'+role]=files[role]+port.replace(b'BRIDGE_COHOST_IDLE',idle)
    layout=files['layout.csl'].decode().replace('"bridge_', '"stream_bridge_')
    layout=layout.replace('@export_name("stream_bridge_arm",fn()void);','@export_name("bridge_stream_arm",fn(u32)void);')
    layout=layout.replace('@export_name("stream_bridge_inspect",','@export_name("bridge_inspect",').replace('@export_name("stream_bridge_status",','@export_name("bridge_status",')
    for sample in samples:
        if sample['source'].startswith('bridge_'):sample['source']='stream_'+sample['source']
    base=ROOT/'evidence/layer-mlp-compile-031'
    manifest=json.loads((base/'source-manifest.json').read_text())['files']
    profiles=json.loads(verified_frozen(base/'source/profiles.json',manifest['profiles.json']))['profiles']
    tight=next(c for c in profiles if [95,58]in c['pes'])
    assert tight['source']=='compact_layer_projection.csl'and not tight['parameters'].get('root',False)
    for variant in range(2):
        parameters=dict(tight['parameters']);name=tight['source']
        if variant:
            name='stream_bridge_'+name
            parameters.update(bridge_group=5,bridge_frames=28,bridge_workers=7,bridge_input=1,bridge_output=18,
                              bridge_return_input=13,bridge_return_output=9,bridge_forward_queue=7,bridge_reverse_queue=5,bridge_switch=True)
        x=10+variant;fields=','.join('.%s=%s'%(k,v if isinstance(v,str)else str(v).lower())for k,v in parameters.items())
        layout=layout.replace(' @export_name',f' @set_tile_code({x},0,"{name}",.{{.memcpy_params=ordinary_memcpy_params({x}),{fields}}});\n @export_name',1)
        samples.append(dict(pe=[x,0],pair=5,variant='candidate'if variant else'baseline',source=name,parameters=parameters,original_pe=[95,58],payload_changed=False))
    layout=layout.replace('.width=10,','.width=12,').replace('@set_rectangle(10,1)','@set_rectangle(12,1)')
    files['layout.csl']=layout.encode();files['gdn_stream_bridge.cslpart']=port
    files['profiles.json']=encoded(dict(samples=samples,application=[12,1],scope=__doc__,base='layer-mlp-compile-031',
                                       full_stage_admission=False,physical=False,executed=False,payload_changed=False))
    return files,samples


if __name__=='__main__':main(build,Path(__file__))
