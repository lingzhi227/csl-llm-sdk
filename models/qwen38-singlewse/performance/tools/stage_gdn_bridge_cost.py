"""Compile exact-bank bridge cohosts and weighted frontend markers in pairs."""
from copy import deepcopy
import json
from pathlib import Path
import re
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.resident_gdn_bridge import compose,weighted_frontend
from tools.build_frontend_stage import verified_frozen,encoded
from stage_gdn_resource_compile import main


def build():
    base=ROOT/'evidence/layer-mlp-compile-029'
    manifest=json.loads((base/'source-manifest.json').read_text())['files']
    files={n:verified_frozen(base/'source'/n,h)for n,h in manifest.items()if n not in ('execute.py','stager.py')}
    profiles={tuple(pe):c for c in json.loads(files['profiles.json'])['profiles']for pe in c['pes']}
    workers=json.loads(files['gdn-bank-placement.json'])['workers']
    selected=[]
    for pe,group,first,out in [((103,60),15,(0,6),(15,20)),((109,145),14,(14,9),(12,16)),((80,145),3,(5,4),(14,9))]:
        w=[w for w in workers if w['head']//3==group and (w['role']!='mix'if group==15 else not(w['role']=='gate_up'and w['pe'][0]<=86+group))]
        c=profiles[pe];mixer=c['source']=='device_mixer_layer_mlp_standby.csl'
        extra=dict(bridge_group=group,bridge_frames=sum(v['columns']//2 for v in w),bridge_workers=len(w),bridge_input=first[0],bridge_return_output=first[1],bridge_output=out[0],bridge_return_input=out[1],bridge_forward_queue=2 if mixer else 7,bridge_reverse_queue=3 if mixer else 5)
        selected.append((pe,c,extra))
    for g in (14,15):selected.append(((86+g,145),profiles[86+g,145],None))
    width=2*len(selected);samples=[]
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});', 'fn ordinary_memcpy_params(x:i16) memcpy.Params {return memcpy.get_params(x);}','layout {',f' @set_rectangle({width},1);']
    exported=set()
    for pair,(pe,c,extra)in enumerate(selected):
        old=c['source'];new=('weighted_'if extra is None else 'bridge_')+old
        files[new]=weighted_frontend(files[old])if extra is None else compose(files[old],old)
        for v,name in enumerate((old,new)):
            params=deepcopy(c['parameters'])
            if v and extra:params.update(extra)
            fields=','.join('.%s=%s'%(k,q if isinstance(q,str)else str(q).lower())for k,q in params.items())
            x=2*pair+v;layout.append(f' @set_tile_code({x},0,"{name}",.{{.memcpy_params=ordinary_memcpy_params({x}),{fields}}});')
            samples.append(dict(pe=[x,0],pair=pair,variant='candidate'if v else'baseline',source=name,parameters=params,original_pe=list(pe),payload_changed=False))
            exported.update(v.split(',')[-1].strip().strip('"')for v in re.findall(r'@export_symbol\(([^;]+)\);',files[name].decode()))
    for dec in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode()):
        if re.search(r'@export_name\("([^"]+)"',dec).group(1)in exported:layout.append(' '+dec)
    layout+=[' @export_name("bridge_arm",fn()void);',' @export_name("bridge_inspect",fn()void);',' @export_name("bridge_status",[*]u32,false);','}']
    files['layout.csl']=('\n'.join(layout)+'\n').encode()
    files['profiles.json']=encoded(dict(samples=samples,application=[width,1],scope=__doc__,base='layer-mlp-compile-029',full_stage_admission=False,physical=False,executed=False,payload_changed=False))
    files['gdn_bridge.cslpart']=(ROOT/'runtime/gdn_bridge.cslpart').read_bytes();files['resident_gdn_bridge.py']=(ROOT/'spatial/resident_gdn_bridge.py').read_bytes()
    return files,samples

if __name__=='__main__':main(build,Path(__file__))
