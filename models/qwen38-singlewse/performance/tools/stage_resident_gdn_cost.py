"""Paired exact-bank compiler measurements of compact resident endpoint ports.

Only selected cohost profiles execute through the compiler; no numerical run or
whole-stage admission. This avoids repeating a known 245-PE SRAM rejection.
"""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from spatial.resident_gdn import compact_worker, compact_producer
from tools.build_frontend_stage import verified_frozen, encoded
from stage_gdn_resource_compile import main


def build():
    baseline=ROOT/'evidence/layer-mlp-compile-028'
    manifest=json.loads((baseline/'source-manifest.json').read_text())['files']
    files={n:verified_frozen(baseline/'source'/n,d)for n,d in manifest.items()if n not in ('execute.py','stager.py')}
    profiles=json.loads(files['profiles.json'])['profiles'];selected=[]
    for rows in (4,8):
        for children in (0,1,2):
            selected.append(next(p for p in profiles if p['source']=='resident_gdn_compact_layer_projection.csl' and
                                 p['parameters']['rows']==rows and p['parameters']['children']==children))
    selected.append(next(p for p in profiles if p['source']=='resident_gdn_device_mixer_layer_mlp_standby.csl'))
    for group in (0,1,2,15):
        selected.append(next(p for p in profiles if p['parameters'].get('frontend_group')==group))
    width=2*len(selected);samples=[]
    layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});',
            'fn ordinary_memcpy_params(x:i16) memcpy.Params {return memcpy.get_params(x);}',
            'layout {',f' @set_rectangle({width},1);']
    for pair,p in enumerate(selected):
        old=p['source'];changed='compact_'+old
        if changed not in files:
            files[changed]=(compact_producer if 'frontend_group'in p['parameters'] else compact_worker)(files[old])
        for v,name in enumerate((old,changed)):
            x=2*pair+v;params=deepcopy(p['parameters'])
            fields=','.join('.%s=%s'%(k,q if isinstance(q,str)else str(q).lower())for k,q in params.items())
            layout.append(f' @set_tile_code({x},0,"{name}",.{{.memcpy_params=ordinary_memcpy_params({x}),{fields}}});')
            samples.append(dict(pe=[x,0],pair=pair,variant='compact'if v else'baseline',source=name,
                                parameters=params,original_pe=p['pes'][0],payload_changed=False))
    exported={v.split(',')[-1].strip().strip('"')for v in re.findall(
        r'@export_symbol\(([^;]+)\);',files['resident_gdn_compact_layer_projection.csl'].decode())}
    for declaration in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode()):
        name=re.search(r'@export_name\("([^"]+)"',declaration).group(1)
        if name in exported:
            layout.append(' '+declaration)
    files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
    files['profiles.json']=encoded(dict(samples=samples,application=[width,1],scope=__doc__,
        base='layer-mlp-compile-028',payload_changed=False,full_bank_admission=False,
        full_stage_admission=False,physical=False,executed=False))
    files['resident_gdn.py']=(ROOT/'spatial/resident_gdn.py').read_bytes()
    return files,samples


if __name__=='__main__':main(build,Path(__file__))
