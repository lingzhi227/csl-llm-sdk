"""Paired compiler cost of exact MLP scale sharing, including ragged K parts."""
import hashlib,json,re,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'performance')]
from spatial.compact_mlp import compact_mlp_csl
from tools.build_frontend_stage import verified_frozen
from stage_gdn_resource_compile import main

def build():
 source=ROOT/'performance/evidence/layer-mlp-compile-022/source'
 manifest=json.loads((source.parent/'source-manifest.json').read_text())['files']
 names=[p.name for p in source.glob('*.csl')]+['source_gate.py','check_sram.py','elf_inventory.py','placement.py']
 files={n:verified_frozen(source/n,manifest[n]) for n in names}
 profiles=json.loads(verified_frozen(source/'profiles.json',manifest['profiles.json']))['profiles'];selected={}
 for p in profiles:
  if p['source']!='layer_projection.csl':continue
  q=p['parameters'];key=tuple(q.get(k,False) for k in ['rows','max_parts','root','fusion_actor','mlp_sender','input_quantizer','output_sink'])
  selected.setdefault(key,p)
 compact_mlp_csl(files);width=2*len(selected);samples=[]
 layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height=1}});','layout {',f' @set_rectangle({width},1);']
 for index,p in enumerate(selected.values()):
  for variant,name in enumerate(['layer_projection.csl','compact_layer_projection.csl']):
   x=2*index+variant;params=deepcopy(p['parameters']);original=params['bank_words'];params['bank_words']=4096
   fields=','.join('.%s=%s'%(k,v if isinstance(v,str) else str(v).lower()) for k,v in params.items())
   layout.append(f' @set_tile_code({x},0,"{name}",.{{.memcpy_params=memcpy.get_params({x}),{fields}}});')
   samples.append(dict(pe=[x,0],pair=index,variant='compact' if variant else 'baseline',source=name,parameters=params,
    original_pe=p['pes'][0],original_bank_words=original,payload_changed=True))
 exported={v.split(',')[-1].strip().strip('"') for v in re.findall(r'@export_symbol\(([^;]+)\);',files['layer_projection.csl'].decode())}
 for declaration in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode()):
  if re.search(r'@export_name\("([^"]+)"',declaration).group(1) in exported:layout.append(' '+declaration)
 files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
 files['compact_mlp.py']=(ROOT/'performance/spatial/compact_mlp.py').read_bytes()
 files['profiles.json']=(json.dumps(dict(samples=samples,application=[width,1],base='layer-mlp-compile-022',
  scope=__doc__,physical=False,executed=False,full_bank_admission=False),indent=2)+'\n').encode()
 return files,samples

if __name__=='__main__':main(build,Path(__file__))
