"""Paired P44 cohost costs for complete recurrent receive/math/scalar-return ports.

Calibration start resets/adopts dummy bank state. These banks replace payloads;
no execution or full-bank placement is claimed. Production admission is separate.
"""
import json,re,sys
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2];sys.path[:0]=[str(ROOT/'performance')]
from spatial.gdn_columns import compose_port
from tools.build_frontend_stage import verified_frozen
from stage_gdn_resource_compile import main

def build():
 source=ROOT/'performance/evidence/layer-mlp-compile-024/source'
 manifest=json.loads((source.parent/'source-manifest.json').read_text())['files']
 names=[p.name for p in source.glob('*.csl')]+['source_gate.py','check_sram.py','elf_inventory.py','placement.py']
 files={n:verified_frozen(source/n,manifest[n]) for n in names}
 profiles=json.loads(verified_frozen(source/'profiles.json',manifest['profiles.json']))['profiles'];selected=[]
 for rows in (4,8):
  p=next(p for p in profiles if p['source']=='compact_layer_projection.csl' and p['parameters']['rows']==rows and p['parameters']['max_parts']==1 and not p['parameters']['root'] and not p['parameters'].get('fusion_actor',False) and not p['parameters'].get('mlp_sender',False))
  for width in (1,4,p['parameters']['columns']//2):selected.append((p,width))
 for root in (False,True):
  p=next(p for p in profiles if p['source']=='device_mixer_layer_mlp_standby.csl' and p['parameters']['mixer_can_root']==root)
  for width in (1,4,64):selected.append((p,width))
 files['gdn_columns.csl']=(ROOT/'performance/csl/gdn_columns.csl').read_bytes()
 files['gdn_columns.py']=(ROOT/'performance/spatial/gdn_columns.py').read_bytes()
 samples=[];proofs=[];application=len(selected)*2
 layout=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={application},.height=1}});','layout {',f' @set_rectangle({application},1);']
 for i,(p,width) in enumerate(selected):
  name=p['source'];changed='gdn_calibration_'+name
  raw,proof=compose_port(files[name],name);anchor=b'fn start() void {sys.unblock_cmd_stream();}'
  if raw.count(anchor)!=1:raise ValueError('Calibration entrypoint')
  files[changed]=raw.replace(anchor,b'fn start() void {gdn_reset();gdn_admit();sys.unblock_cmd_stream();}')
  proofs.append(proof)
  for v,program in enumerate([name,changed]):
   x=2*i+v;params=deepcopy(p['parameters']);original=params['bank_words'];params['bank_words']=max(4096,128*width)
   if v:params.update(gdn_columns=width,gdn_first=128-width,gdn_head=47,gdn_state_word=params['bank_words']-128*width,gdn_input_color=0,gdn_output_color=1,gdn_input_queue=7 if name=='compact_layer_projection.csl' else 2)
   fields=','.join('.%s=%s'%(k,q if isinstance(q,str) else str(q).lower()) for k,q in params.items())
   layout.append(f' @set_tile_code({x},0,"{program}",.{{.memcpy_params=memcpy.get_params({x}),{fields}}});')
   samples.append(dict(pe=[x,0],pair=i,variant='port' if v else 'baseline',source=program,parameters=params,original_pe=p['pes'][0],original_bank_words=original,payload_changed=True))
 exported={v.split(',')[-1].strip().strip('"') for v in re.findall(r'@export_symbol\(([^;]+)\);',files['compact_layer_projection.csl'].decode())}
 for d in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode()):
  if re.search(r'@export_name\("([^"]+)"',d).group(1) in exported:layout.append(' '+d)
 files['layout.csl']=('\n'.join(layout+['}'])+'\n').encode()
 files['profiles.json']=(json.dumps(dict(samples=samples,application=[application,1],base='layer-mlp-compile-024',scope=__doc__,physical=False,executed=False,full_bank_admission=False,compositions=proofs),indent=2)+'\n').encode()
 return files,samples

if __name__=='__main__':main(build,Path(__file__))
