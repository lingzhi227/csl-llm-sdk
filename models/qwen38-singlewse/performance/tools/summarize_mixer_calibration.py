"""Compare identical resource probes; distinguish compiled bytes from estimates."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.layer_schedule import local_rows,xy_rank

def summarize(before,after):
 def read(name):
  base=ROOT/'evidence'/name
  if not json.loads((base/'COMPLETE.json').read_text())['passed']:raise ValueError('Incomplete calibration')
  if not json.loads((base/'workstation-release.json').read_text())['workstation_released']:raise ValueError('Live calibration')
  profiles=json.loads((base/'source/profiles.json').read_text())['samples']
  report=json.loads((base/'sram.json').read_text());assert report['passed']
  by_pe={}
  for r in report['records']:
   for x,y,w,h in r['rectangles']:
    for xx in range(x,x+w):
     for yy in range(y,y+h):by_pe[(xx-4,yy-1)]=r
  assert len(by_pe)==len(profiles)
  return profiles,by_pe
 old,old_elf=read(before);new,new_elf=read(after)
 assert old==new, 'Matched code calibration requires identical bank extents and parameters'
 schedule=json.loads((ROOT/'evidence/layer-native-schedule-002/layer-schedule.json').read_text())
 stage=next(s for s in schedule['stages'] if s['id']=='layer_00');region=next(r for r in stage['regions'] if r['role']=='mix')
 records=[]
 for profile in new:
  pe=tuple(profile['pe']);a=old_elf[pe];b=new_elf[pe]
  assert a['source']==b['source']==profile['source'] and a['stack_allowance_bytes']==b['stack_allowance_bytes']==4096
  actual=b['low_section_end']+4096;prior=a['low_section_end']+4096;payload=profile['parameters'].get('bank_words',0)*4
  matrix=0
  if payload:
   rank=xy_rank(region,profile['original_pe'])
   matrix=sum(local_rows(m,rank)*(256 if m['dtype']=='BF16' else 260) for m in region['matrices'])
  overhead=actual-payload;original=profile['original_bank_words']*4
  records.append(dict(profile,before_bytes_with_stack=prior,after_bytes_with_stack=actual,saved_bytes=prior-actual,
   actual_calibration_bank_bytes=payload,overhead_with_stack=overhead,original_matrix_bytes=matrix,
   original_bank_estimate=overhead+original,matrix_only_estimate=overhead+matrix,
   matrix_only_estimated_margin=48128-overhead-matrix,before_elf_sha256=a['sha256'],after_elf_sha256=b['sha256']))
 return dict(before=before,after=after,physical=False,executed=False,full_bank_admission=False,full_stage_admission=False,samples=records,
  note='Actual bytes are compiled selected programs with their recorded calibration banks and4096-byte stack allowance. Original-bank and matrix-only totals are arithmetic estimates; they do not certify full-bank placement, routing-context overhead or full-stage SRAM.')

def main():
 p=argparse.ArgumentParser();p.add_argument('before');p.add_argument('after');p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 result=summarize(a.before,a.after)
 with a.output.open('x') as f:json.dump(result,f,indent=2);f.write('\n')
 print(json.dumps(dict(samples=len(result['samples']),saved_bytes=[r['saved_bytes'] for r in result['samples']],matrix_only_estimated_margins=[r['matrix_only_estimated_margin'] for r in result['samples']])))

if __name__=='__main__':main()
