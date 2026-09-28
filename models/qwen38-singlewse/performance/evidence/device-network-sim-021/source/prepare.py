"""Extract all eight complete selected banks from the qualified P36 fixture."""
import hashlib,json
from pathlib import Path
import numpy as np
from reference.mixer_operand import prepare

config=json.loads(Path('selection.json').read_text());root=Path(config['source_root'])
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()
for name,digest in config['hashes'].items():
 if sha(root/name)!=digest:raise ValueError('Qualified complete joint fixture changed')
index={tuple(r['pe']):r for r in json.loads((root/'bank-index.json').read_text())['records']}
source=np.load(root/'banks.npy',mmap_mode='r');arrays={}
for i,p in enumerate(config['workers']):
 record=index[tuple(p['original_pe'])]
 if record['words']!=p['parameters']['bank_words']:raise ValueError('Selected original bank was reduced')
 arrays['bank_'+str(i)]=np.array(source[record['offset']:record['offset']+record['words']],np.uint32)
 arrays['setup_'+str(i)]=np.asarray(p['setup'],np.uint16)
cases=[];ids=np.arange(128,dtype=np.uint16)
for mode in ('mixed','zero_after','changed','replay'):
 bits=(((123+ids%5)<<7)|((ids*31+13)%128)|((ids%2)<<15)).astype(np.uint16)
 if mode=='zero_after':bits=np.where(ids%2,32768,0).astype(np.uint16)
 if mode=='changed':bits=(((117+ids%11)<<7)|((ids*17+7)%128)|(((ids+1)%2)<<15)).astype(np.uint16)
 for parts in (40,48):
  case=len(cases);invocation=0xf1230000+case+1;group=parts-1 if case%4 else 0
  arrays['operand_'+str(case)]=bits.view(np.uint32);arrays['packet_'+str(case)]=prepare(bits,invocation,group,parts)
  cases.append(dict(case=case,mode=mode,invocation=invocation,group=group,parts=parts))
with Path('fixture.npz').open('xb') as f:np.savez(f,**arrays)
Path('fixture.json').write_text(json.dumps(dict(fixture_sha256=sha(Path('fixture.npz')),selected_complete_original_banks=len(config['workers']),
 total_selected_bank_words=sum(p['parameters']['bank_words'] for p in config['workers']),source_hashes=config['hashes'],operand_cases=cases,physical=False),indent=2)+'\n')
