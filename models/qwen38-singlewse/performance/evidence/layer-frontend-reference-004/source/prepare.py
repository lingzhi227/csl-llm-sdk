"""Remote full-word proof for original MLP rebalancing and recurrent retile.

Source and target banks remain remote. Actual emitted native descriptors provide
an independent fresh readback addressing path after the migration is flushed.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from spatial.compact_mlp import CompactMlpPlacement
from spatial.gdn_placement import audit,STATE_FIRST,STATE_PAGES
from spatial.layer_schedule import auxiliary_owner,rank_xy


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1<<20),b''):h.update(block)
 return h.hexdigest()


def main():
 started=time.monotonic();read=lambda n:json.loads(Path(n).read_text());config=read('source-bank.json');root=Path(config['root'])
 for name,digest in config['hashes'].items():
  if sha(root/name)!=digest:raise ValueError('Frozen original bank identity changed')
 original=read('frontend-stage.json');before=read('mlp-bank-placement.json');plan=read('gdn-bank-placement.json');audit(original,before,plan)
 left=CompactMlpPlacement(original);right=CompactMlpPlacement(plan['stage'])
 if right.metadata()!=read('gdn-compact-mlp.json'):raise ValueError('Emitted compact MLP metadata changed')
 oldindex=json.loads((root/'bank-index.json').read_text());old={tuple(r['pe']):r for r in oldindex['records']}
 source=np.load(root/'banks.npy',mmap_mode='r');source_sizes={tuple(p['pe']):p['bytes'] for p in before['bank_ends'] if p['bytes']}
 if source.dtype!=np.dtype('<u4') or source.ndim!=1 or source.nbytes!=before['allocated_bytes'] or len(old)!=len(oldindex['records']) or set(old)!=set(source_sizes) or any(r['words']*4!=source_sizes[pe] for pe,r in old.items()):raise ValueError('Original complete bank index')
 cursor=0
 for r in sorted(old.values(),key=lambda r:r['offset']):
  if r['offset']!=cursor:raise ValueError('Original bank overlap/gap')
  cursor+=r['words']
 if cursor!=source.size:raise ValueError('Original source coverage')
 profiles={tuple(pe):c['parameters'] for c in read('profiles.json')['profiles'] for pe in c['pes']}
 # Capabilities, not a historical filename prefix, identify actual ports.
 # Resident composition retains these compiler-bound parameters under its new
 # source name; every field is checked against the original-coordinate map.
 actual_state={tuple(pe):c['parameters'] for c in read('profiles.json')['profiles'] if 'gdn_columns' in c['parameters'] for pe in c['pes']}
 if set(actual_state)!={tuple(w['pe']) for w in plan['workers']}:raise ValueError('Actual recurrent port coverage')
 for w in plan['workers']:
  p=actual_state[tuple(w['pe'])]
  if any(p[name]!=w[field] for name,field in [('gdn_columns','columns'),('gdn_head','head'),('gdn_first','first'),('gdn_state_word','state_word')]):raise ValueError('Actual recurrent state descriptor differs')
 payload={pe:p['bank_words']*4 for pe,p in profiles.items() if p.get('bank_words',0)}
 if payload!={tuple(p['pe']):p['bytes'] for p in plan['bank_ends'] if p['bytes']}:raise ValueError('Compiled bank extent differs')
 new={};cursor=0
 for pe,n in sorted(payload.items(),key=lambda x:(x[0][1],x[0][0])):
  new[pe]=dict(pe=list(pe),words=n//4,offset=cursor);cursor+=n//4
 if cursor*4!=plan['allocated_bytes'] or cursor*4>512<<20:raise ValueError('Candidate bound')
 output=np.lib.format.open_memmap('banks.npy',mode='w+',dtype='<u4',shape=(cursor,))
 source_marks=np.zeros(source.size,np.uint8);target_marks=np.zeros(cursor,np.bool_)
 def pos(index,pe,word,length):
  r=index[tuple(pe)]
  if word<0 or length<1 or word+length>r['words']:raise ValueError('Address outside physical bank')
  return r['offset']+word
 def transfer(i,j,n):
  if np.any(source_marks[i:i+n]) or np.any(target_marks[j:j+n]):raise ValueError('Auxiliary/weight overlap')
  output[j:j+n]=source[i:i+n];source_marks[i:i+n]=1;target_marks[j:j+n]=True
 def table(compact,index,role,branch,setups=None):
  region=compact.regions[role];m=region['matrices'][branch];count=m['tiles']
  data=np.full(count,-1,np.int64);scales=np.full(count,-1,np.int64)
  for group in m['group_partitions']:
   for local in range(group['workers']):
    pe=tuple(rank_xy(region,group['rank_start']+local));r=compact.records[pe]
    # Migration uses compact-record offsets; readback uses only actually loaded
    # five-word descriptors and original cohort rank/K identity.
    if setups is None:a=r['setup'];rows=r['rows'];columns=r['columns'];branches=r['branches'];maximum=r['parts']
    else:
     a=setups[pe];p=profiles[pe];rows=p['rows'];columns=p['columns'];branches=p['branches'];maximum=p['max_parts']
    parts,iterations,base0,base1,first=a;keys=np.arange(local,m['k_blocks'],group['workers'])
    if parts!=keys.size or iterations!=group['output_count'] or first!=group['output_start']*rows or branches!=len(region['matrices']) or parts>maximum or m['shape'][1]//columns!=m['k_blocks']:raise ValueError('Actual original-K/output descriptor changed')
    ids=((first//rows+np.arange(iterations))[:,None]*m['k_blocks']+keys[None,:]).reshape(-1)
    base=a[2+branch];dp=base+64*np.arange(iterations*parts)
    sp=base+iterations*parts*64+(((first%128+np.arange(iterations)*rows)>>7)*parts)[:,None]+np.arange(parts)[None,:]
    if np.any(data[ids]>=0):raise ValueError('Repeated original MLP tile')
    pos(index,pe,int(dp.min()),int(dp.max()+64-dp.min()));pos(index,pe,int(sp.min()),int(sp.max()+1-sp.min()))
    data[ids]=index[pe]['offset']+dp;scales[ids]=index[pe]['offset']+sp.reshape(-1)
  if np.any(data<0) or np.any(scales<0):raise ValueError('Missing original MLP tile')
  return data,scales
 tiles=0
 for role in ('gate_up','down'):
  for branch in range(len(left.regions[role]['matrices'])):
   src,ss=table(left,old,role,branch);dst,ds=table(right,new,role,branch)
   for first in range(0,src.size,4096):
    end=min(first+4096,src.size);i=src[first:end,None]+np.arange(64);j=dst[first:end,None]+np.arange(64)
    if np.any(source_marks[i]) or np.any(target_marks[j]):raise ValueError('Repeated original weight codes')
    output[j]=source[i];source_marks[i]=1;target_marks[j]=True
    si=ss[first:end];sj=ds[first:end];values=source[si];unique,indices,inverse=np.unique(sj,return_index=True,return_inverse=True);selected=values[indices]
    if not np.array_equal(values,selected[inverse]):raise ValueError('Unequal original scale aliases')
    live=target_marks[unique]
    if not np.array_equal(output[unique[live]],selected[live]):raise ValueError('Original scale alias across chunks differs')
    if np.any(source_marks[si]==1):raise ValueError('Source scale aliases weight codes')
    output[unique]=selected;target_marks[unique]=True;source_marks[si]=2
   tiles+=src.size
 oldpages={('mix',s['page_start']+i):(tuple(s['pe']),s['byte_offset']//4+32*i) for s in before['spans'] for i in range(s['pages'])}
 for region in original['regions']:
  if region['role'] not in ('gate_up','down'):continue
  for page in range(region['auxiliary_pages']):
   a=auxiliary_owner(region,page);pe=tuple(a['pe']);r=left.records[pe]
   oldpages[region['role'],page]=(pe,(a['byte_offset']-r['original_bytes']+r['bytes'])//4)
 page_pairs=[]
 for s in plan['spans']:
  for n in range(s['pages']):
   key=s['namespace'],s['page_start']+n;pe,word=oldpages[key];i=pos(old,pe,word,32);j=pos(new,s['pe'],s['byte_offset']//4+32*n,32)
   transfer(i,j,32);page_pairs.append((i,j,32))
 fixed_pairs=[]
 for r in before['bank_prefixes']:
  pe=tuple(r['pe'])
  if pe in left.records or not r['bytes']:continue
  n=r['bytes']//4;i=pos(old,pe,0,n);j=pos(new,pe,0,n);transfer(i,j,n);fixed_pairs.append((i,j,n))
 state_pairs=[];keys=np.arange(128)[:,None]
 for w in plan['workers']:
  values=w['first']+np.arange(w['columns'])[None,:]
  pages=STATE_FIRST+w['head']*512+(keys//4)*16+values//8
  # Translate every old4x8 page identity, never treating the fragmented old
  # storage as a contiguous tensor. Within-page row/column order is explicit.
  bases=np.array([pos(old,*oldpages['mix',int(p)],32) for p in pages.reshape(-1)],np.int64).reshape(pages.shape)
  si=(bases+(keys%4)*8+values%8).reshape(-1);j=pos(new,w['pe'],w['state_word'],128*w['columns'])
  if np.any(source_marks[si]) or len(np.unique(si))!=si.size or np.any(target_marks[j:j+si.size]):raise ValueError('Recurrent state alias')
  output[j:j+si.size]=source[si];source_marks[si]=1;target_marks[j:j+si.size]=True;state_pairs.append((si,j))
 if not np.all(source_marks) or not np.all(target_marks):raise ValueError('Missing original source or destination word')
 source_scale_words=int(np.count_nonzero(source_marks==2));verified=np.zeros(source.size,np.bool_)
 output.flush();del output,target_marks
 actual=np.load('banks.npy',mmap_mode='r');before_setup={tuple(pe):a for pe,a in read('previous-worker-setups.json')};after_setup={tuple(pe):a for pe,a in read('worker-setups.json')}
 if set(before_setup)!=set(left.records) or set(after_setup)!=set(right.records):raise ValueError('Loaded native setup coverage')
 for role in ('gate_up','down'):
  for branch in range(len(left.regions[role]['matrices'])):
   src,ss=table(left,old,role,branch,before_setup);dst,ds=table(right,new,role,branch,after_setup)
   for first in range(0,src.size,4096):
    end=min(first+4096,src.size);i=src[first:end,None]+np.arange(64);j=dst[first:end,None]+np.arange(64)
    if not np.array_equal(source[i],actual[j]) or not np.array_equal(source[ss[first:end]],actual[ds[first:end]]):raise ValueError('Actual native descriptor readback changes original tensor')
    verified[i]=True;verified[ss[first:end]]=True
 for i,j,n in fixed_pairs+page_pairs:
  if not np.array_equal(source[i:i+n],actual[j:j+n]):raise ValueError('Retained original data readback differs')
  verified[i:i+n]=True
 for si,j in state_pairs:
  if not np.array_equal(source[si],actual[j:j+si.size]):raise ValueError('Recurrent state readback differs')
  verified[si]=True
 if not np.all(verified):raise ValueError('Fresh original source-word verification incomplete')
 delta=plan['allocated_bytes']-before['allocated_bytes']
 index=dict(records=list(new.values()),total_words=cursor,allocated_bytes=cursor*4,
  initialized_weight_and_scale_bytes=oldindex['initialized_weight_and_scale_bytes']+delta,
  remaining_bytes_are_zero_state_or_values=oldindex['remaining_bytes_are_zero_state_or_values'],conversations=1,work_slots=2,
  bank_specialization='gdn-bank-placement.json',scale_encoding='Original checkpoint bits; exact shared scale tables')
 Path('bank-index.json').write_text(json.dumps(index,separators=(',',':'))+'\n')
 result=dict(passed=True,physical=False,neural_execution=False,multi_turn_qualified=False,
  source_bytes=source.nbytes,allocated_bytes=cursor*4,verified_source_words=int(verified.size),original_mlp_tiles=tiles,
  preserved_auxiliary_pages=len(page_pairs),recurrent_elements=sum(a.size for a,j in state_pairs),workers=len(state_pairs),
  source_scale_words=source_scale_words,scale_table_delta_bytes=delta,all_source_words_classified=True,
  all_original_values_verified_from_actual_csl_descriptors=True,no_requantization=True,
  hashes={n:sha(Path(n)) for n in ('banks.npy','bank-index.json')},seconds=time.monotonic()-started)
 Path('bank-remap-proof.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
