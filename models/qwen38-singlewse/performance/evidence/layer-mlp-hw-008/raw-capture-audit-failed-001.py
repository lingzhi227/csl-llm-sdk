directory='/srv/qwen38-singlewse-hardware/layer-mlp-hw-008'

import os,resource,time
os.environ.update(OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1')
resource.setrlimit(resource.RLIMIT_AS,(1<<30,1<<30));resource.setrlimit(resource.RLIMIT_CORE,(0,0))
resource.setrlimit(resource.RLIMIT_CPU,(30,30));resource.setrlimit(resource.RLIMIT_FSIZE,(1<<20,1<<20))
import hashlib,json,zipfile
from pathlib import Path
import numpy as np
root=Path(directory);os.chdir(root);began=time.monotonic()
def digest(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def read(name):return json.loads((root/name).read_text())
done=read('COMPLETE.json');result=read('result.json');meta=read('fixture.json');config=read('experiment.json')
assert done['all_owned_jobs_released'] and result['physical'] and result['normal_stop'] and result['passed']
assert result['norm_bridge'] and not result['full_model'] and result['full_model_tps'] is None
for name in ('compile-audit.json','run-audit.json'):
 a=read(name);assert a['exit_code']==0 and a['error'] is None and not a['cleanup_errors']
 assert a['jobs'] and all(j['released'] and j['phase']=='SUCCEEDED' and not j['cancelled'] for j in a['jobs'])
assert digest('fixture.json')==result['fixture_metadata_sha256']==config['fixture_metadata_sha256']
assert digest('fixture.npz')==meta['hashes']['fixture.npz']
sizes={}
for name in ('actual.npz','fixture.npz'):
 with zipfile.ZipFile(name) as z:
  names=z.namelist();assert len(names)==len(set(names))
  sizes[name]=sum(i.file_size for i in z.infolist());assert sizes[name]<=128<<20
actual=np.load('actual.npz',allow_pickle=False);reference=np.load('fixture.npz',allow_pickle=False)
binding=read('network-binding.json');workers=read('workers.json');origin=config['logical_origin']
width,height=config['application'];expected=np.zeros((height,width,4),np.uint32)
for w in workers:
 x,y=w['pe'];expected[y-origin[1],x-origin[0],1:]=[w['parts'],w['iterations'],1]
x,y=binding['controller'];expected[y-origin[1],x-origin[0],1:]=[266,226,1]
norms=[s for s in binding['senders'] if s.get('norm_bridge')]
assert sorted(s['input_group'] for s in norms)==list(range(40))
for s in norms:
 x,y=s['pe'];expected[y-origin[1],x-origin[0],1:]=[2,0,1]
 x,y=s['norm_pe'];expected[y-origin[1],x-origin[0],1:]=[2,2,1]
checked=0;records=[]
assert len(meta['cases'])==len(result['epochs'])==4
for case,reported in zip(meta['cases'],result['epochs']):
 i=case['index'];epoch=case['epoch'];assert (reported['index'],reported['epoch'])==(i,epoch)
 preceding=np.concatenate([actual[f'preceding_{i}_{g}'] for g in range(40)])
 residual=np.concatenate([actual[f'residual_{i}_{g}'] for g in range(40)])
 for candidate,truth in ((actual[f'output_{i}'],reference[f'down_bf16_{i}']),
                         (actual[f'successor_{i}'],reference[f'successor_{i}']),
                         (preceding,reference[f'input_{i}']),(residual,reference[f'residual_{i}'])):
  assert candidate.shape==truth.shape==(5120,) and candidate.dtype==truth.dtype==np.uint16
  np.testing.assert_array_equal(candidate,truth);checked+=candidate.size
 expected[:,:,0]=epoch;np.testing.assert_array_equal(actual[f'audit_{i}'],expected)
 np.testing.assert_array_equal(actual[f'bridge_{i}'],[40,40])
 for s in binding['senders']:
  sid=s['id'];grants=sum(g['target']==sid for g in binding['grant_schedule'])
  prepares=sum(g['target']==sid for g in binding['prepare_schedule'])
  np.testing.assert_array_equal(actual[f'sender_{i}_{sid}'],[epoch,grants+prepares,grants,1])
 ticks=actual[f'ticks_{i}'];assert ticks.shape==(12,) and np.all(ticks<65536)
 stamps=[sum(int(ticks[o+j])<<(16*j) for j in range(3)) for o in range(0,12,3)]
 cycles=(stamps[-1]-stamps[0])%(1<<48);assert cycles==reported['controller_cycles']
 assert sum(reported['controller_phase_cycles'].values())==cycles
 transport=actual[f'transport_{i}'];np.testing.assert_array_equal(transport[:2],[176,176])
 assert int(transport[2])==reported['controller_transport']['grants_during_packet_lease']
 assert int(transport[3])==reported['controller_transport']['frames_arrived_during_packet_lease']
 if case['replay_of'] is not None:
  for prefix in ('output','successor'):
   np.testing.assert_array_equal(actual[f'{prefix}_{i}'],actual[f'{prefix}_{case["replay_of"]}'])
 records.append(dict(epoch=epoch,checked_bf16_values=20480,controller_counts=cycles,all_pe_audits_match=True))
assert checked==81920
print(json.dumps(dict(passed=True,physical_capture=True,new_hardware_job=False,full_model=False,
 checked_bf16_values=checked,epochs=records,all83_endpoint_counters_exact=True,all11388_pe_audits_exact=True,
 raw_hashes={n:dict(bytes=Path(n).stat().st_size,sha256=digest(n)) for n in
 ['actual.npz','fixture.npz','fixture.json','source-manifest.json','result.json','artifact.json']},
 npz_uncompressed_bytes=sizes,seconds=time.monotonic()-began,
 peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
 note='Rechecks retained BF16 boundaries/counters/timestamps. Native operands and bank retention were checked during the released device run; no new device reads or inference occur here.')))
