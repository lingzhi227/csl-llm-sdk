"""Audit saved full-layer captures after confirmed physical release, no SDK."""
import json,sys,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'core'))
from full_capture import require,hash_file,load_prepared
from full_numerics import CASES,check_epoch

def reference_inputs(config):
    root=Path(config['reference']);path=root/'reference.json'
    require(hash_file(path)=='62274b756616094eceff07b27b790da9d74edde997b1300bbac854ce4d380972','Accepted full CPU reference identity')
    record=json.loads(path.read_bytes())
    names=['source-freeze.json']+[prefix+'-'+case+'.npz' for prefix in ('source','official') for case in CASES]
    for name in names:
        path=root/name;spec=record['evidence_files'][name]
        require(path.stat().st_size==spec['bytes'] and hash_file(path)==spec['sha256'],'Accepted reference array hash: '+name)
    return root

def main():
    start=time.monotonic();config=json.loads((ROOT/'inputs.json').read_bytes())
    complete=json.loads((ROOT/'COMPLETE.json').read_bytes());capture=json.loads((ROOT/'capture.json').read_bytes())
    require(complete['all_owned_jobs_released'] and all(not s['cleanup_errors'] and not s['uncorrelated_new_jobs'] and all(j['released'] and j['phase']=='SUCCEEDED' for j in s['jobs']) for s in complete['stages']),
            'Owner watchdog must confirm device release before mathematical audit')
    require(capture['status']=='captured' and capture['normal_stop'] and capture['complete_epochs']==[1,2,3,4] and capture['copies']==253 and capture['launches']==9,'Complete four-input physical capture')
    reference=reference_inputs(config)
    receipt,hidden,_=load_prepared(config['prepared'],config['preparation_sha256'])
    for name,spec in capture['files'].items():
        path=ROOT/'evidence'/name
        require(path.stat().st_size==spec['bytes'] and hash_file(path)==spec['sha256'],'Immutable captured payload')
    for batch in receipt['transfers']:
        require(capture['files'][batch['file']]['sha256']==batch['sha256'],'Every final original weight word and padding retained')
    reports=[]
    for epoch,case in enumerate(CASES,1):
        with np.load(ROOT/'evidence'/f'epoch-{epoch}.npz',allow_pickle=False) as archive:
            data={k:archive[k] for k in archive.files}
        with np.load(reference/('source-'+case+'.npz'),allow_pickle=False) as archive:
            source={k:archive[k] for k in archive.files}
        with np.load(reference/('official-'+case+'.npz'),allow_pickle=False) as archive:
            official={k:archive[k] for k in archive.files}
        reports.append(check_epoch(epoch,data,config['prepared'],receipt,hidden[epoch-1],source,official))
        del data,source,official
    result=dict(status='passed',scope='Full original layer3 MLP at four frozen module inputs',epochs=reports,
                original_weight_bytes=534773760,final_weights_and_padding_retained=True,reference_forward_rerun=False,
                device_already_released=True,seconds=time.monotonic()-start,full_model=False)
    with (ROOT/'numerics.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
