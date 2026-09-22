"""One finite SDK-free exercise of changed full136 capture/store/validator paths."""
from pathlib import Path
import hashlib,json,sys,shutil
import numpy as np
from capture import capture
from original_650_fixtures import valid
from validate_capture import validate

class DataType:MEMCPY_32BIT=32
class Order:ROW_MAJOR=1
class Fake:
    def __init__(self,mode):
        self.mode=mode;self.closed=False;self.stop_attempts=0;self.launches=[];self.fault_reads=0;self.payload=valid()
        if mode=='incomplete':self.payload[0][40][1]=self.payload[0][40][2]=2
    def get_id(self,name):return name
    def launch(self,name,**kwargs):self.launches.append(name)
    def memcpy_d2h(self,out,name,x,y,w,h,count,**kwargs):
        if name=='state':
            data=[[1,4 if yy else 1 if xx==0 else 2 if xx==2 else 3 if 40<=xx<176 else 0,0,xx] for yy in range(2) for xx in range(178)]
        elif name=='diagnostic_status':data=self.payload[0]
        elif name=='fault_snapshot':
            self.fault_reads+=1
            if self.mode=='last_copy_error' and self.fault_reads==2:raise RuntimeError('Injected second fault-bank copy error')
            data=[[0]*128 for _ in range(178)]
            if self.mode in ('packet_fault','invalid_source_length','uncommitted','unstable'):
                r=data[0];r[0]=r[31]=2;r[1]=0x46313336;r[2]=1;r[3]=1
                r[4]=268;r[5]=12;r[6]=4;r[7]=3;r[8]=1;r[9]=0x400128;r[10]=8;r[11]=3
                r[12]=5;r[13]=8;r[15]=3;r[32:40]=[10,1,1,0,0,3,18,0]
                r[64:73]=[0x600128,11,1,1,0,0,2,1,128];r[96:104]=[11,1,1,0,0,2,1,128]
                if self.mode=='invalid_source_length':r[13]=31;r[15]=10;r[96:127]=[0]*31
                if self.mode=='uncommitted':r[0]=r[31]=1
                if self.mode=='unstable' and self.fault_reads==2:r[32]^=1
        elif x==0:data=[self.payload[1]]
        elif x==2:data=[self.payload[2]]
        else:data=self.payload[3]
        array=np.asarray(data,dtype=np.uint32).reshape(-1);assert array.size==out.size;out[:]=array
    def stop(self):
        self.stop_attempts+=1
        if self.mode=='stop_error':raise RuntimeError('Injected stop error')
        self.closed=True

def main():
    base=Path(__file__).parent/'host-evidence';assert not base.exists();base.mkdir();results=[]
    for mode in ('complete','incomplete','packet_fault','invalid_source_length','uncommitted','unstable','last_copy_error','stop_error'):
        root=base/mode;root.mkdir();fake=Fake(mode);error=None
        try:capture(root,fake,DataType,Order)
        except (ValueError,RuntimeError) as exc:error=str(exc)
        result=json.loads((root/'result.json').read_bytes());prefix=validate(root,False)
        assert fake.stop_attempts==1 and fake.launches==['initialize','compute']
        if mode=='complete':
            assert error is None and result['copies']==8 and result['host_slot_bytes']==243904
            assert validate(root)['strict650_and_zero_faults']
        else:assert error and result['status']=='failed'
        if mode=='incomplete':assert result['copies']==15 and result['host_slot_bytes']==283776
        if mode=='packet_fault':assert result['fault_report']['latched_faults']==1 and not result['fault_report']['incomplete_columns']
        if mode in ('invalid_source_length','uncommitted'):assert result['fault_report']['incomplete_columns']==[0]
        if mode=='unstable':assert result['fault_report']['changed_columns']==[0]
        if mode=='last_copy_error':
            assert result['copies']==7 and result['normal_stop'] and (root/'evidence/first-fault.npz').exists()
            assert not (root/'evidence/first-fault-final.npz').exists()
        if mode=='stop_error':assert not result['normal_stop']
        results.append(dict(mode=mode,copies=prefix['copies'],host_bytes=prefix['host_bytes'],status=result['status'],normal_stop=result['normal_stop']))
    # Preserve the successful original, then reject a rehashed changed bank.
    shadow=base/'rehash_contradiction';shutil.copytree(base/'complete',shadow)
    result=json.loads((shadow/'result.json').read_bytes())
    item=next(v for v in result['captures'] if v['path']=='evidence/first-fault-final.npz')
    target=shadow/item['path']
    with np.load(target,allow_pickle=False) as archive:words=archive['words'].copy()
    words[32]=1
    np.savez(target,words=words);raw=target.read_bytes();item.update(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    (shadow/'result.json').write_text(json.dumps(result)+'\n')
    try:validate(shadow)
    except AssertionError:pass
    else:raise AssertionError('Rehashed changed fault bank accepted')
    assert not any(name=='cerebras' or name.startswith('cerebras.') for name in sys.modules)
    raws=list(base.rglob('*.npz'));raw_bytes=sum(p.stat().st_size for p in raws)
    assert len(raws)<=120 and raw_bytes<=3<<20
    result=dict(passed=True,cases=results,raw_files=len(raws),raw_bytes=raw_bytes,SDK_imports=0,SDK_invocations=0,
        scope='Changed capture/store and saved-array validation only; synthetic replies do not qualify device traffic or fault-store ordering')
    (Path(__file__).parent/'host-results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
