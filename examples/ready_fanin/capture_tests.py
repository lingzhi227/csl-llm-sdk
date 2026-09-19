"""Exercise actual capture/store error paths with tiny synthetic SDK-free replies."""
from pathlib import Path
import json
import numpy as np
import capture as module
from source_tests import valid

class Types:MEMCPY_32BIT=32
class Order:ROW_MAJOR=1
class Fake:
    def __init__(self,mode):
        self.mode=mode;self.closed=False;self.stop_attempts=0;self.calls=[];self.payload=valid()
        if mode=='bad_ready':self.payload[1][12]=0x400108;self.payload[1][14]=4
        if mode=='incomplete':self.payload[0][40][1]=self.payload[0][40][2]=2
    def get_id(self,n):return n
    def launch(self,n,**opts):self.calls.append(('launch',n))
    def memcpy_d2h(self,out,name,x,y,w,h,count,**opts):
        self.calls.append(('copy',name,x,y,w,h,count))
        if name=='state':
            data=[[1,4 if yy else 1 if xx==0 else 2 if xx==2 else 3 if 40<=xx<176 else 0,0,xx] for yy in range(2) for xx in range(178)]
        elif name=='diagnostic_status':data=self.payload[0]
        elif x==0:data=[self.payload[1]]
        elif x==2:
            if self.mode=='peer_copy_error':raise RuntimeError('injected peer record transfer failure')
            data=[self.payload[2]]
        else:data=self.payload[3]
        value=np.asarray(data,dtype=np.uint32).reshape(-1);assert value.size==out.size;out[:]=value
    def stop(self):
        self.stop_attempts+=1
        if self.mode=='stop_error':raise RuntimeError('injected context exit failure')
        self.closed=True

def main():
    base=Path(__file__).resolve().parent/'test-evidence';assert not base.exists();base.mkdir();results=[]
    for mode in ['complete','peer_copy_error','stop_error','bad_ready','incomplete']:
        root=base/mode;root.mkdir();runner=Fake(mode);caught=None
        try:module.capture(root,runner,Types,Order)
        except (RuntimeError,ValueError) as exc:caught=str(exc)
        result=json.loads((root/'result.json').read_text());progress=json.loads((root/'capture-progress.json').read_text());raw=list((root/'evidence').glob('*.npz'))
        assert len(raw)==result['copies']==len(result['captures'])==progress['counts']['copies']
        assert runner.stop_attempts==1
        if mode=='complete':assert caught is None and result['status']=='passed' and result['normal_stop'] and result['copies']==6
        else:assert caught and result['status']=='failed'
        if mode=='peer_copy_error':assert len(raw)==3 and (root/'evidence/origin.npz').exists() and not (root/'evidence/peer.npz').exists() and result['normal_stop']
        if mode=='stop_error':assert len(raw)==6 and not result['normal_stop'] and json.loads((root/'monitor-deadline.json').read_text())['active']
        if mode=='bad_ready':assert result['partial_summary']['first_reported_error']['reason']==4 and len(raw)==6
        if mode=='incomplete':assert result['copies']==13 and result['host_slot_bytes']==101504
        for receipt in result['captures']:
            p=root/receipt['path'];assert p.stat().st_size==receipt['bytes']
            import hashlib
            assert hashlib.sha256(p.read_bytes()).hexdigest()==receipt['sha256']
            with np.load(p,allow_pickle=False) as data:assert data.files==['words'] and data['words'].dtype==np.uint32
        results.append(dict(case=mode,copies=result['copies'],raw_bytes=sum(p.stat().st_size for p in raw),normal_stop=result['normal_stop'],status=result['status']))
    out=dict(passed=True,cases=results,SDK_invocations=0,hardware_invocations=0,scope='Actual host capture/store functions and durable fault prefixes only; synthetic replies do not validate scheduling, device data or physical transfers.')
    (Path(__file__).parent/'capture-tests.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out))
if __name__=='__main__':main()
