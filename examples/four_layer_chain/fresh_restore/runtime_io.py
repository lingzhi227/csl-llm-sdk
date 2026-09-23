"""Blocking nonweight/retention copies with actual bindings and durable raw.

An already admitted runtime is injected by its owner. This adapter never
allocates a backend or evaluates a neural reference. Parameter transfers use
immutable prepared-file receipts; successful readbacks retain exact equality
receipts rather than duplicating hundreds of megabytes of raw weights.
"""
import hashlib,time
from runtime_liveness import operation, mark

class IO:
    def __init__(self,runtime,types,order,bindings,store,budget,before_call):
        import numpy as np
        self.np=np;self.runtime=runtime;self.types=types;self.order=order
        self.store=store;self.budget=budget;self.before_call=before_call
        self.copies=self.host_bytes=self.launches=0;self.allowed={};self.symbols={}
        if not bindings['all_planned_transfers_within_actual_arrays']:
            raise ValueError('Actual program/array binding is required before IO')
        for item in bindings['transfers']:
            key=(item['symbol'],*item['box'],item['count'])
            if key in self.allowed and self.allowed[key]['dtype']!=item['dtype']:
                raise ValueError('Conflicting typed copy binding')
            if key not in self.allowed:self.allowed[key]=dict(item,directions=set(item['directions']))
            else:self.allowed[key]['directions'].update(item['directions'])
        for name in {key[0] for key in self.allowed}:self.symbols[name]=runtime.get_id(name)

    def copy(self,direction,item,label,value=None,prepared_receipt=None,retain_raw=True):
        np=self.np;box=[item[k] for k in ('x','y','width','height','count')]
        binding=self.allowed.get((item['symbol'],*box))
        if binding is None or direction not in binding['directions'] or item['dtype']!=binding['dtype']:
            raise ValueError('Exact bound copy geometry/type/direction')
        size=binding['host_bytes'];count=item['width']*item['height']*item['count'];dtype=item['dtype']
        if self.copies>=self.budget['max_copies'] or size>16<<20 or self.host_bytes+size>self.budget['max_host_bytes']:
            raise ValueError('Finite copy and actual host-slot byte budget')
        if not retain_raw and prepared_receipt is None:
            raise ValueError('Only immutable prepared parameters may omit successful raw duplication')
        if prepared_receipt is not None:
            if not {'file','sha256','shape','dtype','native_sha256'}<=prepared_receipt.keys():
                raise ValueError('Prepared file and exact native slice identity required')
        host_type=np.float32 if dtype=='f32' else np.uint32
        expected=None
        if direction=='h2d' or prepared_receipt is not None:
            if value is None or value.size!=count:raise ValueError('Exact prepared input size')
            native_type='<u2' if dtype=='u16' else '<f4' if dtype=='f32' else '<u4'
            if value.dtype!=np.dtype(native_type):raise ValueError('Exact native input dtype')
            expected=np.ascontiguousarray(value).reshape(-1)
            if prepared_receipt is not None and hashlib.sha256(expected.tobytes()).hexdigest()!=prepared_receipt['native_sha256']:
                raise ValueError('Prepared native slice changed')
        data=np.ascontiguousarray(expected,dtype=host_type) if direction=='h2d' else np.empty(count,dtype=host_type)
        input_receipt=None
        if direction=='h2d' and retain_raw:
            input_receipt=self.store.save(label,expected.tobytes(),symbol=item['symbol'],rectangle=box,dtype=dtype)
        self.before_call(direction,item['symbol'],box)
        entered=self.store.event('copy_enter',index=self.copies,direction=direction,symbol=item['symbol'],
            rectangle=box,dtype=dtype,host_bytes=size,label=label,prepared=prepared_receipt)
        options=dict(data_type=self.types.MEMCPY_16BIT if dtype=='u16' else self.types.MEMCPY_32BIT,
            order=self.order.ROW_MAJOR,streaming=False,nonblock=False)
        started=time.monotonic();x,y,w,h,n=box
        with operation('sdk_'+direction,label):
            if direction=='h2d':self.runtime.memcpy_h2d(self.symbols[item['symbol']],data,x,y,w,h,n,**options)
            elif direction=='d2h':self.runtime.memcpy_d2h(data,self.symbols[item['symbol']],x,y,w,h,n,**options)
            else:raise ValueError('Known copy direction')
        self.copies+=1;self.host_bytes+=size
        self.store.event('copy_returned',index=self.copies-1,enter_sequence=entered['sequence'],rpc_seconds=time.monotonic()-started)
        if dtype=='u16' and np.any(data>65535):
            self.store.save(label+'-invalid-container',data.astype('<u4',copy=False).tobytes(),
                symbol=item['symbol'],rectangle=box,dtype='u32')
            raise ValueError('Actual native16 container has nonzero high half')
        native=data.astype('<u2',copy=False) if dtype=='u16' else data.astype('<f4' if dtype=='f32' else '<u4',copy=False)
        raw=native.tobytes();digest=hashlib.sha256(raw).hexdigest();receipt=None
        if prepared_receipt is not None and direction=='d2h':
            if raw!=expected.tobytes():
                self.store.save(label+'-parameter-mismatch',raw,symbol=item['symbol'],rectangle=box,dtype=dtype)
                raise ValueError('Original resident parameter readback mismatch')
            receipt=dict(prepared=prepared_receipt,actual_native_sha256=digest,exact_match=True,raw_parameters_duplicated=False)
            self.store.event('parameter_verified',index=self.copies-1,receipt=receipt)
        elif retain_raw:
            receipt=input_receipt if direction=='h2d' else self.store.save(label,raw,symbol=item['symbol'],rectangle=box,dtype=dtype)
        else:receipt=dict(prepared=prepared_receipt,native_sha256=digest,raw_parameters_duplicated=False)
        self.store.event('copy_exit',index=self.copies-1,receipt=receipt)
        mark('copy_return_to_caller',label)
        return native.reshape(h,w,n),receipt

    def launch(self,name):
        if name not in ('initialize','prepare','compute','reset') or self.launches>=self.budget['max_launches']:
            raise ValueError('Finite known original Layer0 launches')
        self.before_call('launch',name,None);event=self.store.event('launch_enter',name=name,index=self.launches)
        # Match the accepted appliance call mode. Its RPC return is not the
        # device completion fence: independent observer, every-PE state, and
        # two full transport snapshots must pass before reuse/reset. The
        # separately owned parent deadline also covers a blocked launch RPC.
        started=time.monotonic()
        with operation('sdk_launch',name):self.runtime.launch(name,nonblock=False)
        self.launches+=1
        self.store.event('launch_exit',name=name,index=self.launches-1,enter_sequence=event['sequence'],
            rpc_seconds=time.monotonic()-started,nonblock=False,device_completion_proved=False)
