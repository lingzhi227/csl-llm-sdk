"""WP13 projection checks; each frozen profile requires its own controller dispatch."""
import hashlib
import json
import os
from pathlib import Path
import time
import numpy as np
from cerebras.sdk.runtime.sdkruntimepybind import (
    SdkRuntime, MemcpyDataType, MemcpyOrder, SimfabConfig, SdkTarget, get_platform,
)

root = Path.cwd()
profile = json.loads((root/'profile.json').read_text())
if profile['scope'] == 'wp13-four-pe-full5120-dense-diagnostic':
    assert (profile['pes'],profile['first_slab'],profile['case_indices']) == (4,0,[0])
elif profile['scope'] == 'wp13-single-pe-four-case-slab':
    assert profile['pes']==1 and profile['case_indices']==[0,1,2,3]
    assert type(profile['first_slab']) is int and 0<=profile['first_slab']<8
else:
    raise RuntimeError('Unexpected frozen projection scope')
manifest = json.loads((root/'source-manifest.json').read_text())['files']
for name, digest in manifest.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
        raise ValueError('Frozen source mismatch: '+name)
compiled = json.loads((root/'compiled-subset-before-run.json').read_text())
for name, digest in compiled.items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest() != digest:
        raise ValueError('Compiled identity mismatch: '+name)
admission = json.loads((root/'sram-admission.json').read_text())
if not admission['passed'] or len(admission['pes']) != profile['pes']:
    raise ValueError('Every compute PE needs actual SRAM admission')
ref = json.loads((root/'reference-plan.json').read_text())
weights_dir = root/'weights'  # Frozen workstation-local copy, visible in the isolated SDK bind.
for name, identity in ref['weight_files'].items():
    if hashlib.sha256((weights_dir/name).read_bytes()).hexdigest() != identity['sha256']:
        raise ValueError('Original weight identity mismatch')
mapping = json.loads((root/'weight-receipt.json').read_text())['plan']['tensors']

def decode(path, shape, offset=0):
    words = np.memmap(path, mode='r', dtype='<u2', offset=offset, shape=shape)
    return (np.asarray(words, dtype=np.uint32) << 16).view(np.float32)

matrix = np.concatenate([decode(weights_dir/mapping[n]['file'],
    (mapping[n]['selected_rows'][1],5120), mapping[n]['file_offset'])
    for n in ('q_proj','k_proj','v_proj')])
hidden = decode(root/'hidden.bf16',(4,5120))
oracle = np.load(root/'observations.npz',allow_pickle=False)
assert hashlib.sha256((root/'observations.npz').read_bytes()).hexdigest() == profile['observations_sha256']
pes = profile['pes']
first = profile['first_slab']
cases = profile['case_indices']
assert pes in (1,4,8) and 0 <= first <= 8-pes
assert cases == [0] or cases == [0,1,2,3]
assert profile['columns'] == 5120
names = ('weights_storage','input_storage','output_storage','bf16_storage',
         'control','state','timing','weight_guard_snapshot')
journal = []
copies = host_bytes = payload_bytes = launches = 0

def event(kind, **details):
    journal.append(dict(kind=kind,monotonic=time.monotonic(),**details))
    with (root/'journal.jsonl').open('a') as stream:
        stream.write(json.dumps(journal[-1])+'\n')
        stream.flush()
        os.fsync(stream.fileno())

event('construct_enter')
runner = SdkRuntime('out', get_platform(None,SimfabConfig(
    suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3))
ids = {name:runner.get_id(name) for name in names}
options = dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)

def transfer(direction,name,pe,count,bits=32,values=None,dtype=np.uint32):
    global copies,host_bytes,payload_bytes
    if count*4>65536:
        raise ValueError('Physical host buffer cap')
    array = np.zeros(count,dtype=dtype) if values is None else values
    assert array.nbytes == count*4 and array.flags.c_contiguous
    identity=dict(sequence=copies+1,direction=direction,name=name,symbol_id=int(ids[name]),
        pe=pe,x=pe,y=0,width=1,height=1,count=count,bits=bits,host_dtype=str(array.dtype),
        host_shape=list(array.shape),host_bytes=array.nbytes,native_bytes=count*bits//8,
        contiguous=bool(array.flags.c_contiguous))
    event('copy_enter',**identity,host_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    data_type = MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT
    fn = runner.memcpy_h2d if direction=='h2d' else runner.memcpy_d2h
    args = (ids[name],array) if direction=='h2d' else (array,ids[name])
    fn(*args,pe,0,1,1,count,data_type=data_type,**options)
    copies+=1;host_bytes+=count*4;payload_bytes+=count*bits//8
    event('copy_exit',**identity,host_sha256=hashlib.sha256(array.tobytes()).hexdigest())
    return array & np.uint32(65535) if bits==16 else array

def launch(name):
    global launches
    event('launch_enter',name=name)
    runner.launch(name,nonblock=False)
    launches+=1
    event('launch_exit',name=name)

def exact_state(call,tiles,consumed,phase,slab,released):
    return [phase,call,5120,tiles,consumed,call if phase in (2,4) else 0,
            0,80 if consumed==5120 else 112,(call-1)*46+tiles,call,
            call if phase in (2,4) else call-1,slab,released]

def check_fp32(actual, slab, vector, consumed):
    terms = slab[:,:consumed].astype(np.float64)*vector[:consumed].astype(np.float64)
    center = terms.sum(axis=1)
    magnitude = np.abs(terms).sum(axis=1)
    u=2**-24
    bounds = (consumed*u/(1-consumed*u)*magnitude + consumed*2**-53*magnitude + consumed*2**-149)*(1+2**-40)
    if not np.all(np.abs(actual.astype(np.float64)-center)<=bounds):
        raise ValueError('Original-source FP32 prefix/final interval')
    return center,bounds

observed = {}
passed_calls = []
summaries = []
primary_error=None
normal_stop=False
try:
    event('load_enter');runner.load();event('load_exit')
    event('run_enter');runner.run();event('run_exit')
    for pe in range(pes):
        transfer('h2d','output_storage',pe,130,values=np.array([1234.5]+[17.]*128+[-4321.25],dtype=np.float32))
        transfer('h2d','bf16_storage',pe,130,16,values=np.array([0xa55a]+[0x3f00]*128+[0x5aa5],dtype=np.uint32))
    for call, case_index in enumerate(cases,1):
        case = ref['cases'][case_index]
        x = hidden[case_index]
        for pe in range(pes):
            transfer('h2d','control',pe,5,values=np.array([call,5120,0,0,0],dtype=np.uint32))
        launch('begin')
        for tile in range(46):
            offset=tile*112;valid=min(112,5120-offset)
            for pe in range(pes):
                start=(first+pe)*128
                slab=matrix[start:start+128]
                packed=np.full((112,128),0x3e80,dtype=np.uint32)
                packed[:valid]=(slab[:,offset:offset+valid].copy().view(np.uint32)>>16).T
                w=np.concatenate([np.array([0xa55a],dtype=np.uint32),packed.flatten(),np.array([0x5aa5],dtype=np.uint32)])
                current_x=np.array([77.5]+x[offset:offset+valid].tolist()+[7.]*(112-valid)+[-88.25],dtype=np.float32)
                transfer('h2d','weights_storage',pe,14338,16,values=w)
                transfer('h2d','input_storage',pe,114,values=current_x)
                transfer('h2d','control',pe,5,values=np.array([call,5120,tile,offset,valid],dtype=np.uint32))
            launch('accumulate')
            if tile in (0,44):
                for pe in range(pes):
                    partial=transfer('d2h','output_storage',pe,130,dtype=np.float32)
                    state=transfer('d2h','state',pe,13)
                    observed[f'{call}_{pe}_prefix{tile}']=partial
                    observed[f'{call}_{pe}_prefix{tile}_state']=state
                    np.savez(root/'device-observations.npz',**observed)
                    assert partial[0]==1234.5 and partial[-1]==-4321.25
                    assert state.tolist()==exact_state(call,tile+1,offset+valid,1,first+pe,call-1)
                    check_fp32(partial[1:-1],matrix[(first+pe)*128:(first+pe+1)*128],x,offset+valid)
        launch('finalize')
        for pe in range(pes):
            out=transfer('d2h','output_storage',pe,130,dtype=np.float32)
            cast=transfer('d2h','bf16_storage',pe,130,16)
            state=transfer('d2h','state',pe,13)
            guard=transfer('d2h','weight_guard_snapshot',pe,5)
            xb=transfer('d2h','input_storage',pe,114,dtype=np.float32)
            timing=transfer('d2h','timing',pe,276,16)
            for name,array in dict(fp32=out,bf16=cast,state=state,guard=guard,input=xb,timing=timing).items():
                observed[f'{call}_{pe}_{name}']=array
            np.savez(root/'device-observations.npz',**observed)
            start=(first+pe)*128;slab=matrix[start:start+128]
            assert out[0]==1234.5 and out[-1]==-4321.25 and cast[0]==0xa55a and cast[-1]==0x5aa5
            assert state.tolist()==exact_state(call,46,5120,2,first+pe,call-1)
            assert guard.tolist()==[0xa55a,0x5aa5,call,1,2]
            expected_x=np.array([77.5]+x[5040:].tolist()+[7.]*32+[-88.25],dtype=np.float32)
            assert np.array_equal(xb.view(np.uint32),expected_x.view(np.uint32))
            center,bounds=check_fp32(out[1:-1],slab,x,5120)
            raw=out[1:-1].view(np.uint32)
            expected_cast=((raw+np.uint32(0x7fff)+((raw>>16)&1))>>16)&65535
            assert np.array_equal(cast[1:-1],expected_cast)
            actual=((cast[1:-1]<<16).astype(np.uint32)).view(np.float32)
            nominal=oracle[case+'__nominal'][start:start+128]
            radius=oracle[case+'__radius'][start:start+128]
            assert np.all(np.abs(actual.astype(np.float64)-nominal)<=radius)
            official=oracle[case+'__official'][start:start+128]
            if case_index in (2,3):
                exact=slab[:,5119] if case_index==2 else np.zeros(128,dtype=np.float32)
                assert np.array_equal(out[1:-1],exact)
            ticks=[]
            for t in timing.reshape(46,6).tolist():
                ticks.append(((t[3]+(t[4]<<16)+(t[5]<<32))-(t[0]+(t[1]<<16)+(t[2]<<32)))&((1<<48)-1))
            assert all(0<t<1<<40 for t in ticks)
            if call==len(cases):
                wb=transfer('d2h','weights_storage',pe,14338,16)
                observed[f'{call}_{pe}_resident_weights']=wb
                np.savez(root/'device-observations.npz',**observed)
                packed=np.full((112,128),0x3e80,dtype=np.uint32)
                packed[:80]=(slab[:,5040:].copy().view(np.uint32)>>16).T
                assert np.array_equal(wb,np.concatenate([np.array([0xa55a],dtype=np.uint32),packed.flatten(),np.array([0x5aa5],dtype=np.uint32)]))
            observed[f'{call}_{pe}_fp32']=out
            observed[f'{call}_{pe}_bf16']=cast
            observed[f'{call}_{pe}_timing']=timing
            observed[f'{call}_{pe}_state']=state
            summaries.append(dict(call=call,pe=pe,global_slab=first+pe,
                max_fp32_error=float(np.max(np.abs(out[1:-1].astype(np.float64)-center))),
                max_fp32_bound=float(bounds.max()),nominal_bf16_mismatches=int(np.count_nonzero(actual!=nominal)),
                official_bf16_mismatches=int(np.count_nonzero(actual!=official)),
                exact_rne_elements=128,kernel_cycles_per_tile=ticks))
            (root/'checks.json').write_text(json.dumps(summaries,indent=2)+'\n')
        launch('release')
        for pe in range(pes):
            state=transfer('d2h','state',pe,13)
            observed[f'{call}_{pe}_released_state']=state
            assert state.tolist()==exact_state(call,46,5120,4,first+pe,call)
        passed_calls.append(case)
        np.savez(root/'device-observations.npz',**observed)
        print('PASS',case,'selected slabs',first,pes,flush=True)
    assert all(runner.get_id(name)==ids[name] for name in names)
except BaseException as exc:
    primary_error=repr(exc)
    event('execution_error',error=primary_error,error_type=type(exc).__name__)
    raise
finally:
    try:
        event('stop_enter')
        runner.stop()
        event('stop_exit')
        normal_stop=True
    except BaseException as exc:
        event('stop_error',error=repr(exc),original_error=primary_error)
        if primary_error is None:
            raise
assert normal_stop
for name,digest in compiled.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
for name,digest in manifest.items():
    assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
for name,identity in ref['weight_files'].items():
    assert hashlib.sha256((weights_dir/name).read_bytes()).hexdigest()==identity['sha256']
traffic=profile['traffic']
assert (copies,host_bytes,payload_bytes,launches)==tuple(traffic[k] for k in ('copies','host_bytes','payload_bytes','launches'))
(root/'result.json').write_text(json.dumps(dict(passed=True,cases=passed_calls,
    full_head_qualified=pes==8 and first==0 and cases==[0,1,2,3],
    slab_qualified=pes==1 and cases==[0,1,2,3],global_slabs=list(range(first,first+pes)),
    global_row_range=[first*128,(first+pes)*128],input_sha256=profile['input_sha256'],
    composition_qualified=False,normal_stop=True,copies=copies,host_bytes=host_bytes,
    payload_bytes=payload_bytes,launches=launches),indent=2)+'\n')
