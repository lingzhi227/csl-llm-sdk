"""Frozen byte schedule with device-owned persistent dot-product accumulation."""
from array import array
from dataclasses import replace
import hashlib,json,os,struct,sys,time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.codecs import encode,decode as decode_wire
from qwen38.stream_schedule import Schedule,Tile
from qwen38.stream_numerics import policy,decode,vectors,reference,check
from cerebras.sdk.runtime.sdkruntimepybind import SdkRuntime,MemcpyDataType,MemcpyOrder,SimfabConfig,SdkTarget,get_platform
root=Path.cwd();manifest=json.loads((root/'source-manifest.json').read_text())
for name,h in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=h:raise ValueError('Frozen file mismatch: '+name)
profile=json.loads((root/'profile.json').read_text());columns=profile['columns'];plan=profile['tiles']
if json.loads((root/'numerical-policy.json').read_text())!=policy(columns):raise ValueError('Policy mismatch')
weights=decode((root/'slab-row-major.bf16').read_bytes())
phases=[];observations=[]
def record(phase,call=None,tile=None):
    phases.append({'phase':phase,'call':call,'tile':tile,'monotonic':time.monotonic()})
    (root/'lifecycle.json').write_text(json.dumps(phases,indent=2)+'\n')
(root/'runtime-environment.json').write_text(json.dumps({'python':sys.executable,'version':sys.version,'numpy':np.__version__,
 'pid':os.getpid(),'cwd':str(root),'tmpdir':os.environ.get('TMPDIR'),'cgroup':Path('/proc/self/cgroup').read_text()},indent=2)+'\n')
record('construct_enter');runner=SdkRuntime('out',get_platform(None,SimfabConfig(suppress_trace=True,num_threads=1,dump_core=False),SdkTarget.WSE3));record('construct_done')
names=('weights_storage','input_storage','output_storage','control','state','timing','weight_guard_snapshot');ids={n:runner.get_id(n) for n in names}
opts=dict(streaming=False,order=MemcpyOrder.ROW_MAJOR,nonblock=False)
def h2d(name,values,count,bits=32):
    runner.memcpy_h2d(ids[name],values,0,0,1,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT,**opts)
def d2h(name,count,dtype=np.float32,bits=32):
    values=np.zeros(count,dtype=dtype)
    runner.memcpy_d2h(values,ids[name],0,0,1,1,count,data_type=MemcpyDataType.MEMCPY_16BIT if bits==16 else MemcpyDataType.MEMCPY_32BIT,**opts)
    return values
record('load_enter');runner.load();record('load_done');record('run_enter');runner.run();record('run_done')
h2d('output_storage',np.array([1234.5]+[17.0]*128+[-4321.25],dtype=np.float32),130)
for call,(case,vector) in enumerate(vectors(columns),1):
    schedule=Schedule(columns,call);boundary=[]
    record('begin_enter',call);h2d('control',np.array([call,columns,0,0,0],dtype=np.uint32),5);runner.launch('begin',nonblock=False);record('begin_done',call)
    for tile in plan:
        index=tile['index'];valid=tile['valid'];offset=tile['offset']
        schedule.submit(Tile(call,index,offset,valid))
        raw=(root/tile['file']).read_bytes()
        tile_bits=[x[0] for x in struct.iter_unpack('<H',raw)]
        frame=encode(array('H',[0xa55a,*tile_bits,0x5aa5]),'memcpy16_containers')
        current_x=np.array([77.5]+vector[offset:offset+valid]+[7.0]*(112-valid)+[-88.25],dtype=np.float32)
        record('tile_h2d_enter',call,index)
        h2d('weights_storage',np.frombuffer(frame.payload,dtype='<u4').copy(),14338,16)
        h2d('input_storage',current_x,114);h2d('control',np.array([call,columns,index,offset,valid],dtype=np.uint32),5)
        record('tile_h2d_done',call,index);record('accumulate_enter',call,index)
        runner.launch('accumulate',nonblock=False);record('accumulate_done',call,index)
        if index in (0,len(plan)-2):
            record('boundary_readback_enter',call,index)
            partial=d2h('output_storage',130);state=d2h('state',11,np.uint32)
            record('boundary_readback_done',call,index)
            expected,bounds=reference(weights,vector,offset+valid)
            gate=check(partial[1:-1].tolist(),expected,bounds,case=='zero_after_nonzero')
            state_ok=state[0]==1 and state[1]==call and state[2]==columns and state[3]==index+1 and state[4]==offset+valid and state[5]==0 and state[6]==0
            boundary.append({'tile':index,'consumed':offset+valid,'numerical':gate,'state_ok':bool(state_ok),'state':state.tolist(),
                             'actual':partial[1:-1].tolist(),'expected':expected,'bounds':bounds})
    schedule.finalize();record('finalize_enter',call);runner.launch('finalize',nonblock=False);record('finalize_done',call)
    record('final_readback_enter',call)
    actual=d2h('output_storage',130);state=d2h('state',11,np.uint32)
    guard_snapshot=d2h('weight_guard_snapshot',5,np.uint32)
    wback=d2h('weights_storage',14338,np.uint32,16) if call==4 else None
    xback=d2h('input_storage',114)
    timing=d2h('timing',len(plan)*6,np.uint32,16)&np.uint32(65535)
    record('final_readback_done',call)
    values=decode_wire(replace(frame,payload=wback.astype('<u4',copy=False).tobytes()),'memcpy16_containers') if wback is not None else None
    expected,bounds=reference(weights,vector,columns);gate=check(actual[1:-1].tolist(),expected,bounds,case in ('last_valid','zero_after_nonzero'))
    ticks=[]
    for t in timing.reshape(-1,6).tolist():
        start=t[0]+(t[1]<<16)+(t[2]<<32);end=t[3]+(t[4]<<16)+(t[5]<<32);ticks.append((end-start)&((1<<48)-1))
    checks={'state_finalized':state.tolist()==[2,call,columns,len(plan),columns,call,0,plan[-1]['valid'],call*len(plan),call,call],
            'finalize_guard_snapshot':guard_snapshot.tolist()==[0xa55a,0x5aa5,call,1,2],
            'resident_last_input_and_guards':np.array_equal(xback.view(np.uint32),current_x.view(np.uint32)),
            'output_guards':actual[0]==1234.5 and actual[-1]==-4321.25,
            'stable_ids':all(runner.get_id(n)==ids[n] for n in names),
            'timestamps':all(0<t<(1<<40) for t in ticks),'boundary_checks':all(x['numerical']['passed'] and x['state_ok'] for x in boundary)}
    if call==4:checks['final_resident_tile_bits']=values==[0xa55a,*tile_bits,0x5aa5]
    row={'guard_observation':'finalize-time snapshot before commit; source-ordered markers, not a timestamp trace','weight_guard_snapshot':guard_snapshot.tolist(),'resident_tile_full_readback':call==4,'raw_timestamp_words':timing.tolist(),'call':call,'case':case,'onehot_index':columns-1 if case=='last_valid' else None,'numerical':gate,
         'checks':{k:bool(v) for k,v in checks.items()},'state':state.tolist(),'kernel_cycles_per_tile':ticks,
         'boundary':boundary,'actual':actual[1:-1].tolist(),'expected':expected,'bounds':bounds}
    observations.append(row);(root/'observations.json').write_text(json.dumps(observations,indent=2)+'\n')
    print('CALL',call,case,profile['kind'],gate,checks,'kernel_cycles',ticks,flush=True)
record('stop_enter');runner.stop();record('stop_done');record('python_complete')
passed=all(o['numerical']['passed'] and all(o['checks'].values()) for o in observations)
(root/'result.json').write_text(json.dumps({'passed':passed,'profile':profile['kind'],'columns':columns,'calls':4,'runtime_instances':1,
     'full5120_acceptance':passed and columns==5120,'policy':policy(columns)},indent=2)+'\n')
if not passed:raise RuntimeError('Streamed contraction gate failed')
