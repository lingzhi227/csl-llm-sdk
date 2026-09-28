"""Actual native fabric arrival into all16 admitted frontend arithmetic modules.

This component test injects frozen recurrent results; it cannot qualify GDN,
complete-layer execution, conversation service, or model throughput.
"""
import argparse,json,time
from pathlib import Path
import numpy as np
from backend import runtime,file_sha256


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--transport-batches',type=int,default=0);args=parser.parse_args()
    if args.transport_batches not in (0,4):raise ValueError('Bounded diagnostic profile')
    started=time.monotonic()
    def progress(phase,**fields):print(json.dumps(dict(phase=phase,seconds=time.monotonic()-started,**fields)),flush=True)
    progress('fixture')
    meta=json.loads(Path('fixture.json').read_text())
    if not meta['passed'] or file_sha256(Path('fixture.npz'))!=meta['fixture_sha256']:raise ValueError('Frozen numerical reference')
    data=dict(np.load('fixture.npz',allow_pickle=False));records=[];captures={};replay_values={}
    expected_counts=np.array(meta['consumer_projected_packets'],np.uint32)
    progress('runtime_load')
    with runtime(False) as (runner,dtype,order):
        progress('runtime_loaded')
        ids={n:runner.get_id(n) for n in ['wire','weights','history','gain','parameters','packet','output','audit','status']}
        options=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False)
        def write(name,value,y=1,bits=16):
            value=np.ascontiguousarray(value,dtype=np.uint32).reshape(16,-1)
            runner.memcpy_h2d(ids[name],value.reshape(-1),0,y,16,1,value.shape[1],data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**options)
        def read(name,count,y=1,bits=32):
            value=np.zeros((16,count),np.uint32)
            runner.memcpy_d2h(value.reshape(-1),ids[name],0,y,16,1,count,data_type=dtype.MEMCPY_16BIT if bits==16 else dtype.MEMCPY_32BIT,**options)
            return value if bits==32 else (value&65535).astype(np.uint16)
        def launch(name,*args):runner.launch(name,*args,nonblock=False)
        def status():launch('inspect');return read('status',10)
        def stream(frames,base=0):
            for start in range(0,max(map(len,frames)),64):
                if start%512==0:progress('packet_batch',start=start)
                buffer=np.zeros((16,321),np.uint32)
                for group,values in enumerate(frames):
                    chunk=values[start:start+64];buffer[group,0]=len(chunk);buffer[group,1:1+chunk.size]=chunk.reshape(-1)
                write('wire',buffer,y=0,bits=32);launch('send')
                if start==0 or args.transport_batches:
                    s=status();source_audit=read('audit',4,y=0)
                    diagnostic=dict(destination=s.tolist(),source_audit=source_audit.tolist(),expected=(np.asarray(base)+buffer[:,0]).tolist())
                    Path('diagnostic.json').write_text(json.dumps(diagnostic,indent=2)+'\n');progress('first_batch',**diagnostic)
                    target=np.asarray(base)+np.array([min(start+64,len(f)) for f in frames],np.uint32)
                    arrived=wait_count(target)
                    progress('batch_received',start=start,received=arrived[:,8].tolist())
        def wait_count(expected):
            deadline=time.monotonic()+(60 if args.transport_batches else 15)
            while True:
                s=status()
                if np.array_equal(s[:,8],expected):return s
                if np.any(s[:,8]>expected) or time.monotonic()>deadline:
                    Path('diagnostic.json').write_text(json.dumps(dict(expected=np.asarray(expected).tolist(),status=s.tolist(),source_audit=read('audit',4,y=0).tolist()),indent=2)+'\n')
                    raise ValueError('Native transport did not reach exact drain count: '+str(s[:,8].tolist()))
        for name in ['weights','gain','parameters']:
            progress('parameter_upload',name=name);write(name,data[name])
        if args.transport_batches:
            launch('clear');launch('begin',np.uint32(1))
            frames=[data[f'projected_0_{group}'][:64*args.transport_batches] for group in range(16)]
            stream(frames)
            s=wait_count(np.full(16,64*args.transport_batches,np.uint32))
            diagnostic_result=dict(passed=True,physical=False,normal_stop=False,scope='Four paced batches through the exact native receiver; transport diagnostic only.',
                frontend_numerical_qualified=False,recurrent_device_execution=False,full_model_speed_target_achieved=False,
                received=s[:,8].tolist(),source_audit=read('audit',4,y=0).tolist(),fixture_sha256=meta['fixture_sha256'])
        else:
            for replay,length in [(False,meta['positions']),(True,meta['reset_replay_positions'])]:
                progress('reset',replay=replay);launch('clear');np.testing.assert_array_equal(read('history',1920,bits=16),0)
                np.testing.assert_array_equal(status()[:,7],1)
                for position in range(length):
                    token=position+1;progress('token_begin',token=token,replay=replay);launch('begin',np.uint32(token))
                    streams=[data[f'projected_{position}_{group}'] for group in range(16)]
                    stream(streams);s=wait_count(expected_counts)
                    np.testing.assert_array_equal(s[:,0],token);np.testing.assert_array_equal(s[:,1],7)
                    np.testing.assert_array_equal(s[:,2:8],0);np.testing.assert_array_equal(s[:,9],expected_counts+192)
                    history=read('history',1920,bits=16).reshape(16,640,3)
                    np.testing.assert_array_equal(history,data[f'history_{position}'])
                    packets=np.zeros((16,3,387),np.uint32)
                    for head in range(3):
                        launch('prepare',np.uint16(head));packets[:,head]=read('packet',387)
                        np.testing.assert_array_equal(packets[:,head,0],token);launch('consume',np.uint16(head))
                    values=packets[:,:,1:].copy().view(np.float32)
                    error=np.abs(values.astype(np.float64)-data[f'packet_{position}'])
                    good=bool(np.isfinite(values).all() and np.all(error<=data[f'bound_{position}']))
                    if not good:
                        np.savez('rejected.npz',packet=packets,history=history,error=error,bound=data[f'bound_{position}'])
                        bad=np.argwhere(error>data[f'bound_{position}']);raise ValueError('Frozen packet interval failed at '+str(bad[:12].tolist()))
                    np.testing.assert_array_equal(packets[:,0,1:257],packets[:,1,1:257]);np.testing.assert_array_equal(packets[:,0,1:257],packets[:,2,1:257])
                    stream(list(data[f'returns_{position}']),base=expected_counts);s=wait_count(expected_counts+192)
                    np.testing.assert_array_equal(s[:,2],7);np.testing.assert_array_equal(s[:,3],7)
                    for head in range(3):launch('retire',np.uint16(head))
                    s=status();np.testing.assert_array_equal(s[:,4],7);np.testing.assert_array_equal(s[:,7],1)
                    output=read('output',384,bits=16);expanded=(output.astype(np.uint32)<<16).view(np.float32)
                    if not (np.isfinite(expanded).all() and np.all(expanded>=data[f'lower_{position}']) and np.all(expanded<=data[f'upper_{position}'])):
                        np.savez('rejected-gated.npz',actual=output,lower=data[f'lower_{position}'],upper=data[f'upper_{position}']);raise ValueError('Frozen gated output interval failed')
                    np.testing.assert_array_equal(read('audit',8),np.tile([token,710,320,192,3,3,3,1],(16,1)))
                    if replay:
                        for observed,prior in zip((packets,output,history),replay_values[position]):np.testing.assert_array_equal(observed,prior)
                    elif position<meta['reset_replay_positions']:replay_values[position]=(packets.copy(),output.copy(),history.copy())
                    for name,array in [('packet',packets),('output',output),('history',history)]:captures[f'{name}_{int(replay)}_{position}']=array
                    record=dict(replay=replay,position=position,groups=16,packet_interval_pass=True,gated_interval_pass=True,
                        exact_history=True,exact_shared_qk=True,exact_native_drain=True,maximum_packet_error=float(error.max()))
                    records.append(record);print(json.dumps(record),flush=True)
                    np.savez('actual.npz',**captures)
            for name in ['weights','gain','parameters']:
                np.testing.assert_array_equal(read(name,data[name].reshape(16,-1).shape[1],bits=16),data[name].reshape(16,-1))
    if args.transport_batches:
        diagnostic_result['normal_stop']=True;Path('result.json').write_text(json.dumps(diagnostic_result,indent=2)+'\n');return
    result=dict(passed=True,physical=False,normal_stop=True,groups=16,heads=48,positions=meta['positions'],reset_replay_positions=meta['reset_replay_positions'],
        records=records,parameters_retained=True,actual_native_fabric_arrival=True,exact_shared_qk=True,
        fixture_sha256=meta['fixture_sha256'],scope=meta['scope'],recurrent_device_execution=False,full_model_speed_target_achieved=False)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
