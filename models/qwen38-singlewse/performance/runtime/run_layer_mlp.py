"""Full original MLP device qualification; host provides only boundary inputs.

Every intermediate computation and route executes on the device. Frozen oracle
arrays are read only for post-execution comparisons. All copies are bounded by
one physical row; no per-layer weight streaming occurs between warm epochs.
"""
import argparse,json,time
from collections import Counter
from pathlib import Path
import numpy as np
from backend import runtime,file_sha256,complete_mlp_profile


def row_runs(records,key):
    """Consecutive X positions with identical element counts, in one row."""
    run=[]
    for record in sorted(records,key=lambda r:(r['pe'][1],r['pe'][0])):
        if run and (record['pe']!=[run[-1]['pe'][0]+1,run[-1]['pe'][1]] or key(record)!=key(run[0])):
            yield run;run=[]
        run.append(record)
    if run:yield run


def native_code(code):
    code=np.asarray(code,dtype=np.uint16)
    return ((code&127)<<7)|((code&128)<<8)


class BankFile:
    """Read only the current physical-row slice; do not map a400MB bank file.

    gRPC creates many thread stacks in the same4GiB address-space budget. A
    full-bank mmap needlessly reserves space during protobuf marshalling.
    """
    def __init__(self,path):
        self.path=Path(path)
        with self.path.open('rb') as f:
            if np.lib.format.read_magic(f)!=(1,0):raise ValueError('Original bank NPY version')
            shape,fortran,dtype=np.lib.format.read_array_header_1_0(f)
            if fortran or len(shape)!=1 or dtype!=np.dtype('<u4'):raise ValueError('Original bank array format')
            self.offset=f.tell();self.words=shape[0];self.nbytes=4*self.words
        self.stamp=self.path.stat()
        if self.stamp.st_size!=self.offset+self.nbytes:raise ValueError('Original bank extent')

    def __getitem__(self,part):
        if not isinstance(part,slice) or part.step is not None or not 0<=part.start<part.stop<=self.words:
            raise ValueError('Bounded original bank slice')
        size=(part.stop-part.start)*4
        if size>8<<20:raise ValueError('Original bank host buffer limit')
        now=self.path.stat()
        if (now.st_ino,now.st_size,now.st_mtime_ns)!=(self.stamp.st_ino,self.stamp.st_size,self.stamp.st_mtime_ns):raise ValueError('Original bank file changed')
        with self.path.open('rb') as f:f.seek(self.offset+4*part.start);raw=f.read(size)
        if len(raw)!=size:raise ValueError('Short original bank read')
        return np.frombuffer(raw,dtype='<u4')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--physical',action='store_true');physical=parser.parse_args().physical
    config=json.loads(Path('experiment.json').read_text());meta=json.loads(Path('fixture.json').read_text())
    if not complete_mlp_profile(config):raise ValueError('Exact original MLP profile required')
    if file_sha256(Path('fixture.json'))!=config['fixture_metadata_sha256']:raise ValueError('Oracle metadata identity')
    for name,expected in meta['hashes'].items():
        if file_sha256(Path(name))!=expected:raise ValueError('Frozen fixture identity: '+name)
    if meta['acceptance']!=config['acceptance']:raise ValueError('Changed acceptance')
    data=dict(np.load('fixture.npz',allow_pickle=False));banks=BankFile('banks.npy')
    lut=np.load('silu-table.npy',allow_pickle=False)
    if lut.shape!=(3328,) or banks.nbytes!=meta['allocated_bank_bytes']:raise ValueError('Fixture shape')
    index=json.loads(Path('bank-index.json').read_text())['records']
    workers=json.loads(Path('workers.json').read_text());binding=json.loads(Path('network-binding.json').read_text())
    setups=json.loads(Path('worker-setups.json').read_text());senders=binding['senders'];norm_bridge=bool(meta.get('norm_bridge'))
    if norm_bridge!=bool(config.get('norm_bridge')) or norm_bridge!=bool(binding.get('norm_bridge')):raise ValueError('Norm graph/fixture mismatch')
    origin=config['logical_origin'];width,height=config['application'];controller=meta['controller']
    expected_audit=np.zeros((height,width,4),np.uint32)
    setup_array=np.zeros((height,width,5),np.uint32)
    for pe,value in setups:setup_array[pe[1]-origin[1],pe[0]-origin[0]]=value
    for worker in workers:
        x,y=worker['pe'];expected_audit[y-origin[1],x-origin[0],1:]=[worker['parts'],worker['iterations'],1]
    expected_audit[controller[1]-origin[1],controller[0]-origin[0],1:]=[len(binding['prepare_schedule'])+len(binding['grant_schedule']),len(binding['grant_schedule']),1]
    for sender in senders:
        if sender.get('norm_bridge'):
            x,y=sender['pe'];expected_audit[y-origin[1],x-origin[0],1:]=[2,0,1]
            x,y=sender['norm_pe'];expected_audit[y-origin[1],x-origin[0],1:]=[2,2,1]
    grants=Counter(g['target'] for g in binding['grant_schedule']);prepares=Counter(g['target'] for g in binding['prepare_schedule']);captured={};observations=[]
    def save():np.savez('actual.npz',**captured)
    began=time.monotonic()
    with runtime(physical) as (runner,dtype,order):
        loaded=time.monotonic();names=['bank','setup','quant_input','silu_lut','audit','mlp_output','sender_status','native_input','native_scales','ticks','transport_stats']
        if norm_bridge:names += ['residual_input','norm_gains','successor_output','bridge_stats','norm_pre_output']
        ids={name:runner.get_id(name) for name in names}
        def copy(name,pe,count,*,value=None,span=1,lines=1,half=False):
            opts=dict(streaming=False,order=order.ROW_MAJOR,nonblock=False,
                      data_type=dtype.MEMCPY_16BIT if half else dtype.MEMCPY_32BIT)
            x,y=pe[0]-origin[0],pe[1]-origin[1]
            if not (0<=x<x+span<=width and 0<=y<y+lines<=height):raise ValueError('Copy outside component')
            # SDK 16-bit transfers still require one uint32 host entry per u16.
            a=np.zeros(count*span*lines,np.uint32) if value is None else np.ascontiguousarray(value,dtype=np.uint32).reshape(-1)
            if a.size!=count*span*lines or a.nbytes>8<<20:raise ValueError('Bounded copy extent')
            if value is None:runner.memcpy_d2h(a,ids[name],x,y,span,lines,count,**opts)
            else:runner.memcpy_h2d(ids[name],a,x,y,span,lines,count,**opts)
            return a
        bank_runs=list(row_runs(index,lambda r:r['words']))
        print(json.dumps(dict(phase='initialize',bank_bytes=banks.nbytes,copy_calls=len(bank_runs))),flush=True)
        for run in bank_runs:
            count=run[0]['words'];start=run[0]['offset'];length=count*len(run)
            if [r['offset'] for r in run]!=list(range(start,start+length,count)):raise ValueError('Noncontiguous bank fixture')
            copy('bank',run[0]['pe'],count,value=banks[start:start+length],span=len(run))
        for sender in senders:
            if sender.get('fusion_producer') is not None:copy('silu_lut',sender['pe'],3328,value=lut.astype(np.uint32),half=True)
        if norm_bridge:
            for sender in senders:
                if sender.get('norm_bridge'):
                    g=sender['input_group'];gain=data['norm_gains'][:,g*128:(g+1)*128].astype('<u2',copy=True)
                    copy('norm_gains',sender['norm_pe'],128,value=gain.reshape(-1).view('<u4'))
        copy('setup',origin,5,value=setup_array,span=width,lines=height,half=True)
        initialized=time.monotonic();print(json.dumps(dict(phase='initialized',seconds=initialized-loaded)),flush=True)
        input_runs=list(row_runs(workers,lambda w:(w['columns'],w['max_parts'])))
        for case in meta['cases']:
            i=case['index'];epoch=case['epoch']
            for sender in senders:
                if 'input_group' in sender:
                    group=sender['input_group'];values=data[f'input_{i}'][group*128:(group+1)*128]
                    if norm_bridge:
                        copy('residual_input',sender['norm_pe'],64,value=data[f'left_{i}'][group*128:(group+1)*128].astype('<u2',copy=False).view('<u4'))
                        copy('quant_input',sender['norm_pe'],64,value=data[f'mixer_{i}'][group*128:(group+1)*128].astype('<u2',copy=False).view('<u4'))
                    else:copy('quant_input',sender['pe'],64,value=values.astype('<u2',copy=False).view('<u4'))
            started=time.monotonic();runner.launch('arm',np.uint32(epoch),nonblock=False)
            runner.launch('start',nonblock=False);runner.launch('finish',nonblock=False)
            finished=time.monotonic()
            output=copy('mlp_output',controller,2560).view(np.uint16)
            successor=copy('successor_output',controller,2560).view(np.uint16) if norm_bridge else None
            completed_output=time.monotonic();ticks=copy('ticks',controller,12,half=True)
            audit=copy('audit',origin,4,span=width,lines=height).reshape(height,width,4)
            captured[f'output_{i}']=output;captured[f'ticks_{i}']=ticks;captured[f'audit_{i}']=audit;save()
            transport=copy('transport_stats',controller,4);captured[f'transport_{i}']=transport;save()
            packet_count=len(binding['distribution_packets'])
            np.testing.assert_array_equal(transport[:2],[packet_count,packet_count])
            if np.any(transport[2:]>len(binding['grant_schedule'])):raise ValueError('Invalid prefetch lease counters')
            np.testing.assert_array_equal(output,data[f'down_bf16_{i}'])
            if norm_bridge:
                captured[f'successor_{i}']=successor;bridge=copy('bridge_stats',controller,2);captured[f'bridge_{i}']=bridge;save()
                np.testing.assert_array_equal(successor,data[f'successor_{i}']);np.testing.assert_array_equal(bridge,[40,40])
                for sender in senders:
                    if sender.get('norm_bridge'):
                        group=sender['input_group'];first=group*128
                        residual=copy('residual_input',sender['norm_pe'],64).view(np.uint16)
                        preceding=copy('norm_pre_output',sender['pe'],64).view(np.uint16)
                        captured[f'residual_{i}_{group}']=residual;captured[f'preceding_{i}_{group}']=preceding;save()
                        np.testing.assert_array_equal(residual,data[f'residual_{i}'][first:first+128])
                        np.testing.assert_array_equal(preceding,data[f'input_{i}'][first:first+128])
            expected_audit[:,:,0]=epoch;np.testing.assert_array_equal(audit,expected_audit)
            for sender in senders:
                actual=copy('sender_status',sender['pe'],4);sid=sender['id'];captured[f'sender_{i}_{sid}']=actual
                np.testing.assert_array_equal(actual,np.array([epoch,prepares[sid]+grants[sid],grants[sid],1],np.uint32))
            # Read retained ingress, including every ragged K shard. Unused
            # capacity is intentionally excluded: only the live parts are input.
            for run in input_runs:
                columns=run[0]['columns'];parts=run[0]['max_parts'];count=columns*parts
                values=copy('native_input',run[0]['pe'],count,span=len(run),half=True).reshape(len(run),count)
                scales=copy('native_scales',run[0]['pe'],parts,span=len(run)).reshape(len(run),parts)
                key='_'.join(map(str,run[0]['pe']));captured[f'inputs_{i}_{key}']=values;captured[f'scales_{i}_{key}']=scales
                for j,w in enumerate(run):
                    family='gate' if w['role']=='gate_up' else 'down'
                    codes=data[f'{family}_operand_codes_{i}'].reshape(-1);truth=data[f'{family}_operand_scales_{i}'].reshape(-1).view(np.uint32)
                    for part,k in enumerate(w['native_input_slices']):
                        first=k*columns
                        try:
                            np.testing.assert_array_equal(values[j,part*columns:(part+1)*columns],native_code(codes[first:first+columns]))
                            np.testing.assert_array_equal(scales[j,part],truth[first//128])
                        except AssertionError:save();raise
            stamps=[sum(int(ticks[offset+j])<<(16*j) for j in range(3)) for offset in range(0,12,3)]
            phases={name:(b-a)%(1<<48) for name,a,b in zip(['input_prepare_and_distribute','fused_gate_up_and_distribute','down_and_collect'],stamps,stamps[1:])}
            if norm_bridge:phases['down_residual_successor_rms_and_collect']=phases.pop('down_and_collect')
            observation=dict(**case,all_norm_and_residuals_exact=True if norm_bridge else None,all_outputs_bit_exact=True,all_native_inputs_exact=True,all_counters_exact=True,
                all_pe_drained=True,host_arm_start_finish_seconds=finished-started,completed_output_seconds=completed_output-started,
                controller_cycles=(stamps[-1]-stamps[0])%(1<<48),controller_phase_cycles=phases,
                controller_transport=dict(zip(['packets_issued','packets_retired','grants_during_packet_lease','frames_arrived_during_packet_lease'],map(int,transport))))
            observations.append(observation);save();print(json.dumps(dict(phase='epoch_passed',**observation)),flush=True)
        print(json.dumps(dict(phase='retention',bank_bytes=banks.nbytes)),flush=True)
        for run in bank_runs:
            count=run[0]['words'];start=run[0]['offset'];length=count*len(run)
            np.testing.assert_array_equal(copy('bank',run[0]['pe'],count,span=len(run)),banks[start:start+length])
        for sender in senders:
            if sender.get('fusion_producer') is not None:np.testing.assert_array_equal(copy('silu_lut',sender['pe'],3328,half=True),lut)
        if norm_bridge:
            for sender in senders:
                if sender.get('norm_bridge'):
                    first=sender['input_group']*128
                    retained_gains=copy('norm_gains',sender['norm_pe'],128).view(np.uint16).reshape(2,128)
                    np.testing.assert_array_equal(retained_gains,data['norm_gains'][:,first:first+128])
        retained=time.monotonic();save()
    result=dict(passed=True,physical=physical,normal_stop=True,full_model=False,scope=meta['scope'],
        fixture_metadata_sha256=config['fixture_metadata_sha256'],original_weights_resident=True,
        all_original_banks_retained=True,all_luts_retained=True,all_native_inputs_exact=True,
        all_counters_exact=True,all_pe_drained=True,epochs=observations,
        norm_bridge=norm_bridge,all_norm_gains_retained=True if norm_bridge else None,
        original_output_values_checked=4*5120,additional_norm_and_residual_values_checked=3*4*5120 if norm_bridge else 0,runtime_load_seconds=loaded-began,
        initialization_seconds=initialized-loaded,full_runtime_seconds=time.monotonic()-began,
        normal_stop_seconds=time.monotonic()-retained,full_model_tps=None)
    Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)


if __name__=='__main__':main()
