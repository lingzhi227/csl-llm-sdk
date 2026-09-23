"""Separate post-release complete evidence and original Layer3 numeric audit."""
from pathlib import Path
import hashlib,io,json,time
from runtime_gate import ROOT,verify,pinned
from runtime_plan import build,box
from runtime_prepared import Prepared
from runtime_boundary import Boundary,observer,state,require
from runtime_evidence import Evidence
from runtime_numeric import audit
from runtime_capture import atomic_json

def scheduled(copies,observations):
    def transfers(direction,items,prefix):
        for i,item in enumerate(items):yield dict(kind='copy',direction=direction,item=item,label=f'{prefix}-{i:04}')
    def command(name):return dict(kind='launch',name=name)
    weights_done=False
    for i,item in enumerate(copies['uploads']):
        if item['symbol']=='weights':
            if not weights_done:
                yield dict(kind='weight_upload',items=[p for p in copies['uploads'] if p['symbol']=='weights']);weights_done=True
        else:yield dict(kind='copy',direction='h2d',item=item,label=f'upload-once-{i:04}')
    yield command('initialize')
    yield from transfers('d2h',copies['uploads'],'initial-retention')
    yield from transfers('d2h',copies['state_copies'],'s0-initialized-state')
    yield from transfers('d2h',copies['persistent_copies'],'initialized-cache')
    for serial in (1,2,3):
        if serial==3:
            yield command('reset');yield from transfers('d2h',copies['state_copies'],'s2-reset-state')
            yield from transfers('d2h',copies['persistent_copies'],'reset-cache')
        yield dict(kind='copy',direction='h2d',item=copies['hidden_input'],label=f's{serial}-original-input')
        yield command('prepare');yield from transfers('d2h',copies['state_copies'],f's{serial}-prepared-state')
        yield command('compute')
        attempts=observations[serial-1]['attempts'];require(1<=attempts<=128,'Finite recorded observer attempts')
        for i in range(attempts):yield dict(kind='copy',direction='d2h',item=copies['observer'],label=f's{serial}-observer-{i:03}')
        yield from transfers('d2h',copies['state_copies'],f's{serial}-complete-state')
        for snapshot in range(2):yield from transfers('d2h',copies['transport_copies'],f's{serial}-transport-{snapshot}')
        yield from transfers('d2h',copies['diagnostic_copies'],f's{serial}-diagnostic')
    yield from transfers('d2h',copies['uploads'],'terminal-retention')

def journal(root,captured,copies,prepared):
    sequence=0;seconds=-1.;active=None;pending=None;copy_index=launch_index=0;host_bytes=0
    raw_seen=set();parameter_reads=0;events=iter(scheduled(copies,captured['observations']));last=None;waiting_input=None
    weight_events=[];weight_result=None
    weight_kinds={'weight_batch_enter','weight_submit_enter','weight_submit_returned','weight_wait_enter',
                  'weight_task_completed','weight_batch_exit','all_original_weights_completed'}
    path=Path(root)/'journal.jsonl';require(path.stat().st_size==captured['journal_bytes']<=128<<20,'Complete bounded journal size')
    with path.open('rb') as f:
        for raw in f:
            require(len(raw)<=65536,'Bounded single journal record');row=json.loads(raw)
            require(row['sequence']==sequence and row['seconds']>=seconds,'Contiguous monotonic journal')
            sequence+=1;seconds=row['seconds'];kind=row['kind'];last=kind
            if kind.startswith('weight_') or kind=='all_original_weights_completed':
                require(kind in weight_kinds,'No asynchronous failure or unknown weight event in accepted capture')
                if not weight_events:
                    require(active is None and waiting_input is None and weight_result is None,'Exactly one distinct weight upload phase')
                    active=next(events,None);require(active is not None and active['kind']=='weight_upload','Scheduled original weight phase')
                require(active is not None and active['kind']=='weight_upload','No async event outside its phase')
                weight_events.append(row);require(len(weight_events)<=4*252+2*252+1,'Finite weight event prefix')
                if kind=='all_original_weights_completed':
                    from runtime_weights import audit_upload
                    weight_result=audit_upload(weight_events,active['items'],prepared)
                    copy_index+=weight_result['copies'];host_bytes+=weight_result['host_bytes'];active=None
            elif kind in ('copy_enter','launch_enter'):
                require(active is None,'No overlapping blocking host operations');active=next(events,None)
                require(active is not None and kind==active['kind']+'_enter','Exact complete operation schedule')
                pending=dict(returned=False,parameter=False,receipt=None,enter_sequence=row['sequence'])
                if active['kind']=='launch':
                    require(waiting_input is None,'No unattached raw input before launch')
                    require(row['name']==active['name'] and row['index']==launch_index,'Exact launch sequence')
                else:
                    item=active['item'];require(row['index']==copy_index and row['direction']==active['direction'] and
                        row['symbol']==item['symbol'] and row['rectangle']==box(item) and row['dtype']==item['dtype'] and
                        row['label']==active['label'],'Exact original bound copy sequence')
                    size=item['width']*item['height']*item['count']*4
                    require(row['host_bytes']==size,'Exact actual host-slot bytes');host_bytes+=size
                    if 'prepared' in item:
                        require(waiting_input is None,'No unattached raw input before parameter copy')
                        _,expected=prepared.transfer(item);require(row['prepared']==expected,'Original prepared slice pin')
                        pending['expected']=expected
                    else:
                        require(row['prepared'] is None,'No unbound original parameter substitution')
                        if active['direction']=='h2d':
                            require(waiting_input is not None and waiting_input['path']=='raw/'+active['label']+'.bin',
                                'Original input durable immediately before H2D begins')
                            pending['raw']=waiting_input;waiting_input=None
                        else:require(waiting_input is None,'No unattached raw input before D2H')
            elif kind=='copy_returned':
                require(active is not None and active['kind']=='copy' and row['index']==copy_index and not pending['returned'] and
                        row['enter_sequence']==pending['enter_sequence'],'One actual copy return paired with its call')
                pending['returned']=True
            elif kind=='parameter_verified':
                require(active is not None and active['kind']=='copy' and active['direction']=='d2h' and pending['returned'] and
                        'expected' in pending and not pending['parameter'] and row['index']==copy_index,'One exact retention proof after D2H')
                receipt=row['receipt'];require(receipt['exact_match'] and receipt['prepared']==pending['expected'] and
                    receipt['actual_native_sha256']==pending['expected']['native_sha256'],'Resident readback equals original native slice')
                pending['parameter']=True;pending['receipt']=receipt;parameter_reads+=1
            elif kind=='raw_durable':
                receipt=row['receipt'];name=Path(receipt['path']).name
                require(name not in raw_seen and receipt==captured['raw_files'][name],'Unique exact durable raw receipt')
                raw_seen.add(name)
                if active is None:
                    require(waiting_input is None,'Only one pending original input');waiting_input=receipt
                else:
                    require(active['kind']=='copy' and active['direction']=='d2h' and pending['returned'] and
                        'expected' not in pending and 'raw' not in pending,'One raw D2H saved after successful return')
                    pending['raw']=receipt
            elif kind=='copy_exit':
                require(active is not None and active['kind']=='copy' and pending['returned'] and row['index']==copy_index,'One complete copy exit')
                receipt=row['receipt']
                if 'expected' in pending:
                    if active['direction']=='d2h':require(pending['parameter'] and receipt==pending['receipt'],'Matching final parameter proof')
                    else:require(receipt['prepared']==pending['expected'] and receipt['native_sha256']==pending['expected']['native_sha256'],'Original H2D native identity')
                else:
                    name=active['label']+'.bin';item=active['item']
                    require(name in raw_seen and receipt==captured['raw_files'][name] and receipt==pending.get('raw'),'Raw durable before copy exit')
                    require(receipt['path']=='raw/'+name and receipt['symbol']==item['symbol'] and receipt['rectangle']==box(item) and
                        receipt['dtype']==item['dtype'] and receipt['bytes']==item['width']*item['height']*item['count']*(2 if item['dtype']=='u16' else 4),
                        'Every raw copy has its independently scheduled exact geometry/type/size')
                copy_index+=1;active=pending=None
            elif kind=='launch_exit':
                require(active is not None and active['kind']=='launch' and row['name']==active['name'] and row['index']==launch_index and
                        row['enter_sequence']==pending['enter_sequence'] and row['nonblock'] is False and
                        row['device_completion_proved'] is False,'Historical launch mode; no return-as-completion claim')
                launch_index+=1;active=pending=None
    require(active is None and waiting_input is None and next(events,None) is None and last=='stop_exit','Complete schedule followed by normal stop')
    require(copy_index==captured['copies'] and launch_index==8 and sequence==captured['journal_events'],'Exact journal event and operation totals')
    require(host_bytes==captured['host_bytes']<=copies['budget']['max_host_bytes'],'Complete exact physical host-byte total')
    require(parameter_reads==2*len(copies['uploads']) and raw_seen==captured['raw_files'].keys(),'Both complete retention passes and every raw receipt')
    require(weight_result is not None,'Complete independently checked asynchronous weight phase')
    for key in ('copies','host_bytes','batches','maximum_tasks','maximum_retained_bytes'):
        require(captured['weight_upload'][key]==weight_result[key],'Capture weight ownership totals match independent journal')
    return dict(events=sequence,copies=copy_index,launches=launch_index,parameter_readbacks=parameter_reads,raw_files=len(raw_seen),weight_upload=weight_result)

def control(evidence,plan,transport,copies):
    import numpy as np
    gate=Boundary(plan,transport);sequence=0
    def check_states(serial,generation,position,phase):
        for i,item in enumerate(copies['state_copies']):
            state(item,evidence.raw(f's{serial}-{phase}-state-{i:04}.bin'),plan,serial,generation,position,phase)
    check_states(0,1,0,'initialized');check_states(2,2,0,'reset')
    for i,item in enumerate(copies['persistent_copies']):
        initial=evidence.raw(f'initialized-cache-{i:04}.bin');require(not np.any(initial),'All initial physical KV bytes zero')
        retained=evidence.raw(f'reset-cache-{i:04}.bin').reshape(-1)
        previous=evidence.coordinate(2,'cache',item['x'],item['y'])
        require(retained.tobytes()==previous.tobytes(),'Reset preserves entire actual physical KV state before replay')
    for step in plan['schedule']:
        serial,generation,position=(step[k] for k in ('serial','reset_generation','position'))
        check_states(serial,generation,position,'prepared');check_states(serial,generation,position,'complete')
        observed=evidence.capture['observations'][serial-1]
        for attempt in range(observed['attempts']):
            result=observer(evidence.raw(f's{serial}-observer-{attempt:03}.bin'),serial,sequence);sequence=result['sequence']
            require(result['complete']==(attempt==observed['attempts']-1),'Observer stops on first full coherent completion')
        require(result['origin_state']==observed['origin_state'],'Capture observer matches raw')
        for i,item in enumerate(copies['transport_copies']):
            first=evidence.raw(f's{serial}-transport-0-{i:04}.bin');second=evidence.raw(f's{serial}-transport-1-{i:04}.bin')
            gate.transport(item,first,serial,generation-1);gate.transport(item,second,serial,generation-1)
            require(first.tobytes()==second.tobytes(),'Two complete physical transport fences equal')
        metadata={}
        for i,item in enumerate(copies['diagnostic_copies']):
            value=evidence.raw(f's{serial}-diagnostic-{i:04}.bin');gate.lifecycle(item,value,serial)
            if item['symbol'] in ('observer','qk_archive_status','qk_archive'):
                from runtime_boundary import rows
                for xy,row in rows(item,value):metadata[item['symbol'],xy]=row.copy()
        require(gate.final_metadata(metadata,serial)==evidence.capture['head_metadata'][serial-1],'Every head metadata and archive gate independently reproduced')
    require(evidence.coordinate(1,'hidden',*plan['post_norm']).tobytes()==evidence.coordinate(3,'hidden',*plan['post_norm']).tobytes(),'Exact original Layer3 output reset replay')
    return dict(all_states=True,initial_full_KV_zero=True,reset_retains_all_KV_bytes=True,all_transport=True,all_lifecycle=True,all_head_archives=True,hidden_reset_replay=True)

def main():
    import numpy as np
    started=time.monotonic();inputs,profiles,_=verify('audit')
    complete=json.loads((ROOT/'COMPLETE.json').read_bytes());captured=json.loads((ROOT/'capture.json').read_bytes())
    require(complete['all_owned_jobs_released'] and captured['normal_stop'] and captured['completed']==[1,2,3],
            'Separate released successful physical owner before numerical audit')
    if (ROOT/'numeric-audit.json').exists():raise ValueError('Offline audit already attempted')
    plan,transport,host,typed=[json.loads((ROOT/n).read_bytes()) for n in ('plan.json','transport-map.json','host-map.json','EXPORTS.json')]
    copies=build(plan,transport,host,typed);prepared=Prepared(inputs['prepared_root'],inputs['prepared_pin'])
    checked_journal=journal(ROOT,captured,copies,prepared);evidence=Evidence(ROOT,captured,copies)
    require({p.name for p in (ROOT/'raw').iterdir()}==captured['raw_files'].keys(),'No missing or unexpected raw files')
    for name in captured['raw_files']:evidence.raw(name)
    checked_control=control(evidence,plan,transport,copies)
    nominal,nominal_pin=prepared.array('reference-output.npy');nominal=nominal.copy()
    pin={k:nominal_pin[k] for k in ('bytes','sha256')}
    from qualified_math import load
    result=audit(evidence,prepared,plan,copies,load(),nominal)
    prepared.verify_unchanged();verify('audit')
    result.update(journal=checked_journal,control=checked_control,nominal_reference=pin,
        total_seconds=time.monotonic()-started,SDK_imported=False,controller_acceptance=False)
    atomic_json(ROOT/'numeric-audit.json',result)

if __name__=='__main__':main()
