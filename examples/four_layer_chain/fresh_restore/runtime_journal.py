"""Independent exact operation ledger; adapted qualified completion checks."""
from pathlib import Path
import hashlib,json
from runtime_boundary import require
from runtime_evidence import box
from runtime_schedule import scheduled

def journal(root,captured,host,prepared):
    copies=host['copies'];budget=host['baseline_budget' if captured['mode']=='baseline' else 'restored_budget']
    sequence=0;seconds=-1.;active=None;pending=None;copy_index=launch_index=0;host_bytes=0
    raw_seen=set();parameter_reads=0;events=iter(scheduled(host,captured['observations'],captured['mode']));last=None;waiting_input=None
    weight_events=[];weight_result=None
    weight_kinds={'weight_batch_enter','weight_submit_enter','weight_submit_returned','weight_wait_enter',
                  'weight_task_completed','weight_batch_exit','all_original_weights_completed'}
    path=Path(root)/'journal.jsonl';require(path.stat().st_size==captured['journal_bytes']<=128<<20,'Complete bounded journal size')
    durability=captured.get('journal_durability')
    require(isinstance(durability,dict) and durability.get('final') is True and
            durability.get('events')==captured['journal_events'] and durability.get('bytes')==captured['journal_bytes'] and
            durability.get('maximum_pending_events')==128 and durability.get('reason')=='final_close',
            'Complete explicitly synced final journal is required')
    require(json.loads((Path(root)/'journal-durability.json').read_bytes())==durability,
            'Capture binds the final durable journal receipt')
    digest=hashlib.sha256()
    with path.open('rb') as f:
        for raw in f:
            digest.update(raw)
            require(len(raw)<=65536,'Bounded single journal record');row=json.loads(raw)
            require(row['sequence']==sequence and row['seconds']>=seconds,'Contiguous monotonic journal')
            sequence+=1;seconds=row['seconds'];kind=row['kind'];last=kind
            if kind.startswith('weight_') or kind=='all_original_weights_completed':
                require(kind in weight_kinds,'No asynchronous failure or unknown weight event in accepted capture')
                if not weight_events:
                    require(active is None and waiting_input is None and weight_result is None,'Exactly one distinct weight upload phase')
                    active=next(events,None);require(active is not None and active['kind']=='weight_upload','Scheduled original weight phase')
                require(active is not None and active['kind']=='weight_upload','No async event outside its phase')
                weight_events.append(row);require(len(weight_events)<=4*822+2*822+1,'Finite weight event prefix')
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
    require(digest.hexdigest()==durability['sha256'],'Independent full durable journal digest')
    require(active is None and waiting_input is None and next(events,None) is None and last=='stop_exit','Complete schedule followed by normal stop')
    require(copy_index==captured['copies'] and launch_index==budget['max_launches'] and sequence==captured['journal_events'],'Exact journal event and operation totals')
    require(host_bytes==captured['host_bytes']<=budget['max_host_bytes'],'Complete exact physical host-byte total')
    require(parameter_reads==2*len(copies['uploads']) and raw_seen==captured['raw_files'].keys(),'Both complete retention passes and every raw receipt')
    require(weight_result is not None,'Complete independently checked asynchronous weight phase')
    for key in ('copies','host_bytes','batches','maximum_tasks','maximum_retained_bytes'):
        require(captured['weight_upload'][key]==weight_result[key],'Capture weight ownership totals match independent journal')
    return dict(events=sequence,copies=copy_index,launches=launch_index,parameter_readbacks=parameter_reads,raw_files=len(raw_seen),weight_upload=weight_result)
