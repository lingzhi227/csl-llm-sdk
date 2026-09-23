"""Complete finite original Layer3 host sequence for positions0,1,reset0.

The caller injects the admitted runtime IO, immutable prepared files and a
separate parent deadline monitor. This function neither allocates a backend
nor runs an arithmetic oracle. Every failure leaves its durable raw prefix.
"""
from pathlib import Path
import hashlib,json,os,time
from reference_inputs import SCHEDULE
from runtime_boundary import Boundary,observer,state,require

def atomic_json(path,value):
    path=Path(path);raw=(json.dumps(value,separators=(',',':'))+'\n').encode()
    require(len(raw)<=16<<20,'Bounded capture metadata')
    temporary=path.with_suffix(path.suffix+'.tmp')
    with temporary.open('wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path);fd=os.open(path.parent,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)

class Timing:
    """Publish hard deadlines BEFORE calls; the separate parent enforces them."""
    def __init__(self,root,profile,store):
        self.root=Path(root);self.profile=profile;self.store=store;self.started=time.monotonic();self.compute_deadline=None
    def before(self,kind,name,rectangle=None):
        now=time.monotonic()
        require(now<self.started+self.profile['capture_seconds'],'Capture wall deadline')
        if self.compute_deadline is not None:require(now<self.compute_deadline,'Compute/observer absolute deadline')
        atomic_json(self.root/'progress.json',dict(monotonic=now,operation=[kind,name],rectangle=rectangle,
            raw_bytes=self.store.raw_bytes,raw_files=len(self.store.files)))
    def begin_compute(self,serial):
        require(self.compute_deadline is None,'No outstanding serial')
        started=time.monotonic();self.compute_deadline=started+self.profile['compute_observer_seconds']
        atomic_json(self.root/'monitor-deadline.json',dict(active=True,kind='compute_observer',serial=serial,
            started=started,deadline=self.compute_deadline))
    def observed(self):
        require(self.compute_deadline is not None and time.monotonic()<self.compute_deadline,'Timely durable observer')
        self.compute_deadline=None
        # Refresh progress before retiring the long window so the parent
        # never applies15s to an already completed long observer RPC.
        self.before('boundary','observer-complete')
        atomic_json(self.root/'monitor-deadline.json',dict(active=False,reason='complete_origin_observer_durable'))

def capture(io,prepared,root,plan,transport,copies,profile,timing):
    import numpy as np
    root=Path(root);gate=Boundary(plan,transport);completed=[];observations=[];outputs=[]
    require(plan['schedule']==SCHEDULE,'Exact original causal schedule')
    runtime_started=time.monotonic();previous_sequence=0
    reset_signatures={};replay_count=0;zero_checks=0;previous_cache={};metadata_reports=[]
    def progress(phase,serial=0):
        atomic_json(root/'capture-progress.json',dict(phase=phase,serial=serial,completed=completed,
            copies=io.copies,launches=io.launches,host_bytes=io.host_bytes,
            raw_files=len(io.store.files),raw_bytes=io.store.raw_bytes))
        io.store.event('phase',phase=phase,serial=serial,copies=io.copies)
    def parameters(direction,phase):
        weights_done=False
        for i,item in enumerate(copies['uploads']):
            if direction=='h2d' and item['symbol']=='weights':
                if not weights_done:
                    require(getattr(io,'weight_session',None) is None,'One strongly owned asynchronous weight session')
                    from runtime_weights import Session
                    io.weight_session=Session(io,prepared,root,[p for p in copies['uploads'] if p['symbol']=='weights'],profile['weight_upload'])
                    io.weight_result=io.weight_session.run();weights_done=True
                continue
            value,receipt=prepared.transfer(item)
            io.copy(direction,item,f'{phase}-{i:04}',value=value,prepared_receipt=receipt,retain_raw=False)
        progress(phase)
    def states(serial,generation,position,phase):
        for i,item in enumerate(copies['state_copies']):
            value,_=io.copy('d2h',item,f's{serial}-{phase}-state-{i:04}')
            state(item,value,plan,serial,generation,position,phase)
        io.store.event('all_PE_state_passed',phase=phase,serial=serial,generation=generation,position=position)
    def persistent_boundary(label):
        nonlocal zero_checks
        for i,item in enumerate(copies['persistent_copies']):
            value,receipt=io.copy('d2h',item,f'{label}-cache-{i:04}')
            if label=='initialized':
                require(not np.any(value),'Initially empty complete physical KV cache');zero_checks+=1
            else:require(receipt['sha256']==previous_cache[item['x'],item['y']],
                         'Original reset preserves every inactive and active KV byte until overwrite')
    def transport_fence(serial,resets):
        first={};points=0
        for snapshot in range(2):
            for i,item in enumerate(copies['transport_copies']):
                value,receipt=io.copy('d2h',item,f's{serial}-transport-{snapshot}-{i:04}')
                count=gate.transport(item,value,serial,resets)
                if snapshot==0:first[i]=receipt['sha256'];points+=count
                else:require(first[i]==receipt['sha256'],'Two complete identical transport snapshots')
            require(points==33640,'All application coordinates in each transport fence')
        io.store.event('transport_fence_passed',serial=serial,coordinates=points,snapshots=2,
            endpoint_counts=True,directed_edge_counts=True,leases=True,stable=True)
    try:
        parameters('h2d','upload-once');io.launch('initialize')
        parameters('d2h','initial-retention');states(0,1,0,'initialized');persistent_boundary('initialized')
        for step in SCHEDULE:
            serial,generation,position=(step[k] for k in ('serial','reset_generation','position'))
            if serial==3:
                require(completed==[1,2],'Reset only after both complete global fences')
                io.launch('reset');states(2,2,0,'reset');persistent_boundary('reset')
            io.copy('h2d',copies['hidden_input'],f's{serial}-original-input',value=prepared.input_row(step['input_row']))
            io.launch('prepare');states(serial,generation,position,'prepared');progress('prepared',serial)
            timing.begin_compute(serial);compute_started=time.monotonic();io.launch('compute')
            complete=None
            for attempt in range(copies['budget']['max_observer_polls_per_serial']):
                value,receipt=io.copy('d2h',copies['observer'],f's{serial}-observer-{attempt:03}')
                observation=observer(value,serial,previous_sequence);previous_sequence=observation['sequence']
                if observation['complete']:
                    complete=dict(observation,receipt=receipt,attempts=attempt+1,
                        host_submission_to_observed_seconds=time.monotonic()-compute_started,
                        interval_includes_RPC_and_observation=True,pure_device_time=False);break
                time.sleep(profile['observer_poll_interval_seconds'])
            require(complete is not None,'Finite observer poll budget exhausted')
            timing.observed();observations.append(complete)
            states(serial,generation,position,'complete');transport_fence(serial,generation-1)
            hidden_output=None;metadata={}
            for i,item in enumerate(copies['diagnostic_copies']):
                value,receipt=io.copy('d2h',item,f's{serial}-diagnostic-{i:04}')
                gate.lifecycle(item,value,serial)
                if item['symbol'] in ('observer','qk_archive_status','qk_archive'):
                    from runtime_boundary import rows
                    for xy,row in rows(item,value):metadata[item['symbol'],xy]=row.copy()
                if item['symbol']=='cache':
                    require(item['height']==item['width']==1,'One complete physical head cache per ROI')
                    previous_cache[item['x'],item['y']]=receipt['sha256']
                if item['dtype']=='f32':require(np.all(np.isfinite(value)),'Finite retained arithmetic operands/results')
                elif item['dtype']=='u16' and item['symbol']!='linear_receive_offsets':
                    require(not np.any((value&0x7f80)==0x7f80),'Finite retained BF16 operands/results')
                if item['symbol']=='hidden':
                    out=copies['hidden_output'];x,y=out['x'],out['y']
                    if item['x']<=x<item['x']+item['width'] and item['y']<=y<item['y']+item['height']:
                        native=value[y-item['y'],x-item['x']].copy()
                        hidden_output=dict(receipt=receipt,coordinate=[x,y],native_BF16_sha256=hashlib.sha256(native.tobytes()).hexdigest(),
                            words=5120,actual_device_output=True)
                # Retained attention scratch/cache suffixes follow the accepted
                # Layer3 reset contract. Only the full hidden output is the
                # whole-array replay boundary; the offline oracle checks all
                # active operands, old prefixes and untouched cache tails.
            metadata_reports.append(gate.final_metadata(metadata,serial))
            require(hidden_output is not None,'Actual Layer3 hidden output is retained')
            if serial==1:reset_signatures['hidden']=hidden_output['native_BF16_sha256']
            elif serial==3:
                require(hidden_output['native_BF16_sha256']==reset_signatures['hidden'],'Exact first-position hidden reset replay');replay_count=1
            outputs.append(dict(serial=serial,position=position,generation=generation,**hidden_output))
            completed.append(serial);progress('serial-complete',serial)
        parameters('d2h','terminal-retention');preparation=prepared.verify_unchanged()
        require(completed==[1,2,3] and replay_count==len(reset_signatures),'Complete causal and exact reset replay schedule')
        require(io.launches==8,'Exactly initialize +3prepare +3compute +reset')
        expected=copies['budget']['max_copies']-3*copies['budget']['max_observer_polls_per_serial']+sum(x['attempts'] for x in observations)
        require(io.copies==expected,'Every finite scheduled copy executed exactly once')
        result=dict(status='captured_pending_independent_numeric_audit',completed=completed,observations=observations,
            hidden_outputs=outputs,preparation=preparation,raw_files=io.store.files,raw_bytes=io.store.raw_bytes,
            copies=io.copies,host_bytes=io.host_bytes,launches=io.launches,journal_events=io.store.sequence,
            journal_bytes=io.store.journal_bytes,persistent_zero_checks=zero_checks,reset_replay_arrays=replay_count,
            capture_seconds=time.monotonic()-runtime_started,mathematical_audit_passed=False,
            full_layer_accepted=False,full_model=False,CPU_forward=False,per_position_parameter_H2D=False,
            weight_upload=io.weight_result,head_metadata=metadata_reports,reset_preserves_full_cache_before_replay=True)
        atomic_json(root/'capture.json',result);return result
    except BaseException as exc:
        atomic_json(root/'capture-failure.json',dict(error=type(exc).__name__,message=str(exc),completed=completed,
            copies=io.copies,launches=io.launches,raw_files=io.store.files,raw_bytes=io.store.raw_bytes))
        raise
