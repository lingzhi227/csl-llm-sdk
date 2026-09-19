"""Inspect actual connected fixture placement and one aligned weight bank."""
from pathlib import Path
import hashlib,json,re
from cerebras.elf.cself import ELFMemory
from hw00.store import atomic_json
from elf_symbols import symbols

ROOT=Path(__file__).resolve().parent
width,height=json.loads((ROOT/'layout.json').read_bytes())['application']
if (width,height)!=(750,45):raise ValueError('Exact complete layer3 geometry required')
plan=json.loads((ROOT/'layer3-plan.json').read_bytes())
matrix_coordinates={(s['x']+ordinal+4,s['y']+1) for s in plan['stripes'] for ordinal in range(s['columns'])}
covered=set();rows=[]
attention_coordinates={(x+4,1) for x in range(16,40)};checked_attention=set()
head_sinks={(752,y+1) for y in range(1,25)};checked_sinks=set();global_sink=(752,1)
sram=json.loads((ROOT/'sram.json').read_bytes())
trace_specs=json.loads((ROOT/'fifo-trace-contract.json').read_bytes())['sources']
trace_sources={(a['source'][0]+4,a['source'][1]+1):a for a in trace_specs}
trace_sinks={(a['sink'][0]+4,a['sink'][1]+1):a for a in trace_specs}
zero_sinks={(753,y+1) for y in range(25)}-set(trace_sinks)
checked_trace_sources=set();checked_trace_sinks=set();checked_zero=set();checked_slots=set()
read_slots={(x+4,y+1) for x in (748,749) for y in range(25)}
def select(records,name):
 found={}
 for v in records:
  canonical=v['name'].rsplit('$$',1)[-1] if v['name'].startswith('$$csl_base_address$$') else v['name']
  if canonical==name and v['bytes']>0:found[v['address'],v['bytes']]=v
 return list(found.values())
def packet_array(records,name,extent):
 found={}
 for value in records:
  canonical=value['name'].rsplit('$$',1)[-1] if value['name'].startswith('$$csl_base_address$$') else value['name']
  if (canonical==name or (extent==4 and canonical==name+'.0')) and value['bytes']>0:
   found[value['address'],value['bytes']]=value
 if len(found)!=1:raise ValueError('One real packet array: '+name)
 value=next(iter(found.values()))
 if value['bytes']!=extent or value['address']%4:raise ValueError('Packet array extent/alignment: '+name)
 return value
def probe_storage(records,required):
 whole=select(records,'probe_record')
 if whole:
  if len(whole)!=1 or whole[0]['bytes']!=(68 if required else 4) or whole[0]['address']%4:raise ValueError('Bounded aligned probe record body')
  return whole
 cells={}
 for v in records:
  name=v['name'].rsplit('$$',1)[-1] if v['name'].startswith('$$csl_base_address$$') else v['name']
  m=re.fullmatch(r'probe_record\.(\d+)',name)
  if m and v['bytes']>0:
   i=int(m.group(1));row=dict(v,name=name)
   if i in cells and (cells[i]['address'],cells[i]['bytes'])!=(v['address'],v['bytes']):raise ValueError('Conflicting scalarized probe field')
   cells[i]=row
 if not required:
  if cells:raise ValueError('Unexpected scalarized head record outside head')
  return []
 if set(cells)!=set(range(17)):raise ValueError('Every scalarized probe field0..16 required')
 start=cells[0]['address']
 if start%4 or any(v['bytes']!=4 or v['address']!=start+4*i for i,v in cells.items()):raise ValueError('Actual contiguous scalarized17-word record')
 return [cells[i] for i in range(17)]
for p in sorted((ROOT/'compiled/out/bin').glob('*.elf')):
    raw=p.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=sram['application_files'][p.name]['sha256']:raise ValueError('Actual ELF identity')
    reader=ELFMemory(p)
    if tuple(reader.get_fabric_dimensions())!=(762,1172):raise ValueError('Hardware fabric mismatch')
    raw_count=0;unique=set()
    for r in reader.iter_rectangles():
        raw_count+=1
        if raw_count>width*height*32:raise ValueError('Raw ELF metadata bound')
        unique.add(tuple(r))
    rectangles=sorted(unique);coordinates=set()
    if not 0<len(rectangles)<=width*height:raise ValueError('Bounded nonempty actual placement')
    for x,y,w,h in rectangles:
        if w<1 or h<1 or x<4 or y<1 or x+w>width+4 or y+h>height+1:raise ValueError('Rectangle outside full MLP')
        for xx in range(x,x+w):
            for yy in range(y,y+h):
                coordinates.add((xx,yy))
    if coordinates & covered:raise ValueError('Overlapping placement between distinct ELFs')
    covered.update(coordinates)
    matrix=bool(coordinates & matrix_coordinates)
    if matrix and not coordinates<=matrix_coordinates:raise ValueError('Mixed matrix/nonmatrix program mapping')
    banks=[s for s in symbols(p.read_bytes()) if 'weight_storage' in s['name'] and s['bytes']==24576]
    if matrix:
        if len(banks)!=1 or banks[0]['address']%4:raise ValueError('One actual aligned24576-byte weight bank required')
    elif banks:raise ValueError('Unexpected matrix bank outside matrix placement')
    records=symbols(raw)
    snapshot=select(records,'observer.snapshot');head_snapshot=select(records,'head_observer.snapshot')
    side_buffer=select(records,'head_observer.buffer')
    is_global=global_sink in coordinates;is_side=bool(coordinates & head_sinks)
    is_head=bool(coordinates & attention_coordinates)
    probe=probe_storage(records,is_head)
    if is_global:
        if coordinates!={global_sink} or len(snapshot)!=1 or snapshot[0]['bytes']!=640 or snapshot[0]['address']%4:raise ValueError('Singleton global observer snapshot')
    elif snapshot and any(v['bytes']!=4 for v in snapshot):raise ValueError('Unexpected global snapshot outside sink')
    if is_side:
        if not coordinates<=head_sinks or len(head_snapshot)!=1 or head_snapshot[0]['bytes']!=640 or head_snapshot[0]['address']%4:raise ValueError('All head sinks have real aligned160-word snapshots')
        checked_sinks.update(coordinates)
    elif head_snapshot and any(v['bytes']!=4 for v in head_snapshot):raise ValueError('Unexpected head snapshot outside idle sinks')
    if is_side or is_head:
        if len(side_buffer)!=1 or side_buffer[0]['bytes']!=128 or side_buffer[0]['address']%4:raise ValueError('One owned aligned128-byte sideband buffer')
    elif side_buffer:raise ValueError('Unexpected sideband buffer')
    records=symbols(raw)
    packet_header=packet_array(records,'packets.header_buffer',4)
    packet_payload=packet_array(records,'packets.receive_buffer',124)
    if max(packet_header['address'],packet_payload['address'])<min(packet_header['address']+4,packet_payload['address']+124):raise ValueError('Independent header and payload arrays')
    diagnostics=select(records,'packets.stats')
    if diagnostics:raise ValueError('Fixture diagnostic stats retained in default production code')
    cache=select(records,'attn.cache');rms=select(records,'attn.rms_stages');workspace=select(records,'attn.stages')
    if coordinates & attention_coordinates:
        if not coordinates<=attention_coordinates:raise ValueError('Mixed attention placement')
        for values,extent,label in [(cache,8192,'cache'),(workspace,5120,'attention workspace')]:
            if len(values)!=1 or values[0]['bytes']!=extent or values[0]['address']%4:raise ValueError('Original attention '+label+' extent/alignment')
        if rms and (len(rms)!=1 or rms[0]['bytes']!=4):raise ValueError('Separate QK diagnostic record must be removed by selected lifetime alias')
        ca=cache[0];ws=workspace[0]
        if max(ca['address'],ws['address'])<min(ca['address']+8192,ws['address']+5120):raise ValueError('Original KV cache must not alias attention workspace')
        checked_attention.update(coordinates)
    elif cache or rms or workspace:raise ValueError('Unexpected attention allocation outside heads')
    observer_buffer=select(records,'observer.buffer')
    if (4,1) in coordinates:
        if coordinates!={(4,1)} or len(observer_buffer)!=1 or observer_buffer[0]['bytes']!=128:raise ValueError('Original global observer origin buffer')
    elif observer_buffer and not is_global:raise ValueError('Unexpected observer source')
    trace_arrays={}
    trace_arrays['trace_status']=packet_array(records,'trace_status',16)
    source_coords=coordinates & set(trace_sources)
    if source_coords:
        if not coordinates<=set(trace_sources):raise ValueError('Mixed traced/nontraced source mapping')
        sizes={trace_sources[c]['capacity']*4 for c in source_coords}
        if len(sizes)!=1:raise ValueError('Mixed FIFO source capacities')
        trace_arrays['fifo_trace.storage']=packet_array(records,'fifo_trace.storage',next(iter(sizes)))
        checked_trace_sources.update(source_coords)
        if is_head:
            for name,size in {'attn.input':2048,'attn.qk_input':1024,'attn.qk_weights':1024,
                    'attn.normalized':1024,'attn.trig':512,'attn.rotary_stages':1536,
                    'attn.casts':1536}.items():trace_arrays[name]=packet_array(records,name,size)
        elif matrix:
            for name,size in {'lane.kernel.expanded':512,'lane.active_input':384,
                    'lane.partial':512,'lane.result':512}.items():trace_arrays[name]=packet_array(records,name,size)
        else:raise ValueError('Trace source must be original root or head')
    elif select(records,'fifo_trace.storage'):raise ValueError('Unexpected FIFO outside selected sources')
    sink_coords=coordinates & set(trace_sinks)
    if sink_coords:
        if len(coordinates)!=1 or not coordinates<=set(trace_sinks):raise ValueError('Each identified trace sink must be singleton')
        trace_arrays['trace_sink.snapshot']=packet_array(records,'trace_sink.snapshot',640)
        trace_arrays['trace_sink.incoming']=packet_array(records,'trace_sink.incoming',4)
        checked_trace_sinks.update(sink_coords);checked_slots.update(sink_coords)
    elif select(records,'trace_sink.snapshot'):raise ValueError('Unexpected active trace sink')
    zero_coords=coordinates & zero_sinks
    if zero_coords:
        if not coordinates<=zero_sinks:raise ValueError('Mixed zero trace backing')
        trace_arrays['idle_trace']=packet_array(records,'idle_trace',640)
        checked_zero.update(zero_coords);checked_slots.update(zero_coords)
    elif select(records,'idle_trace'):raise ValueError('Unexpected inactive trace backing')
    if is_global:checked_slots.update(coordinates)
    if is_side:checked_slots.update(coordinates)
    live=[packet_header,packet_payload,*trace_arrays.values(),*banks,*cache,*workspace,
        *observer_buffer,*side_buffer]
    if is_global:live+=snapshot
    if is_side:live+=head_snapshot
    for i,a in enumerate(live):
        for z in live[i+1:]:
            if max(a['address'],z['address'])<min(a['address']+a['bytes'],z['address']+z['bytes']):
                raise ValueError('Actual live original/trace arrays overlap: '+a['name']+' '+z['name'])
    rows.append(dict(trace_arrays=trace_arrays,packet_header_array=packet_header,packet_payload_array=packet_payload,packet_diagnostics=diagnostics,attention_cache=cache,qk_rms_record=rms,attention_workspace=workspace,observer_source_buffer=observer_buffer,observer_snapshot=snapshot,head_snapshot=head_snapshot,head_source_buffer=side_buffer,head_event_record=probe,file=p.name,rectangles=rectangles,raw_rectangle_count=raw_count,weight_banks=banks))
if len(covered)!=width*height:raise ValueError('Full rectangle not completely covered')
if checked_attention!=attention_coordinates:raise ValueError('All24 original attention allocations required')
if checked_sinks!=head_sinks:raise ValueError('All24 idle head sinks required')
if checked_trace_sources!=set(trace_sources):raise ValueError('All six trace sources required')
if checked_trace_sinks!=set(trace_sinks) or checked_zero!=zero_sinks:raise ValueError('All 25 new idle trace locations required')
if checked_slots!=read_slots:raise ValueError('All 50 host-read locations need actual160-word backing')
if any(packets for row in rows for packets in row['packet_diagnostics']):raise ValueError('Packet diagnostics DCE')
atomic_json(ROOT/'placement.json',dict(passed=True,trace_sources=6,trace_sinks=6,inactive_trace_sinks=19,actual_read_locations=50,head_sink_count=24,global_sink=[748,0],head_sink_rectangle=[748,1,1,24],snapshot_rectangle=[748,0,2,25],snapshot_words=160,host_snapshot_bytes=32000,diagnostic_RMS_alias_bytes=4096,retained_QK_record_audit_valid=False,PEs=len(covered),matrix_PEs=len(matrix_coordinates),geometry=[762,1172],application=[width,height],files=rows,runtime_created=False,packed_alias_runtime_qualification_pending=True))
