"""Compare exactly 1,000 fixed reference-selected BF16 hidden values.

No other actual neural values are numerically checked. Source/file identity
and hash checks protect the selected evidence; they are not operator tests.
The selection is sealed before any actual hidden payload is opened.
"""
from pathlib import Path
import hashlib,io,json,math,os,sys,time
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent))
import numpy as np
from audit_gate import checked,pin,verify_audit_owner
from runtime_gate import source
from runtime_pins import document
from runtime_evidence import RawEvidence,Evidence
from stage_contract import context_record,MODEL,REVISION

def encode(value):return (json.dumps(value,indent=2,allow_nan=False)+'\n').encode()

def seal(name,value):
    raw=encode(value);assert 0<len(raw)<=2<<20
    path=ROOT/name
    with path.open('xb') as f:f.write(raw);f.flush();os.fchmod(f.fileno(),0o444);os.fsync(f.fileno())
    fd=os.open(ROOT,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(path='comparison-1000-001/'+name,**pin(raw))

def value(bits):
    return float(np.array([int(bits)<<16],dtype='<u4').view('<f4')[0])

def classification(v):
    return 'nan' if math.isnan(v) else ('positive_infinity' if v==math.inf else 'negative_infinity' if v==-math.inf else 'finite')

def select(reference,count):
    # Endpoint-inclusive integer rounding is independent of floating rounding.
    fixed=[(i*5119+9)//19 for i in range(20)] if count==100 else [0,5119]
    decoded=(reference.astype('<u4')<<16).view('<f4')
    def rank(d):
        v=float(decoded[d])
        return (-1,0,d) if math.isnan(v) else (0,-abs(v),d)
    candidates=sorted(range(5120),key=rank)
    chosen=set(fixed)
    for d in candidates:
        if len(chosen)==count:break
        chosen.add(d)
    assert len(chosen)==count
    return sorted(chosen)

def main():
    started=time.monotonic();_,context=verify_audit_owner()
    inputs,_,_=source();scope=json.loads(checked(ROOT/'scope.json'))
    envelope=json.loads(checked(ROOT.parent/'audit-input.json'))
    capture_row=envelope['files']['capture']
    capture=document(ROOT.parent/capture_row['path'],capture_row,128<<10)
    assert capture['context']==context_record(context) and capture['normal_stop'] and capture['completed']==[1,2,3,4,5]
    assert inputs['schedule']['prompt_tokens']==capture['input_receipt']['token_ids']
    nominal=inputs['nominal'];reference_root=Path(nominal['root'])
    provenance=document(reference_root/nominal['provenance']['path'],nominal['provenance'],128<<10)
    assert provenance['model']==MODEL and provenance['revision']==REVISION
    assert provenance['independent_original_reference'] and provenance['model_weights_unmodified'] and provenance['reference_forward_complete']
    assert provenance['outputs']==nominal['outputs'] and provenance['positions']==nominal['positions']==list(range(5))
    assert provenance['token_ids']==nominal['token_ids']==inputs['schedule']['prompt_tokens']
    references={};reference_pins={};reference_stamps={};selection=[]
    for layer in range(20):
        entries=nominal['outputs'][str(layer)]
        assert [r['position'] for r in entries]==list(range(5))
        for entry in entries:
            name=entry['path'];assert Path(name).name==name
            p=reference_root/name
            if name not in references:
                raw=checked(p,entry['pin'],limit=2<<20)
                with np.load(io.BytesIO(raw),allow_pickle=False) as data:
                    assert data.files==[entry['array']]
                    array=data[entry['array']].copy()
                assert list(array.shape)==entry['shape']==[1,5,5120] and array.dtype==np.dtype('<u2')
                references[name]=array;reference_pins[name]=pin(raw)
                s=p.stat();reference_stamps[name]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode]
            assert reference_pins[name]==entry['pin'] and entry['tensor_width']==5120 and 0<=entry['row']<5
            row=references[name][0,entry['row']]
            count=100 if layer==19 else 6 if layer in (0,3,7,11,15) else 5
            for dimension in select(row,count):
                selection.append(dict(layer=layer,position=entry['position'],dimension=dimension,
                    reference_file=name,reference_row=entry['row'],reference_BF16_bits=int(row[dimension])))
    keys=[(v['layer'],v['position'],v['dimension']) for v in selection]
    assert len(selection)==len(set(keys))==1000
    assert sum(v['layer']==19 for v in selection)==500
    selection_pin=seal('selection.json',dict(schema=1,context=context_record(context),scope=scope,
        reference_pins=reference_pins,selection=selection,actual_payload_opened=False,
        selected_values=1000,selection_depends_on_actual=False))
    print(json.dumps(dict(phase='selection_sealed_before_actual_payloads',selected_values=1000,selection_pin=selection_pin)),flush=True)
    # Actual payload access begins only after durable immutable selection.
    host=document(inputs['host_plan_path'],inputs['host_plan_pin'],64<<20)
    lowered=document(inputs['lowered_path'],inputs['lowered_pin'],16<<20)
    assert host['application']==lowered['application']==capture['application']
    raw=RawEvidence(ROOT.parent,capture,context,lowered['application'])
    plans={family:json.loads(checked(ROOT.parent/('qualified_'+family)/'plan.json',limit=4<<20)) for family in ('linear','attention')}
    results=[];selected_payloads={};selected_stamps={}
    for layer in range(20):
        evidence=Evidence(raw,host,lowered,layer)
        family=evidence.region['family'];plan=plans[family]
        xy=plan['support']['post_norm'] if family=='linear' else plan['post_norm']
        for position in range(5):
            serial=position+1
            if layer==19:
                name=f's{serial}-stage-output.bin'
                actual=raw.checked(name,host['copies']['hidden_output']).reshape(5120)
            else:
                ordinal,item,ix,iy=evidence.owners('hidden',*xy)
                name=f's{serial}-diagnostic-{ordinal:04}.bin'
                actual=raw.checked(name,item)[iy,ix]
            assert actual.dtype==np.dtype('<u2') and actual.shape==(5120,)
            receipt=raw.receipts[name];selected_payloads[name]=receipt
            s=(ROOT.parent/receipt['path']).stat()
            selected_stamps[name]=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode]
            points=[v for v in selection if v['layer']==layer and v['position']==position]
            for point in points:
                got=int(actual[point['dimension']]);wanted=point['reference_BF16_bits']
                a,b=value(got),value(wanted);finite=math.isfinite(a) and math.isfinite(b)
                absolute=abs(a-b) if finite else None
                relative=(absolute/abs(b) if b!=0 else (0.0 if a==0 else None)) if finite else None
                results.append(dict(layer=layer,position=position,dimension=point['dimension'],
                    actual_BF16_bits=got,reference_BF16_bits=wanted,
                    actual_value=a if math.isfinite(a) else None,reference_value=b if math.isfinite(b) else None,
                    actual_class=classification(a),reference_class=classification(b),
                    BF16_exact_match=got==wanted,absolute_difference=absolute,relative_difference=relative,
                    relative_undefined=(not finite or (b==0 and a!=0))))
    assert [(v['layer'],v['position'],v['dimension']) for v in results]==keys
    for name,before in reference_stamps.items():
        s=(reference_root/name).stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode]==before
    for name,receipt in selected_payloads.items():
        p=ROOT.parent/receipt['path'];s=p.stat()
        assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode]==selected_stamps[name]
        assert s.st_size==receipt['bytes'] and not s.st_mode&0o222
    finite_abs=[r['absolute_difference'] for r in results if r['absolute_difference'] is not None]
    finite_rel=[r['relative_difference'] for r in results if r['relative_difference'] is not None]
    summary=dict(compared_values=1000,exact_BF16_matches=sum(r['BF16_exact_match'] for r in results),
        nonfinite_pairs=sum(r['actual_class']!='finite' or r['reference_class']!='finite' for r in results),
        maximum_finite_absolute_difference=max(finite_abs,default=None),
        mean_finite_absolute_difference=math.fsum(finite_abs)/len(finite_abs) if finite_abs else None,
        maximum_defined_relative_difference=max(finite_rel,default=None),
        relative_undefined_pairs=sum(r['relative_undefined'] for r in results))
    result=dict(status='comparison_1000_completed',context=context_record(context),scope=scope,summary=summary,
        selection_pin=selection_pin,selection_frozen_before_actual=True,selection_depends_on_actual=False,
        source_capture_pin={k:capture_row[k] for k in ('bytes','sha256')},
        reference_provenance_pin=nominal['provenance'],reference_pins=reference_pins,selected_payload_receipts=selected_payloads,results=results,
        additional_actual_numeric_comparisons=0,error_threshold=None,overall_numeric_pass_claimed=False,
        full_operator_coverage=False,full_control_numeric_recheck=False,full_raw_inventory_rechecked=False,
        full_journal_replayed=False,whole_model_error_bound=False,SDK_imported=False,CPU_forward=False,
        full_model=False,controller_acceptance=False,total_seconds=time.monotonic()-started)
    verify_audit_owner()
    result_pin=seal('comparison.json',result)
    print(json.dumps(dict(phase='comparison_complete',result_pin=result_pin,summary=summary,total_seconds=result['total_seconds'])),flush=True)

if __name__=='__main__':main()
