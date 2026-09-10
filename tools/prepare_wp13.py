"""Freeze a reviewed diagnostic or single-slab preparation; never execute SDK."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
from plan_wp13_device import plan, symbolic_sequence


def prepare(root,run,image,reference,slab=None):
    run=run.resolve();reference=reference.resolve();image=image.resolve()
    if slab is not None and (type(slab) is not int or not 0<=slab<8):
        raise ValueError('Selected global slab must be0..7')
    pes,cases,first=(4,[0],0) if slab is None else (1,[0,1,2,3],slab)
    if not image.is_file():raise ValueError('Missing prequalified SDK image')
    result=json.loads((reference/'result.json').read_text())
    if not result['passed']:raise ValueError('Reference did not pass')
    ref=json.loads((reference/'reference-plan.json').read_text())
    if hashlib.sha256((reference/'observations.npz').read_bytes()).hexdigest()!=result['observations_sha256']:
        raise ValueError('Reference observations identity mismatch')
    original_manifest=json.loads((reference/'source-manifest.json').read_text())
    for name,digest in original_manifest.items():
        if hashlib.sha256((reference/name).read_bytes()).hexdigest()!=digest:
            raise ValueError('Reference frozen source mismatch')
    weights=Path(ref['weights_directory'])
    for name,identity in ref['weight_files'].items():
        if hashlib.sha256((weights/name).read_bytes()).hexdigest()!=identity['sha256']:
            raise ValueError('Original weight identity mismatch')
    run.mkdir(parents=True,exist_ok=False)
    (run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir();(run/'weights').mkdir()
    for name in ('driver.py','compile_entry.py','pe.csl','layout.csl'):
        shutil.copyfile(root/'examples/wp13'/name,run/name)
    shutil.copyfile(root/'csl/kernels/persistent_gemv_bf16_f32.csl',run/'kernel.csl')
    for name in ('__init__.py','resources.py','locking.py','elf_inventory.py'):
        shutil.copyfile(root/'core/qwen38'/name,run/'core/qwen38'/name)
    for name in ('guarded_run.py','sdk_container.py','prepare_wp13.py','plan_wp13_device.py'):
        shutil.copyfile(root/'tools'/name,run/'tools'/name)
    for name in ('reference-plan.json','hidden.bf16','observations.npz'):
        shutil.copyfile(reference/name,run/name)
    shutil.copyfile(reference/'result.json',run/'reference-result.json')
    shutil.copyfile(weights/'download.json',run/'weight-receipt.json')
    for name in ref['weight_files']:
        shutil.copyfile(weights/name,run/'weights'/name)
    for path in run.rglob('*.py'):ast.parse(path.read_text())
    for path in run.glob('*.csl'):
        source=re.sub(r'//[^\n]*|"(?:\\.|[^"\\])*"','',path.read_text());stack=[]
        for char in source:
            if char in '([{':stack.append(char)
            elif char in ')]}':
                if not stack or stack.pop()!={')':'(',']':'[','}':'{'}[char]:
                    raise ValueError('CSL delimiter mismatch: '+path.name)
        if stack:raise ValueError('Unclosed CSL delimiters: '+path.name)
    traffic=plan(pes,len(cases))
    expected=(612,10889472,5494096,49) if slab is None else (603,10714296,5405728,196)
    assert tuple(traffic[k] for k in ('copies','host_bytes','payload_bytes','launches'))==expected
    sequence=symbolic_sequence(pes,len(cases))
    copies=[s for s in sequence if s['kind']=='copy']
    assert (len(copies),sum(s['host_bytes'] for s in copies),sum(s['native_bytes'] for s in copies),
            sum(s['kind']=='launch' for s in sequence))==expected
    assert max(s['host_bytes'] for s in copies)==57352
    (run/'physical-sequence.json').write_text(json.dumps(sequence,indent=2)+'\n')
    raw_hidden=(reference/'hidden.bf16').read_bytes()
    assert len(raw_hidden)==4*5120*2
    input_hashes={ref['cases'][i]:hashlib.sha256(raw_hidden[i*10240:(i+1)*10240]).hexdigest() for i in cases}
    profile={'scope':'wp13-four-pe-full5120-dense-diagnostic' if slab is None else 'wp13-single-pe-four-case-slab',
        'pes':pes,'first_slab':first,'columns':5120,'case_indices':cases,'traffic':traffic,
        'input_sha256':input_hashes,'global_row_range':[first*128,(first+pes)*128],
        'estimate_seconds':250 if slab is None else 240,
        'estimate_basis':'Equal2621440FMA work: accepted WP03 205.785s, four-PE diagnostic233.915s. Single-PE/four-case batch has one final whole-weight read instead of four,196launches and repeated generations.240s is a conservative planning hypothesis, not a hardware speedup claim.' if slab is not None else 'Historical diagnostic preparation estimate250; measured233.915s.',
        'hard_seconds':300,'sdk_launch_permission':'External controller review required; freeze creation does not grant permission.',
        'observations_sha256':result['observations_sha256'],
        'prefix_policy':'Independent original row-major FP64 products/sum at n112 and5040, error inflated(gamma_n_FP32*abs_sum+n*2^-53*abs_sum+n*2^-149); final n5120. Cast gate uses unchanged full CPU source interval and exact actual FP32 RNE bits.',
        'weight_tail_poison_bf16':0x3e80,'input_tail_poison_f32':7.0,
        'ownership_scope':'Finalized output retained until explicit host release; begin refusal and future consumer transport not exercised.',
        'raw_weight_copy':'10MiB workstation-local frozen run copy; no additional model network payload or Mac copy.'}
    (run/'profile.json').write_text(json.dumps(profile,indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    spec={'required_profile':'sdk','planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3',
         f'--fabric-dims={pes+7},3','--fabric-offsets=4,1',f'--params=pe_count:{pes},first_slab:{first}',
         '-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']},
        {'name':'simulate','seconds':300,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':files,'scope':profile['scope']},indent=2)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--image',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--slab',type=int,help='Prepare exactly one full four-case global slab0..7; omission retains the diagnostic profile.')
    args=parser.parse_args()
    print(prepare(Path(__file__).resolve().parents[1],args.run,args.image,args.reference,args.slab))
