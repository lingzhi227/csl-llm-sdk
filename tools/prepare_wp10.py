"""Freeze one D256 cache8 attention candidate from qualified CPU fixtures."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.attention_numerics import POLICY
from qwen38.attention_protocol import LEDGER,validate_ledger


def prepare(root,run,image,reference):
    run=run.resolve();image=image.resolve();reference=reference.resolve()
    if not image.is_file():raise ValueError('Reuse an existing qualified SDK image')
    result=json.loads((reference/'result.json').read_text())
    if result.get('passed') is not True or len(result.get('checks',[]))!=10:raise ValueError('CPU reference must pass ten checks')
    required={'fixtures.json','oracles.json','numerical-policy.json'}
    if set(result['output_hashes'])!=required:raise ValueError('Incomplete reference hashes')
    for name,digest in result['output_hashes'].items():
        if hashlib.sha256((reference/name).read_bytes()).hexdigest()!=digest:raise ValueError('Reference mismatch: '+name)
    if json.loads((reference/'numerical-policy.json').read_text())!=POLICY:raise ValueError('Numerical policy changed')
    validate_ledger()
    run.mkdir(parents=True,exist_ok=False);(run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for n in ('pe.csl','layout.csl','driver.py','compile_entry.py'):shutil.copyfile(root/'examples/wp10'/n,run/n)
    shutil.copyfile(root/'csl/kernels/attention256_bf16.csl',run/'kernel.csl')
    shutil.copyfile(root/'csl/kernels/exp_bounded_f32.csl',run/'exp_kernel.csl')
    for n in ('__init__.py','resources.py','locking.py','elf_inventory.py','rms_numerics.py','gated_numerics.py','interval_numerics.py','attention_numerics.py','attention_protocol.py'):
        shutil.copyfile(root/'core/qwen38'/n,run/'core/qwen38'/n)
    for n in ('guarded_run.py','sdk_container.py'):shutil.copyfile(root/'tools'/n,run/'tools'/n)
    for n in required:shutil.copyfile(reference/n,run/n)
    (run/'reference-result.json').write_text(json.dumps(result,indent=2)+'\n')
    from qwen38.attention_protocol import budget
    (run/'readback-plan.json').write_text(json.dumps({**budget(),'planning_seconds':150,'hard_seconds':180},indent=2)+'\n')
    (run/'resource-ledger.json').write_text(json.dumps(LEDGER,indent=2)+'\n')
    (run/'sram-policy.json').write_text(json.dumps({'ordinary_address_ceiling':49152,'stack_allowance_bytes':4096,
        'scope':'actual application section end plus stack allowance gate'},indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    spec={'required_profile':'sdk','planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']},
        {'name':'simulate','seconds':180,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':files,'scope':'synthetic D256 cache8 attention and persistent history'},indent=2)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','image','reference'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(prepare(Path(__file__).resolve().parents[1],args.run,args.image,args.reference))
