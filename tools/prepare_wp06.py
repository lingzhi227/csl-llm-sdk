"""Freeze one full128x128 recurrent candidate from qualified CPU fixtures."""
import argparse,hashlib,json,shutil,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.recurrent_numerics import POLICY
from qwen38.recurrent_protocol import LEDGER,validate_ledger


def prepare(root,run,image,reference):
    run=run.resolve();image=image.resolve();reference=reference.resolve()
    if not image.is_file():raise ValueError('Reuse an existing qualified SDK image')
    result=json.loads((reference/'result.json').read_text())
    if result.get('passed') is not True or len(result.get('checks',[]))!=8:raise ValueError('CPU reference must pass eight checks')
    required={'fixtures.json','oracles.json','numerical-policy.json'}
    if set(result['output_hashes'])!=required:raise ValueError('Incomplete reference hashes')
    for name,digest in result['output_hashes'].items():
        if hashlib.sha256((reference/name).read_bytes()).hexdigest()!=digest:raise ValueError('Reference mismatch: '+name)
    if json.loads((reference/'numerical-policy.json').read_text())!=POLICY:raise ValueError('Numerical policy changed')
    validate_ledger()
    run.mkdir(parents=True,exist_ok=False);(run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for n in ('pe.csl','layout.csl','driver.py','compile_entry.py'):shutil.copyfile(root/'examples/wp06'/n,run/n)
    shutil.copyfile(root/'csl/kernels/recurrent_head_f32.csl',run/'kernel.csl')
    for n in ('__init__.py','resources.py','locking.py','elf_inventory.py','recurrent_numerics.py','recurrent_protocol.py','resource_ledger.py'):
        shutil.copyfile(root/'core/qwen38'/n,run/'core/qwen38'/n)
    for n in ('guarded_run.py','sdk_container.py'):shutil.copyfile(root/'tools'/n,run/'tools'/n)
    for n in required:shutil.copyfile(reference/n,run/n)
    (run/'reference-result.json').write_text(json.dumps(result,indent=2)+'\n')
    (run/'resource-ledger.json').write_text(json.dumps(LEDGER,indent=2)+'\n')
    (run/'sram-policy.json').write_text(json.dumps({'ordinary_address_ceiling':49152,'stack_allowance_bytes':4096,
        'persistent_state_bytes_per_pe':32768,'scope':'both actual application section ends plus stack allowance must fit'},indent=2)+'\n')
    # Four mandatory full-state observations dominate host traffic. This is a planning
    # allowance based on WP05 measured57.1s, not an isolated bandwidth measurement.
    (run/'transfer-plan.json').write_text(json.dumps({'full_state_readbacks':4,'full_state_values_each':16384,
        'estimated_host_words_upper':110000,'fabric_words_per_token':393,'planning_sim_seconds':220,
        'hard_sim_seconds':300,'basis':'WP05 observed57.1s with roughly80k host words; allow substantial command/protocol overhead; no bandwidth guarantee'},indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    spec={'required_profile':'sdk','planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3','--fabric-dims=9,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']},
        {'name':'simulate','seconds':300,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':files,'scope':'synthetic full128x128 two-PE recurrent head'},indent=2)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('run','image','reference'):parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();print(prepare(Path(__file__).resolve().parents[1],args.run,args.image,args.reference))
