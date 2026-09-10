"""Freeze one synthetic full5120 RMS candidate with explicit SRAM admission."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.rms_numerics import POLICY, PROBE_BITS, fixtures, reference, f32, bits, bf16_rne, from_bits


def prepare(root,run,image):
    run=run.resolve();image=image.resolve()
    if not image.is_file():raise ValueError('Reuse an existing qualified SDK image')
    cases=[[n,x,w] for n,x,w in fixtures()]
    name,x,w=cases[-1];ref=reference(x,w);inverse=f32(1/(ref['q']**0.5))
    before=[bf16_rne(bits(f32(f32(a*inverse)*f32(1+g)))) for a,g in zip(x,w)]
    after=[bf16_rne(bits(f32(from_bits(bf16_rne(bits(f32(a*inverse)))<<16)*f32(1+g)))) for a,g in zip(x,w)]
    differences=sum(a!=b for a,b in zip(before,after))
    if not differences:raise ValueError('Missing gain-before-cast adversary')
    run.mkdir(parents=True,exist_ok=False);(run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for n in ('pe.csl','layout.csl','driver.py','compile_entry.py'):shutil.copyfile(root/'examples/wp05'/n,run/n)
    shutil.copyfile(root/'csl/kernels/rmsnorm_f32_bf16.csl',run/'kernel.csl')
    for n in ('__init__.py','resources.py','locking.py','elf_inventory.py','rms_numerics.py'):shutil.copyfile(root/'core/qwen38'/n,run/'core/qwen38'/n)
    for n in ('guarded_run.py','sdk_container.py'):shutil.copyfile(root/'tools'/n,run/'tools'/n)
    (run/'fixtures.json').write_text(json.dumps(cases)+'\n')
    (run/'numerical-policy.json').write_text(json.dumps(POLICY,indent=2)+'\n')
    (run/'conversion-fixtures.json').write_text(json.dumps({'raw_fp32_bits':PROBE_BITS,'expected_bf16':[bf16_rne(x) for x in PROBE_BITS],
       'gain_before_cast_fixture_differences':differences,'synthetic':True},indent=2)+'\n')
    (run/'sram-policy.json').write_text(json.dumps({'ordinary_address_ceiling':49152,'stack_allowance_bytes':4096,
       'scope':'conservative project admission ceiling, not all SRAM availability or measured dynamic stack usage',
       'nominal_payload_bytes':30720,'inputs':'FP32 staging in-place; BF16 gain/output share storage, gain consumed before write, full gain re-upload next call'},indent=2)+'\n')
    prefix=[sys.executable,str(run/'tools/sdk_container.py'),'--image',str(image),'--','python']
    spec={'required_profile':'sdk','planned_write_bytes':134217728,'steps':[
        {'name':'compile','seconds':300,'argv':prefix+['compile_entry.py','layout.csl','--arch=wse3','--fabric-dims=8,3','--fabric-offsets=4,1','-o=out','--memcpy','--channels=1','--max-parallelism=1','--dump-dsr-alloc-graph']},
        {'name':'simulate','seconds':180,'argv':prefix+['driver.py']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    m={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps({'files':m,'scope':'synthetic ordinary RMS5120 BF16 RNE'},indent=2)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run',type=Path,required=True);parser.add_argument('--image',type=Path,required=True)
    args=parser.parse_args();print(prepare(Path(__file__).resolve().parents[1],args.run,args.image))
