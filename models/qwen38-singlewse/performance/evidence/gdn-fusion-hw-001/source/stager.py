"""Stage the complete connected component after numerical smoke and ELF census."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]


def qualified_sources(sim,full):
    complete=json.loads((sim/'COMPLETE.json').read_text())
    result=json.loads((sim/'result.json').read_text())
    if (not complete['passed'] or complete['physical'] or not result['passed'] or
        not result['normal_stop'] or result['workers']!=39 or result['groups']!=[0] or
        result['positions']!=2 or result['host_injected_recurrent_results']):
        raise ValueError('Connected three-head, two-position simulator qualification required')
    if len(result['records'])!=2 or not all(all(r[k] for k in
        ['frontend_interval','recurrent_state_interval','core_interval','gated_interval',
         'exact_history','all_return_markers','delayed_source_callback']) for r in result['records']):
        raise ValueError('Incomplete connected numerical qualification')
    if not all(json.loads((p/'release.json').read_text())['released'] for p in [sim,full]):
        raise ValueError('Workstation owner not released')
    census=json.loads((full/'sram.json').read_text())
    if not census['passed'] or census['application']!=[160,5] or census['application_pes']!=800:
        raise ValueError('Complete connected compiler admission required')
    if not json.loads((full/'COMPLETE.json').read_text())['passed']:
        raise ValueError('Full compile did not complete')
    manifests=[json.loads((p/'source-manifest.json').read_text())['files'] for p in [sim,full]]
    for p,m in zip([sim,full],manifests):
        for n,digest in m.items():
            if hashlib.sha256((p/'source'/n).read_bytes()).hexdigest()!=digest:
                raise ValueError('Frozen source changed: '+n)
    names=[n for n in manifests[1] if n.endswith('.csl') or n.startswith('reference/') or n in ['probe-plan.json','run.py']]
    files={n:(full/'source'/n).read_bytes() for n in names}
    for n in names:
        if n not in ['layout.csl','probe-plan.json'] and manifests[0][n]!=manifests[1][n]:
            raise ValueError('Full component differs from qualified smoke: '+n)
    small=json.loads((sim/'source/probe-plan.json').read_text())
    large=json.loads(files['probe-plan.json'])
    if (len(large['workers'])!=753 or sum(w['columns'] for w in large['workers'])!=6144 or
        not large['full_original_value_coverage'] or large['host_injected_recurrent_results'] or
        any(w not in large['workers'] for w in small['workers'])):
        raise ValueError('Incomplete original state ownership')
    return files


def main():
    parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--simulation',required=True);parser.add_argument('--full-compile',required=True);args=parser.parse_args()
    if not re.fullmatch('[0-9]{3}',args.attempt) or any(not re.fullmatch('gdn-fusion-sim-[0-9]{3}',v) for v in [args.simulation,args.full_compile]):raise ValueError('Attempt identity')
    name='gdn-fusion-hw-'+args.attempt;out=ROOT/'performance/evidence'/name
    if out.exists():raise ValueError('Frozen attempt')
    sim=ROOT/'performance/evidence'/args.simulation;full=ROOT/'performance/evidence'/args.full_compile
    files=qualified_sources(sim,full)
    files['reference/gdn_fusion_audit.py']=(ROOT/'performance/reference/gdn_fusion_audit.py').read_bytes()
    small_remote=json.loads((sim/'dispatch.json').read_text())['remote'];full_remote=json.loads((full/'dispatch.json').read_text())['remote']
    # Verify the complete fixture and exact smoke subset remotely. Model payloads
    # do not pass through local storage, and smoke data cannot stand in for16 groups.
    script='small='+repr(small_remote)+'\nfull='+repr(full_remote)+'\n'+'''import hashlib,json,numpy as np
from pathlib import Path
roots=[Path(small),Path(full)];metas=[json.loads((p/'fixture.json').read_text()) for p in roots]
for p,m in zip(roots,metas):assert hashlib.sha256((p/'fixture.npz').read_bytes()).hexdigest()==m['fixture_sha256']
assert metas[0]['groups']==[0] and metas[1]['groups']==list(range(16))
assert metas[0]['original_fixture_sha256']==metas[1]['original_fixture_sha256']
with np.load(roots[0]/'fixture.npz',allow_pickle=False) as a,np.load(roots[1]/'fixture.npz',allow_pickle=False) as b:
 for n in a.files:
  np.testing.assert_array_equal(a[n],b[n] if n.startswith('projected_') else b[n][:1])
print(json.dumps(dict(passed=True,fixture_sha256=metas[1]['fixture_sha256'],original_fixture_sha256=metas[1]['original_fixture_sha256'],exact_smoke_subset=True)))
'''
    fixture_audit=json.loads(subprocess.check_output(['ssh','workstation','/usr/bin/python3 -c '+shlex.quote(script)],text=True,timeout=30))
    if fixture_audit['fixture_sha256']!=json.loads((full/'fixture.json').read_text())['fixture_sha256']:raise ValueError('Full fixture receipt mismatch')
    files['qualified-smoke-manifest.json']=(sim/'source-manifest.json').read_bytes();files['full-compile-manifest.json']=(full/'source-manifest.json').read_bytes()
    files['fixture-subset-audit.json']=(json.dumps(fixture_audit,indent=2)+'\n').encode()
    files['experiment.json']=(json.dumps(dict(original_frontend_gdn_fusion=True,full_model=False,application=[160,5],application_pes=800,frontend_shape=[16,3,128],gdn_shape=[48,128,128],gdn_workers=753,host_injected_recurrent_results=False,fabric_offset=[4,1],revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',artifact_single_message_limit=16<<20,scope=json.loads(files['probe-plan.json'])['scope'],simulation_protocol_source=args.simulation,full_compiler_source=args.full_compile),indent=2)+'\n').encode()
    for n in ['bounded_client.py','run_hw.py','job_capture.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
    for n in ['backend.py','supervise.py','check_sram.py','placement.py','bounded_compiler.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
    files['compile_hw.py']=(ROOT/'performance/runtime/compile_component.py').read_bytes()
    for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
    files['stager.py']=Path(__file__).read_bytes();files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
    destination='/srv/qwen38-singlewse-hardware/'+name;session=['sh','/path/to/alcf-session.sh','host'];buffer=io.BytesIO()
    with tarfile.open(fileobj=buffer,mode='w') as tar:
        for n,b in files.items():item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
    subprocess.run(session+['mkdir '+shlex.quote(destination)+' && tar -xf - -C '+shlex.quote(destination)],input=buffer.getvalue(),check=True,timeout=30)
    reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(full_remote)+' fixture.npz fixture.json'],stdout=subprocess.PIPE)
    try:subprocess.run(session+['tar -xf - -C '+shlex.quote(destination)],stdin=reader.stdout,check=True,timeout=60)
    finally:
        reader.stdout.close()
        try:
            if reader.wait(timeout=20)!=0:raise ValueError('Fixture transfer failed')
        except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
    expected=fixture_audit['fixture_sha256']
    script='from pathlib import Path;import hashlib,json;p=Path('+repr(destination)+');m=json.loads((p/"fixture.json").read_text());assert hashlib.sha256((p/"fixture.npz").read_bytes()).hexdigest()==m["fixture_sha256"]=='+repr(expected)
    subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=20)
    out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'payload-staging.json').write_text(json.dumps(dict(remote=destination,simulation=args.simulation,full_compile=args.full_compile,payload_verified=True,fixture_sha256=expected,physical_dispatched=False),indent=2)+'\n')
    print(json.dumps(dict(name=name,physical_dispatched=False,application_pes=800)))


if __name__=='__main__':main()
