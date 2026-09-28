"""Stage all753 compiled slices after exact8/32-column transport smoke."""
import argparse,hashlib,io,json,re,shlex,subprocess,tarfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def main():
 parser=argparse.ArgumentParser();parser.add_argument('attempt');parser.add_argument('--simulation',required=True);parser.add_argument('--full-compile',required=True);args=parser.parse_args()
 if not re.fullmatch('[0-9]{3}',args.attempt) or any(not re.fullmatch('gdn-columns-sim-[0-9]{3}',v) for v in [args.simulation,args.full_compile]):raise ValueError('Attempt identity')
 name='gdn-columns-hw-'+args.attempt;out=ROOT/'performance/evidence'/name
 if out.exists():raise ValueError('Frozen attempt')
 sim=ROOT/'performance/evidence'/args.simulation;complete=json.loads((sim/'COMPLETE.json').read_text());result=complete['result']
 if not result['passed'] or not result['normal_stop'] or complete['physical'] or result['workers']!=3 or not result['records']:raise ValueError('Exact-port simulator smoke required')
 if not all(r['state_interval_pass'] and r['output_interval_pass'] and r['transport_drained'] for r in result['records']):raise ValueError('Incomplete simulator qualification')
 if not json.loads((sim/'workstation-release.json').read_text())['workstation_released']:raise ValueError('Simulator still owned')
 frozen=json.loads((sim/'source-manifest.json').read_text())['files'];files={}
 full=ROOT/'performance/evidence'/args.full_compile;full_hashes=json.loads((full/'source-manifest.json').read_text())['files']
 census=json.loads((full/'sram.json').read_text())
 if not census['passed'] or census['application']!=[251,6] or census['application_pes']!=1506 or not json.loads((full/'workstation-release.json').read_text())['workstation_released']:raise ValueError('Complete all-slice compiler admission required')
 for n,digest in full_hashes.items():
  if n.endswith('.csl') or n=='probe-plan.json':
   b=(full/'source'/n).read_bytes()
   if hashlib.sha256(b).hexdigest()!=digest:raise ValueError('Full compiled source identity changed')
   if n.endswith('.csl') and n!='layout.csl' and hashlib.sha256(b).hexdigest()!=frozen[n]:raise ValueError('Complete kernel differs from exact-port simulator')
   files[n]=b
 files['run.py']=(sim/'source/run.py').read_bytes()
 if hashlib.sha256(files['run.py']).hexdigest()!=frozen['run.py']:raise ValueError('Qualified driver changed')
 sample=json.loads((sim/'source/probe-plan.json').read_text());all_workers=json.loads(files['probe-plan.json'])['workers']
 if sorted(w['columns'] for w in sample['workers'])!=[8,8,32] or any(w not in all_workers for w in sample['workers']):raise ValueError('Original value slices not covered by smoke')
 if len(all_workers)!=753 or sum(w['columns'] for w in all_workers)!=6144:raise ValueError('Incomplete physical value ownership')
 files['qualified-smoke-manifest.json']=(sim/'source-manifest.json').read_bytes();files['full-compile-manifest.json']=(full/'source-manifest.json').read_bytes()
 files['experiment.json']=(json.dumps(dict(original_gdn_columns=True,full_model=False,application=[251,6],application_pes=1506,gdn_shape=[48,128,128],gdn_workers=753,fabric_offset=[4,1],revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',artifact_single_message_limit=16<<20,scope=json.loads((sim/'fixture.json').read_text())['scope'],simulation_protocol_source=args.simulation,full_compiler_source=args.full_compile),indent=2)+'\n').encode()
 for n in ['bounded_client.py','run_hw.py','job_capture.py','source_gate.py','elf_inventory.py']:files[n]=(ROOT/'runtime'/n).read_bytes()
 for n in ['backend.py','supervise.py','check_sram.py','placement.py','bounded_compiler.py']:files[n]=(ROOT/'performance/runtime'/n).read_bytes()
 for n in ['check_sram.py','placement.py']:
  if hashlib.sha256(files[n]).hexdigest()!=frozen[n]:raise ValueError('Simulator admission source changed')
 files['compile_hw.py']=(ROOT/'performance/runtime/compile_component.py').read_bytes()
 for n in ['lifecycle.py','store.py','__init__.py']:files['runtime/'+n]=(ROOT/'runtime'/n).read_bytes()
 files['stager.py']=Path(__file__).read_bytes();files['source-manifest.json']=(json.dumps(dict(files={n:hashlib.sha256(b).hexdigest() for n,b in files.items()}),indent=2)+'\n').encode()
 destination='/srv/qwen38-singlewse-hardware/'+name;session=['sh','/path/to/alcf-session.sh','host'];buffer=io.BytesIO()
 with tarfile.open(fileobj=buffer,mode='w') as tar:
  for n,b in files.items():item=tarfile.TarInfo(n);item.size=len(b);tar.addfile(item,io.BytesIO(b))
 subprocess.run(session+['mkdir '+shlex.quote(destination)+' && tar -xf - -C '+shlex.quote(destination)],input=buffer.getvalue(),check=True,timeout=30)
 source=json.loads((sim/'dispatch.json').read_text())['remote']
 reader=subprocess.Popen(['ssh','workstation','tar -cf - -C '+shlex.quote(source)+' fixture.npz fixture.json'],stdout=subprocess.PIPE)
 try:subprocess.run(session+['tar -xf - -C '+shlex.quote(destination)],stdin=reader.stdout,check=True,timeout=60)
 finally:
  reader.stdout.close()
  try:
   if reader.wait(timeout=20)!=0:raise ValueError('Reference transfer failed')
  except subprocess.TimeoutExpired:reader.kill();reader.wait();raise
 expected=json.loads((sim/'fixture.json').read_text())['fixture_sha256']
 script='from pathlib import Path;import hashlib,json;p=Path('+repr(destination)+');m=json.loads((p/"fixture.json").read_text());assert hashlib.sha256((p/"fixture.npz").read_bytes()).hexdigest()==m["fixture_sha256"]=='+repr(expected)
 subprocess.run(session+['python3 -c '+shlex.quote(script)],check=True,timeout=20)
 out.mkdir();(out/'source-manifest.json').write_bytes(files['source-manifest.json']);(out/'payload-staging.json').write_text(json.dumps(dict(remote=destination,simulation=args.simulation,payload_verified=True,fixture_sha256=expected,physical_dispatched=False),indent=2)+'\n')
 print(json.dumps(dict(name=name,physical_dispatched=False,application_pes=1506)))

if __name__=='__main__':main()
