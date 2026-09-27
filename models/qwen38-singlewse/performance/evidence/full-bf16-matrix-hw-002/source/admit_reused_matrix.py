"""No-job source, archive and SRAM admission after the original host size gate."""
import hashlib,json
from pathlib import Path
from compile_hw import admit
from source_gate import verify

verify();root=Path.cwd();reuse=json.loads((root/'reuse-compiler-output.json').read_text())
config=json.loads((root/'experiment.json').read_text())
assert reuse['compiler_succeeded_and_released'] and reuse['original_host_stage_exit_code']==1
artifact=root/reuse['artifact_file'];assert artifact.stat().st_size==reuse['artifact_bytes']<=16<<20
assert hashlib.sha256(artifact.read_bytes()).hexdigest()==reuse['artifact_sha256']
assert {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in root.glob('*.csl')}==reuse['csl_hashes']
sram=admit(root,artifact,config);verify()
result=dict(passed=True,physical_jobs=0,source_attempt=reuse['source_attempt'],artifact_sha256=reuse['artifact_sha256'],
            compressed_bytes=artifact.stat().st_size,application_pes=sram['application_pes'],
            max_sram_with_stack=max(r['low_section_end']+r['stack_allowance_bytes'] for r in sram['records']),
            scope='Reuses successfully compiled and released SDK output; prior host stage failed its8MiB archive gate. This admission checks every embedded CSL source, actual1262PE coverage and SRAM. No numerical/runtime acceptance implied.')
with (root/'artifact-admission.json').open('x') as f:json.dump(result,f,indent=2);f.write('\n')
print(json.dumps(result))
