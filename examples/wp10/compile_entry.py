"""Record compile inputs/output identity before any runtime products exist."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent/'core'))
from qwen38.elf_inventory import inventory, admit_wse3_sram
root=Path.cwd()
manifest=json.loads((root/'source-manifest.json').read_text())
for name,digest in manifest['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Frozen input mismatch: '+name)
info={'python':sys.executable,'version':sys.version,'cwd':str(root),'pid':os.getpid(),
      'tmpdir':os.environ.get('TMPDIR'),'cgroup':Path('/proc/self/cgroup').read_text()}
(root/'container-compile.json').write_text(json.dumps(info,indent=2)+'\n')
result=subprocess.run(['sdk_debug_shell','compile',*sys.argv[1:]])
if result.returncode: raise SystemExit(result.returncode)
files={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
       for p in sorted((root/'out').rglob('*')) if p.is_file()}
(root/'compiled-subset-before-run.json').write_text(json.dumps(files,indent=2)+'\n')
(root/'pe-footprint.json').write_text(json.dumps(inventory((root/'out/bin/out_0_0.elf').read_bytes()),indent=2)+'\n')

footprint=json.loads((root/'pe-footprint.json').read_text())
policy=json.loads((root/'sram-policy.json').read_text())
admission=admit_wse3_sram(footprint,policy['stack_allowance_bytes'],policy['ordinary_address_ceiling'])
(root/'sram-admission.json').write_text(json.dumps(admission,indent=2)+'\n')
if not admission['passed']:raise RuntimeError('Static SRAM plus declared stack allowance exceeds admission ceiling')
