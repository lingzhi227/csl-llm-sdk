"""Frozen-profile compile and actual per-PE ELF SRAM admission; no simulation here."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
sys.path.insert(0,str(Path.cwd()/'core'))
from qwen38.elf_inventory import inventory, admit_wse3_sram

root=Path.cwd()
for name,digest in json.loads((root/'source-manifest.json').read_text())['files'].items():
    if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
        raise ValueError('Frozen identity mismatch: '+name)
result=subprocess.run(['sdk_debug_shell','compile',*sys.argv[1:]])
if result.returncode:
    raise SystemExit(result.returncode)
compiled={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted((root/'out').rglob('*')) if p.is_file()}
(root/'compiled-subset-before-run.json').write_text(json.dumps(compiled,indent=2)+'\n')
paths=sorted((root/'out/bin').glob('out_*_*.elf'))
profile=json.loads((root/'profile.json').read_text())
expected_pes={'wp13-four-pe-full5120-dense-diagnostic':4,'wp13-single-pe-four-case-slab':1}[profile['scope']]
assert profile['pes']==expected_pes
assert {p.name for p in paths}=={f'out_{x}_0.elf' for x in range(expected_pes)}
footprints={p.name:inventory(p.read_bytes()) for p in paths}
admissions={name:admit_wse3_sram(fp,4096,49152) for name,fp in footprints.items()}
(root/'pe-footprint.json').write_text(json.dumps(footprints,indent=2)+'\n')
result={'passed':all(a['passed'] for a in admissions.values()),'pes':admissions}
(root/'sram-admission.json').write_text(json.dumps(result,indent=2)+'\n')
if not result['passed']:
    raise RuntimeError('Per-PE actual SRAM end plus stack exceeds49152')
