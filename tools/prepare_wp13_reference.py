"""Recreate the accepted selected-head CPU inputs and source closure; never run them."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct


def hidden_bytes():
    vectors = [[((c*13+7)%33-16)/4096 for c in range(5120)],
               [((c*19+11)%31-15)/4096 for c in range(5120)],
               [float(c==5119) for c in range(5120)], [0.]*5120]
    return b''.join(struct.pack('<H',struct.unpack('<I',struct.pack('<f',v))[0]>>16)
                    for row in vectors for v in row)


def prepare(root,run,source_dir,weights,interpreter):
    run=run.resolve();weights=weights.resolve();interpreter=interpreter.absolute()
    if not interpreter.is_file():raise ValueError('Missing existing reference interpreter')
    recipe=json.loads((root/'examples/wp13/reference-recipe.json').read_text())
    source=source_dir/'modeling_qwen3_5.py'
    if hashlib.sha256(source.read_bytes()).hexdigest()!=recipe['source_sha256']:
        raise ValueError('Pinned official source identity mismatch')
    receipt=json.loads((weights/'download.json').read_text())
    plan=json.loads((root/'examples/wp13/acquisition-plan.json').read_text())
    if receipt['status']!='complete' or receipt['plan']!=plan or receipt['payload_bytes_received']!=10486784:
        raise ValueError('Original selected-head acquisition mismatch')
    for name,identity in receipt['files'].items():
        if hashlib.sha256((weights/name).read_bytes()).hexdigest()!=identity['sha256']:
            raise ValueError('Original weight hash mismatch')
    hidden=hidden_bytes()
    if hashlib.sha256(hidden).hexdigest()!=recipe['hidden_sha256']:
        raise ValueError('Frozen hidden input mismatch')
    run.mkdir(parents=True,exist_ok=False)
    (run/'core/qwen38').mkdir(parents=True);(run/'tools').mkdir()
    for name in ('__init__.py','resources.py','locking.py','rms_numerics.py',
                 'gated_numerics.py','interval_numerics.py','attention_numerics.py'):
        shutil.copyfile(root/'core/qwen38'/name,run/'core/qwen38'/name)
    shutil.copyfile(root/'tools/guarded_run.py',run/'tools/guarded_run.py')
    shutil.copyfile(root/'examples/wp13/build_reference.py',run/'build_reference.py')
    shutil.copyfile(source,run/'modeling_qwen3_5.py')
    (run/'hidden.bf16').write_bytes(hidden)
    recipe.update(weights_directory=str(weights),weight_files=receipt['files'],
                  dependencies='Existing CPU Python3.11, torch2.4.0 and numpy1.25.0; no installation by this preparer.')
    (run/'reference-plan.json').write_text(json.dumps(recipe,indent=2)+'\n')
    spec={'planned_write_bytes':8*1024*1024,'required_profile':'cpu-reference',
          'steps':[{'name':'reference','seconds':60,'argv':[str(interpreter),'build_reference.py']}]}
    (run/'steps.json').write_text(json.dumps(spec,indent=2)+'\n')
    files={str(p.relative_to(run)):hashlib.sha256(p.read_bytes()).hexdigest()
           for p in sorted(run.rglob('*')) if p.is_file()}
    (run/'source-manifest.json').write_text(json.dumps(files,indent=2)+'\n')
    return run


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--source-dir',type=Path,required=True)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--python',type=Path,required=True)
    args=parser.parse_args()
    print(prepare(Path(__file__).resolve().parents[1],args.run,args.source_dir,args.weights,args.python))
