"""Load only the frozen arithmetic namespace; never a sys.path-selected qwen38."""
from pathlib import Path
import hashlib,importlib,inspect,json

def load():
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'manifest.json').read_bytes())
    for name,pin in manifest.items():
        p=root/name;raw=p.read_bytes()
        if Path(name).name!=name or p.is_symlink() or dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())!=pin:
            raise ValueError('Frozen qualified arithmetic source identity')
    targets=dict(local_bounds=('local_bounds','local_bounds'),exact_dot=('exact_arithmetic','exact_dot'),norm_gate=('projection_checks','norm_gate'),nonlinear_gate=('projection_checks','nonlinear_gate'),validate_qk=('qk_native_conditional','validate_qk'),validate_attention=('attention_conditional','validate_attention'))
    result={}
    for key,(module,function) in targets.items():
        value=getattr(importlib.import_module('.'+module,__name__),function)
        if Path(inspect.getsourcefile(value)).resolve()!=root/(module+'.py'):
            raise ValueError('Qualified function came from a different module cache/path')
        result[key]=value
    return result
