"""Load package-isolated qualified family host protocol checks."""
from pathlib import Path
import json
from qualified_linear import runtime_boundary as linear
from qualified_attention import runtime_boundary as attention


def load():
    root = Path(__file__).resolve().parent
    result = {}
    for name, module in [('linear', linear), ('attention', attention)]:
        directory = root/('qualified_'+name)
        result[name] = dict(module=module, plan=json.loads((directory/'plan.json').read_bytes()),
                            transport=json.loads((directory/'transport-map.json').read_bytes()))
    return result
