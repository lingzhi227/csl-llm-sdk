"""Package-isolated family protocols and context-bound attention decoders."""
from pathlib import Path
import json
from attention_context import bind
from linear_context import bind as bind_linear


def load(context,root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parent
    linear=bind_linear(root/'qualified_linear',context)
    attention=bind(root/'qualified_attention',context)
    result={}
    for name,module in [('linear',linear),('attention',attention)]:
        directory=root/('qualified_'+name)
        result[name]=dict(module=module,context=context,plan=json.loads((directory/'plan.json').read_bytes()),
                          transport=json.loads((directory/'transport-map.json').read_bytes()))
    return result
