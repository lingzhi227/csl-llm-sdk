"""Package-isolated linear boundary with the derived context8 observer limit."""
from pathlib import Path
import hashlib, types
from stage_contract import Context, CONTEXT_CAPACITY, require
from observer_capacity import wire_limit

BOUNDARY_SHA = '2f1f45737c402bfd4aaea31baaa301d025fc24563f598ec9f760a4ef84f05496'


def bind(directory, context):
    require(isinstance(context, Context) and 0 < len(context.positions) <= CONTEXT_CAPACITY,
            'Finite scheduled linear context')
    raw = (Path(directory) / 'runtime_boundary.py').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == BOUNDARY_SHA, 'Exact accepted linear boundary source')
    text = raw.decode()
    old = 'a[0]<=32768'
    require(text.count(old) == 1, 'One original linear observer bound')
    text = text.replace(old, 'a[0]<=ORIGIN_WIRE_LIMIT', 1)
    module = types.ModuleType('scheduled_linear_boundary')
    module.__package__ = 'qualified_linear'
    module.ORIGIN_WIRE_LIMIT = wire_limit('linear')
    exec(compile(text, '<pinned-context-linear-boundary>', 'exec'), module.__dict__)
    module.context = context
    return module
