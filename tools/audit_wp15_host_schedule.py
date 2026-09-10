"""Static AST enumeration of host copy/launch order; no SDK/numpy/model import."""
import argparse,ast,json,operator,re,sys
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from qwen38.wp15_layout import ARRAYS,QK_INITIAL,ATTENTION_INITIAL,QK_READS,ATTENTION_READS,sequence,declared_bytes,integer
from qwen38.projected_attention_protocol import FRAGMENTS


def audit(directory):
    driver=ast.parse((directory/'driver.py').read_text());checks=ast.parse((directory/'consumer_checks.py').read_text())
    methods={n.name:n for c in checks.body if isinstance(c,ast.ClassDef) for n in c.body if isinstance(n,ast.FunctionDef)}
    followed={'initialize','begin_metadata','verify_begin','handoff_and_compute','preprocess','verify_release'}
    operations=[];shared=SimpleNamespace()
    constants=dict(pes=8,cases=[0,1],FRAGMENTS=FRAGMENTS,QK_INITIAL=QK_INITIAL,ATTENTION_INITIAL=ATTENTION_INITIAL,
                   QK_READS=QK_READS,ATTENTION_READS=ATTENTION_READS,range=range,enumerate=enumerate,
                   ref={'cases':['bounded_dyadic','changed_dyadic','last_column_onehot','zero_after_nonzero']},self=shared)
    binops={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.FloorDiv:operator.floordiv,ast.Mod:operator.mod}
    cmps={ast.Eq:operator.eq,ast.NotEq:operator.ne,ast.Lt:operator.lt,ast.LtE:operator.le,ast.Gt:operator.gt,ast.GtE:operator.ge,ast.In:lambda a,b:a in b}
    def value(n,env):
        if isinstance(n,ast.Constant):return n.value
        if isinstance(n,ast.Name):return env[n.id]
        if isinstance(n,(ast.Tuple,ast.List)):
            out=[]
            for x in n.elts:
                if isinstance(x,ast.Starred):out.extend(value(x.value,env))
                else:out.append(value(x,env))
            return tuple(out) if isinstance(n,ast.Tuple) else out
        if isinstance(n,ast.Attribute):return getattr(value(n.value,env),n.attr)
        if isinstance(n,ast.Subscript):return value(n.value,env)[value(n.slice,env)]
        if isinstance(n,ast.UnaryOp) and isinstance(n.op,ast.USub):return -value(n.operand,env)
        if isinstance(n,ast.BinOp):return binops[type(n.op)](value(n.left,env),value(n.right,env))
        if isinstance(n,ast.Compare):
            vals=[value(n.left,env),*[value(x,env) for x in n.comparators]]
            return all(cmps[type(o)](a,b) for o,a,b in zip(n.ops,vals,vals[1:]))
        if isinstance(n,ast.BoolOp):return (all if isinstance(n.op,ast.And) else any)(value(x,env) for x in n.values)
        if isinstance(n,ast.IfExp):return value(n.body if value(n.test,env) else n.orelse,env)
        if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('range','enumerate'):
            return constants[n.func.id](*[value(x,env) for x in n.args])
        raise ValueError('Expression is not needed for static schedule: '+ast.dump(n)[:100])
    def assign(n,v,env):
        if isinstance(n,ast.Name):env[n.id]=v
        elif isinstance(n,(ast.Tuple,ast.List)):
            for a,b in zip(n.elts,v):assign(a,b,env)
        elif isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='self':setattr(shared,n.attr,v)
    def name(call):
        return call.func.id if isinstance(call.func,ast.Name) else call.func.attr if isinstance(call.func,ast.Attribute) else ''
    def relevant(n):
        return any(isinstance(x,ast.Call) and name(x) in {'transfer','launch','read','write',*followed} for x in ast.walk(n))
    def expression(n,env):
        if isinstance(n,(ast.DictComp,ast.ListComp)) and relevant(n):
            assert len(n.generators)==1 and not n.generators[0].ifs
            g=n.generators[0]
            for v in value(g.iter,env):
                e=env.copy();assign(g.target,v,e);expression(n.value if isinstance(n,ast.DictComp) else n.elt,e)
            return
        if isinstance(n,ast.Call):
            cmd=name(n)
            if cmd in followed and isinstance(n.func,ast.Attribute):
                e={**constants,**{arg.arg:value(v,env) for arg,v in zip(methods[cmd].args.args[1:],n.args)}}
                statements(methods[cmd].body,e);return
            if cmd in ('transfer','read','write'):
                if cmd=='transfer':direction,symbol,pe,count=[value(v,env) for v in n.args[:4]];bits=value(n.args[4],env) if len(n.args)>4 else 32
                else:
                    symbol=value(n.args[0],env);direction='h2d' if cmd=='write' else 'd2h'
                    pe=value(n.args[2 if cmd=='write' else 1],env);count,bits=ARRAYS[symbol]
                operations.append(dict(kind='copy',direction=direction,name=symbol,pe=pe,count=count,bits=bits,
                                       host_bytes=count*4,native_bytes=count*bits//8));return
            if cmd=='launch':operations.append(dict(kind='launch',name=value(n.args[0],env)));return
        for child in ast.iter_child_nodes(n):
            if relevant(child):expression(child,env)
    def statements(nodes,env):
        for n in nodes:
            if isinstance(n,(ast.Assign,ast.AnnAssign)):
                if relevant(n.value):expression(n.value,env)
                else:
                    try:v=value(n.value,env)
                    except (ValueError,KeyError,AttributeError,TypeError):continue
                    for target in (n.targets if isinstance(n,ast.Assign) else [n.target]):assign(target,v,env)
            elif isinstance(n,ast.For) and relevant(n):
                for v in value(n.iter,env):assign(n.target,v,env);statements(n.body,env)
            elif isinstance(n,ast.If) and relevant(n):statements(n.body if value(n.test,env) else n.orelse,env)
            elif isinstance(n,ast.Try):statements(n.body,env);statements(n.finalbody,env)
            elif isinstance(n,ast.Expr) and relevant(n):expression(n.value,env)
    # Only the actual runtime try body contains candidate operations. Helpers
    # like transfer()/launch() are handled as endpoints, never traversed twice.
    runtime=next(n for n in driver.body if isinstance(n,ast.Try))
    statements(runtime.body,constants.copy())
    copies=launches=0
    for r in operations:
        if r['kind']=='copy':copies+=1;r['sequence']=copies
        else:launches+=1;r['ordinal']=launches
    expected=sequence()
    if operations!=expected:
        for i,(a,b) in enumerate(zip(operations,expected)):
            if a!=b:raise AssertionError(f'Operation{i}: actualAST={a}, frozenplan={b}')
        raise AssertionError(f'Operation count differs:{len(operations)} vs{len(expected)}')
    inventories={}
    for role in ('producer','qk','attention'):
        text=(directory/(role+'.csl')).read_text()
        actual={name:(int(n),16 if t=='u16' else 32) for name,n,t in re.findall(r'var (\w+)=@zeros\(\[(\d+)\](u16|u32|f32)\);',text) if name in ARRAYS}
        assert actual==declared_bytes(role)['arrays'],role
        for name,n,t in re.findall(r'var (\w+)=@zeros\(\[(\d+)\](u16|u32|f32)\);',text):
            if name in ARRAYS:assert t==('u16' if ARRAYS[name][1]==16 else 'u32' if integer(name) else 'f32'),(role,name)
        inventories[role]=len(actual)
    return dict(passed=True,scope='Static host AST copy/launch order and declared CSL arrays; no compile/runtime/fabric qualification',
                operations=len(operations),copies=copies,launches=launches,inventory_symbols=inventories)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path);args=parser.parse_args()
    print(json.dumps(audit(args.directory)))
