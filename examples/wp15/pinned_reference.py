"""Selected official projection/RMS/RoPE/eager statements with transparent hooks."""
import ast
import copy
import hashlib
from contextlib import nullcontext
from types import MethodType, SimpleNamespace
import torch
from torch import nn


def node_label(node):
    suffix=type(node.op).__name__ if isinstance(node,ast.BinOp) else node.func.attr
    return f'{node.lineno}:{node.col_offset}:{suffix}'


class Hooks(ast.NodeTransformer):
    def visit_ClassDef(self,node):
        node.decorator_list=[]
        return self.generic_visit(node)

    def visit_FunctionDef(self,node):
        node.decorator_list=[]
        return self.generic_visit(node)

    def wrap(self,node,label):
        return ast.copy_location(ast.Call(func=ast.Name(id='observe',ctx=ast.Load()),args=[ast.Constant(label),node],keywords=[]),node)

    def visit_BinOp(self,node):
        label=node_label(node)
        return self.wrap(self.generic_visit(node),label)

    def visit_Call(self,node):
        attr=node.func.attr if isinstance(node.func,ast.Attribute) else ''
        label=node_label(node) if attr in ('matmul','softmax','sigmoid','pow','mean','rsqrt') else None
        result=self.generic_visit(node)
        return self.wrap(result,label) if label else result


class PinnedReference:
    def __init__(self,raw,weights,norm_weights):
        assert hashlib.sha256(raw).hexdigest()=='762feb6c7426a7f15b5bf830df54c07438bf9e7c27b8cdb23179045920412c3b'
        tree=ast.parse(raw);self.trace={};self.phase='setup';self.captured={}
        rms=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5RMSNorm')
        rotary=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5TextRotaryEmbedding')
        functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ('rotate_half','apply_rotary_pos_emb','repeat_kv','eager_attention_forward')]
        methods=[n for n in rotary.body if isinstance(n,ast.FunctionDef) and n.name in ('compute_default_rope_parameters','forward','recomposition_frequencies')]
        selected=[rms,*functions,*methods]
        module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),
            *[Hooks().visit(copy.deepcopy(n)) for n in selected]],type_ignores=[])
        self.ns=dict(torch=torch,nn=nn,observe=self.observe,maybe_autocast=lambda **kwargs:nullcontext())
        exec(compile(ast.fix_missing_locations(module),'pinned_projected_attention','exec'),self.ns)
        attention=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Qwen3_5Attention')
        forward=next(n for n in attention.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
        prefix=[]
        for statement in forward.body:
            prefix.append(copy.deepcopy(statement))
            if isinstance(statement,ast.Assign) and any(isinstance(n,ast.Name) and n.id=='value_states' for n in statement.targets):break
        assert prefix[-1].targets[0].id=='value_states'
        self.prefix=compile(ast.fix_missing_locations(ast.Module(body=prefix,type_ignores=[])),'pinned_projection_prefix','exec')
        gate=next(n for n in forward.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp)
                  and isinstance(n.value.right,ast.Call) and isinstance(n.value.right.func,ast.Attribute) and n.value.right.func.attr=='sigmoid')
        self.gate_code=compile(ast.fix_missing_locations(ast.Module(body=[Hooks().visit(copy.deepcopy(gate))],type_ignores=[])),'pinned_gate_statement','exec')
        eager=next(n for n in functions if n.name=='eager_attention_forward')
        matmuls=[n for n in ast.walk(eager) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='matmul']
        qk=next(n for n in matmuls if isinstance(n.args[0],ast.Name) and n.args[0].id=='query')
        av=next(n for n in matmuls if n is not qk)
        scaled=next(n for n in ast.walk(eager) if isinstance(n,ast.BinOp) and isinstance(n.left,ast.Call) and n.left is qk)
        softmax=next(n for n in ast.walk(eager) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='softmax')
        self.attention_labels=dict(dot_bf16=node_label(qk),scaled_bf16=node_label(scaled),
            probability=node_label(softmax),attention_bf16=node_label(av),gate_bf16=node_label(gate.value.right),output=node_label(gate.value))
        apply=next(n for n in functions if n.name=='apply_rotary_pos_emb')
        self.rotary_labels={}
        for name in ('q_embed','k_embed'):
            expression=next(n.value for n in apply.body if isinstance(n,ast.Assign) and isinstance(n.value,ast.BinOp)
                and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
            self.rotary_labels[name]=[node_label(n) for n in (expression.left,expression.right,expression)]
        self.model=SimpleNamespace(head_dim=256)
        for name in ('q_proj','k_proj','v_proj'):
            layer=nn.Linear(5120,weights[name].shape[0],bias=False,dtype=torch.bfloat16)
            layer.weight=nn.Parameter(torch.from_numpy(weights[name]).to(torch.bfloat16),requires_grad=False)
            layer.register_forward_hook(self.capture(name));setattr(self.model,name,layer)
        for name,w in zip(('q_norm','k_norm'),norm_weights):
            layer=self.ns['Qwen3_5RMSNorm'](256,eps=1e-6)
            layer.weight=nn.Parameter(torch.from_numpy(w).to(torch.bfloat16),requires_grad=False)
            layer.register_forward_pre_hook(self.before(name));layer.register_forward_hook(self.capture(name));setattr(self.model,name,layer)
        config=SimpleNamespace(rope_parameters={'rope_theta':10000000,'partial_rotary_factor':.25},head_dim=256)
        self.frequencies,scale=self.ns['compute_default_rope_parameters'](config)
        assert self.frequencies.dtype==torch.float32 and self.frequencies.shape==(32,) and scale==1
        self.rope=SimpleNamespace(inv_freq=self.frequencies,attention_scaling=scale,mrope_section=[11,11,10])
        self.rope.recomposition_frequencies=MethodType(self.ns['recomposition_frequencies'],self.rope)
        lines=raw.decode().splitlines(True)
        self.identity=dict(prefix_lines=[prefix[0].lineno,prefix[-1].end_lineno],gate_line=gate.lineno,
            bodies={n.name:dict(span=[n.lineno,n.end_lineno],sha256=hashlib.sha256(''.join(lines[n.lineno-1:n.end_lineno]).encode()).hexdigest()) for n in selected},
            adaptation='Selected row modules; original projection/view/chunk/RMS/eager/gate statements. Wrappers return identical tensor objects; integration decorators removed; fixed text axes/default RoPE and disabled CPU autocast.')

    def observe(self,label,value):
        if isinstance(value,torch.Tensor):self.trace[self.phase+':'+label]=value.detach().clone()
        return value

    def capture(self,name):
        def hook(module,args,output):self.captured[name]=output.detach().clone()
        return hook

    def before(self,name):
        def hook(module,args):self.phase=name
        return hook

    def rotate(self,qnorm,knorm,position):
        self.phase='frequency'
        cosine,sine=self.ns['forward'](self.rope,qnorm,torch.full((3,1,1),position,dtype=torch.int64))
        self.phase='rotary'
        q,k=self.ns['apply_rotary_pos_emb'](qnorm.reshape(1,1,1,256),knorm.reshape(1,1,1,256),cosine,sine)
        return q,k,cosine,sine

    def project(self,x,position):
        self.trace={};self.captured={}
        with torch.no_grad():
            xt=torch.from_numpy(x.copy()).to(torch.bfloat16).reshape(1,1,5120)
            env=dict(self.ns,self=self.model,hidden_states=xt);exec(self.prefix,env)
            q,k,cosine,sine=self.rotate(env['query_states'],env['key_states'],position)
        return dict(q=q,k=k,v=env['value_states'],gate=env['gate'],cosine=cosine,sine=sine,
                    captured=dict(self.captured),trace=dict(self.trace))

    def attend(self,q,keys,values,gate):
        self.trace={};self.phase='attention'
        with torch.no_grad():
            output,probability=self.ns['eager_attention_forward'](SimpleNamespace(num_key_value_groups=1,training=False),q,
                torch.cat(keys,dim=2),torch.cat(values,dim=2),None,1/16)
            env=dict(self.ns,attn_output=output.reshape(1,1,256),gate=gate.reshape(1,1,256))
            exec(self.gate_code,env)
        return dict(output=env['attn_output'],attention=output,probability=probability,
                    stages={name:self.trace['attention:'+label] for name,label in self.attention_labels.items()},trace=dict(self.trace))
