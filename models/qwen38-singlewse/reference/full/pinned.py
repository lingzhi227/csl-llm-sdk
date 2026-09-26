"""Source-only full-text reference adapter; imports Torch only when constructed.

Selected official arithmetic/cache method bodies remain unchanged. Two rotary
buffer declarations use equivalent register_buffer calls for Torch2.4. Optional
integration/deprecation/dynamic-RoPE decorators are removed for eager CPU use.
No model weights, SDK or network are accessed by module import.
"""
import ast
import copy
import hashlib
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace

SOURCES = {
    'modeling_qwen3_5.py':'f567c3e4b7db57b5d04be57a8fe9788e7c19aaa01aad877ee0daf94b2dab1925',
    'cache_utils.py':'136b3d85b4496b6c4d614a676bdfd4f32e926b88eafe9a88c7cf1e173e8390b5',
    'activations.py':'5b20c0a3625edc0001a98f09ce3c6b5baa1100e1d7ad8dee649e4d45c8468665',
}
SELECT = {
    'activations.py': ['SiLUActivation','ClassInstantier'],
    'cache_utils.py':['CacheLayerMixin','DynamicLayer','LinearAttentionCacheLayerMixin','LinearAttentionLayer','Cache'],
    'modeling_qwen3_5.py':[
        'Qwen3_5TextRotaryEmbedding','Qwen3_5RMSNormGated','apply_mask_to_padding_states',
        'causal_conv1d_update','causal_conv1d_fn','l2norm','torch_chunk_gated_delta_rule',
        'torch_recurrent_gated_delta_rule','Qwen3_5GatedDeltaNet','rotate_half',
        'apply_rotary_pos_emb','repeat_kv','eager_attention_forward','Qwen3_5Attention',
        'Qwen3_5MLP','Qwen3_5RMSNorm','Qwen3_5DecoderLayer'],
}
OPTIONAL = {'use_kernel_forward_from_hub','use_kernel_func_from_hub_with_fallback',
            'use_kernelized_func','force_accelerate_hooks','deprecate_kwarg','dynamic_rope_update'}


def observations(node):
    """Add read-only callbacks; no existing statement or expression is modified."""
    inserted=[]
    for function in ast.walk(node):
        if not isinstance(function,ast.FunctionDef):continue
        body=[]
        for statement in function.body:
            body.append(statement)
            target=statement.targets[0].id if isinstance(statement,ast.Assign) and isinstance(statement.targets[0],ast.Name) else None
            callback=None
            if node.name=='eager_attention_forward' and target=='attn_weights' and isinstance(statement.value,ast.BinOp):
                callback="_observe_attention(attn_weights, attention_mask)"
            elif node.name in ('causal_conv1d_fn','causal_conv1d_update') and isinstance(statement,ast.If) and ast.unparse(statement.test)=='activation is not None':
                body.pop();body.append(ast.parse("_observe('conv_preactivation', out)").body[0]);body.append(statement)
                inserted.append('conv_preactivation')
            elif node.name=='Qwen3_5GatedDeltaNet' and function.name=='forward' and target=='g':
                callback="_observe_delta_gate(a, self.dt_bias, self.A_log, g)"
            if callback:
                body.append(ast.parse(callback).body[0]);inserted.append(callback)
        function.body=body
    return inserted


def decorator_name(node):
    if isinstance(node,ast.Call):return decorator_name(node.func)
    return node.id if isinstance(node,ast.Name) else ast.unparse(node)


def extract(source_folder,observe=True):
    result=[];record={}
    for file,names in SELECT.items():
        raw=(Path(source_folder)/file).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=SOURCES[file]:raise ValueError('Pinned official source changed: '+file)
        tree=ast.parse(raw);by_name={n.name:n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef))}
        for name in names:
            original=by_name[name];node=copy.deepcopy(original);removed=[];buffer_adaptations=[]
            for child in ast.walk(node):
                if isinstance(child,(ast.ClassDef,ast.FunctionDef,ast.AsyncFunctionDef)):
                    removed.extend(decorator_name(d) for d in child.decorator_list if decorator_name(d) in OPTIONAL)
                    child.decorator_list=[d for d in child.decorator_list if decorator_name(d) not in OPTIONAL]
            if name=='Qwen3_5TextRotaryEmbedding':
                constructor=next(n for n in node.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
                for i,statement in enumerate(constructor.body):
                    if not(isinstance(statement,ast.Assign) and isinstance(statement.value,ast.Call)
                           and ast.unparse(statement.value.func)=='nn.Buffer'):continue
                    target=statement.targets[0]
                    if not(isinstance(target,ast.Attribute) and isinstance(target.value,ast.Name) and target.value.id=='self'):
                        raise ValueError('Unexpected official buffer declaration')
                    constructor.body[i]=ast.Expr(value=ast.Call(func=ast.Attribute(value=ast.Name(id='self',ctx=ast.Load()),attr='register_buffer',ctx=ast.Load()),
                        args=[ast.Constant(target.attr),*statement.value.args],keywords=statement.value.keywords))
                    buffer_adaptations.append(target.attr)
                if buffer_adaptations!=['inv_freq','original_inv_freq']:raise ValueError('Expected two nonpersistent rotary buffers')
            inserted=observations(node) if observe else []
            result.append(node)
            record[file+':'+name]=dict(original_ast_sha256=hashlib.sha256(ast.dump(original).encode()).hexdigest(),
                selected_ast_sha256=hashlib.sha256(ast.dump(node).encode()).hexdigest(),removed_decorators=removed,
                torch24_register_buffer_adaptations=buffer_adaptations,read_only_callbacks=inserted)
        if file=='modeling_qwen3_5.py':
            forward=next(n for n in by_name['Qwen3_5TextModel'].body if isinstance(n,ast.FunctionDef) and n.name=='forward')
            node=copy.deepcopy(forward);node.name='text_model_forward';node.decorator_list=[];result.append(node)
            record[file+':Qwen3_5TextModel.forward']=dict(body_ast_sha256=hashlib.sha256(ast.dump(ast.Module(body=forward.body,type_ignores=[])).encode()).hexdigest(),
                adaptation='Function name and optional outer decorators only; body unchanged; caller provides original eager causal mask mapping and cache.')
    future=ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0)
    return ast.fix_missing_locations(ast.Module(body=[future,*result],type_ignores=[])),record


def load(source_folder,config,observe=True):
    import logging
    from abc import ABC,abstractmethod
    import torch
    from torch import nn
    if config.rope_parameters['rope_type']!='default':raise ValueError('Only pinned default RoPE supported')
    if config._attn_implementation!='eager':raise ValueError('Original eager attention required')
    class AttentionInterfaces:
        @staticmethod
        def get_interface(name,fallback):
            if name!='eager':raise ValueError('Unexpected attention backend')
            return fallback
    namespace=dict(torch=torch,nn=nn,F=torch.nn.functional,ABC=ABC,abstractmethod=abstractmethod,
        OrderedDict=OrderedDict,GradientCheckpointingLayer=nn.Module,
        ALL_ATTENTION_FUNCTIONS=AttentionInterfaces,
        is_torchdynamo_exporting=lambda:False,is_torchdynamo_compiling=lambda:False,
        maybe_autocast=torch.autocast,ROPE_INIT_FUNCTIONS={},logger=logging.getLogger('pinned_reference'),
        Qwen3_5ModelOutputWithPast=SimpleNamespace,
        _observe=lambda *args:None,_observe_attention=lambda *args:None,_observe_delta_gate=lambda *args:None)
    module,record=extract(source_folder,observe=observe)
    exec(compile(module,'pinned_official_text_reference','exec'),namespace)
    namespace['ACT2FN']=namespace['ClassInstantier']({'silu':namespace['SiLUActivation']})
    namespace['config']=config
    namespace['extraction']=record
    return namespace


def original_cache(namespace,layer_types):
    layers=[namespace['LinearAttentionLayer']() if kind=='linear_attention' else namespace['DynamicLayer']() for kind in layer_types]
    return namespace['Cache'](layers=layers,offloading=False)
