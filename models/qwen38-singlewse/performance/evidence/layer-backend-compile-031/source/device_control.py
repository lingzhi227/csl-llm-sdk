"""Explicit internal-PE control adaptation, preserving all callable operations.

This is a local port prototype. Full-stage routes/gateway ownership are separate
lowering work; omission of SDK RPC is not proof of complete executable admission.
"""
import hashlib,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def adapt_mlp(source,forward_host_commands=False,routed=False):
 """Preserve the native MLP operations through a bounded typed device ABI.

 Setup has one zero padding halfword so routed paired reads cover all five
 existing fields. Segments3/7 also use16-bit transfers; callers must select
 this ABI explicitly rather than assume the mixer's segment table.
 """
 marker=b'const sys=@import_module("<memcpy/memcpy>",memcpy_params);'
 if source.count(marker)!=1 or source.count(b'var setup=@zeros([5]u16);')!=1:
    raise ValueError('Expected native MLP SDK endpoint and five-field setup')
 args=b'.command_task=@get_local_task_id(24)'
 if forward_host_commands:args+=b',.forward_host_commands=true'
 if routed:args+=b',.routed=true,.reply_rx=control_reply_rx,.reply_tx=control_reply_tx'
 result=source.replace(marker,b'const sys=@import_module("device_control.csl",.{'+args+b'});')
 result=result.replace(b'var setup=@zeros([5]u16);',b'var setup=@zeros([6]u16);')
 if routed:result=b'param control_reply_rx:direction;param control_reply_tx:direction;\n'+result
 result,count=re.subn(rb'@export_symbol\([^;]*\);',b'',result)
 if count!=13:raise ValueError('Unexpected MLP export surface')
 result+=b'\n'+(ROOT/'runtime/device_mlp_dispatch.cslpart').read_bytes()
 return result,dict(original_sha256=hashlib.sha256(source).hexdigest(),adapted_sha256=hashlib.sha256(result).hexdigest(),
    role='layer_projection.csl',exported_symbols_removed=count,entrypoints_preserved=['arm','start','finish','fusion_step'],
    segment_bits={str(i):16 if i in (1,3,7) else 32 for i in range(9)},setup_logical_halfwords=5,setup_storage_halfwords=6,
    bank_loader=True,bounded_readback=True,routed_port_mode=routed,full_stage_routed=False,executed=False)

def adapt_mixer(source,role,forward_host_commands=False,routed=False):
 if role not in ('mixer_layer_mlp_standby.csl','mixer_layer_norm_bridge.csl','mixer_layer_norm_sender.csl'):
    raise ValueError('Unqualified device-control role')
 marker=b'const sys=@import_module("<memcpy/memcpy>",memcpy_params);'
 if source.count(marker)!=1:raise ValueError('Expected exactly one SDK endpoint')
 result=source.replace(marker,b'const sys=@import_module("device_control.csl",.{.command_task=@get_local_task_id(24)});')
 if forward_host_commands:
  result=result.replace(b'.command_task=@get_local_task_id(24)}',b'.command_task=@get_local_task_id(24),.forward_host_commands=true}')
 if routed:
  result=b'param control_reply_rx:direction;param control_reply_tx:direction;\n'+result
  result=result.replace(b'.command_task=@get_local_task_id(24)',b'.command_task=@get_local_task_id(24),.routed=true,.reply_rx=control_reply_rx,.reply_tx=control_reply_tx')
 result,count=re.subn(rb'@export_symbol\([^;]*\);',b'',result)
 if count<8:raise ValueError('Unexpected cohost export surface')
 extra=b'';idle=b''
 if role=='mixer_layer_norm_bridge.csl':
  idle=b'@assert(!active);'
  extra=b'''else if(segment==5){pointer=rp;extent=64;writable=true;}
 else if(segment==6){pointer=gp;extent=128;writable=true;}
 else if(segment==7){pointer=qp;extent=64;writable=true;}
 else if(segment==8){pointer=ap;extent=4;}'''
 elif role=='mixer_layer_norm_sender.csl':
  idle=b'@assert(!active);'
  extra=b'''else if(segment==5){pointer=np;extent=64;}
 else if(segment==6){pointer=sa;extent=4;}
 else if(segment==8){pointer=ap;extent=4;}'''
 else:extra=b'else if(segment==8){pointer=ap;extent=4;}'
 dispatch=(ROOT/'runtime/device_mixer_dispatch.cslpart').read_bytes().replace(b'DEVICE_ROLE_SEGMENTS',extra).replace(b'DEVICE_ROLE_IDLE',idle)
 result+=b'\n'+dispatch
 return result,dict(original_sha256=hashlib.sha256(source).hexdigest(),adapted_sha256=hashlib.sha256(result).hexdigest(),
  role=role,exported_symbols_removed=count,entrypoints_preserved=['arm','start','finish','mixer_begin','mixer_consume'],
  bank_loader=True,bounded_readback=True,root_lease_requires_explicit_consume=True,routed_port_mode=routed,full_stage_routed=False,executed=False)
