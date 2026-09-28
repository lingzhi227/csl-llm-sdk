"""Compose a full-key recurrent port with explicit projection scratch leases."""
import hashlib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def compose_port(source,role):
 if role=='compact_layer_projection.csl':
  workspace=b'@ptrcast([*]f32,&native.decoded)';values=b'@ptrcast([*]f32,&native.inputs)'
  idle=b'@assert(!running and !finish_requested);'
  guard=b'@comptime_assert(!fusion_actor and !mlp_sender and gdn_columns*2<=max_parts*columns);'
 elif role=='device_mixer_layer_mlp_standby.csl':
  workspace=b'@ptrcast([*]f32,&mixer.native.decoded)';values=b'@ptrcast([*]f32,&mixer.native.encoded)'
  idle=b'@assert(mixer.drained() and mixer_status[2]==0);';guard=b''
 else:raise ValueError('Unqualified recurrent cohost')
 port=(ROOT/'runtime/gdn_columns_port.cslpart').read_bytes()
 port=port.replace(b'GDN_WORKSPACE',workspace).replace(b'GDN_VALUES',values).replace(b'GDN_COHOST_IDLE',idle)
 result=source+b'\n'+port+b'\ncomptime {'+guard+b'}\n'
 return result,dict(role=role,original_sha256=hashlib.sha256(source).hexdigest(),adapted_sha256=hashlib.sha256(result).hexdigest(),
  input_words=387,output_words_per_value=5,fp32_state=True,local_dma_completion_is_not_downstream_retirement=True,
  scratch_lease='Previous projection and transfer drained; current MLP/mixer input must wait for recurrent return source retirement.',
  neural_executed=False,physical=False)
