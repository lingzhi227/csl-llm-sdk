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

def compose_mlp_shared_port(source):
 """Reuse native operand RX and result TX, including their existing callbacks.

 The two kernels use separate input/output queues and unchanged logical wire
 protocols, but one512B arena, RX/TX descriptor setup and completion dispatch.
 """
 def one(raw,before,after):
  if raw.count(before)!=1:raise ValueError('Shared native transport anchor changed')
  return raw.replace(before,after)
 result,proof=compose_port(source,'compact_layer_projection.csl')
 first=result.index(b'fn gdn_receive(');last=result.index(b'fn gdn_admit()',first)
 result=result[:first]+result[last:]
 result=one(result,b'const gdn_ds=@get_dsr(dsr_src0,5);const gdn_dd=@get_dsr(dsr_dest,5);\nconst gdn_task=@get_local_task_id(10);\n',b'')
 result=one(result,b'gdn_phase=1;gdn_receive(&gdn_header,3);',b'gdn_phase=1;receive_operand();')
 first=result.index(b' const m=@get_dsd(mem1d_dsd,.{.base_address=frame,.extent=5});',result.index(b'fn gdn_send()'))
 last=result.index(b'\n}',first)
 result=result[:first]+b' send_result_frame();'+result[last:]
 result=one(result,b'task gdn_completed() void {',b'fn gdn_completed() void {')
 result=one(result,b' if(gdn_phase==1){\n  @assert',b' if(gdn_phase==1){\n  const words=@ptrcast([*]u32,workspace);gdn_header[0]=words[0];gdn_header[1]=words[1];gdn_header[2]=words[2];\n  @assert')
 first=result.index(b' }else{\n  @assert(gdn_phase==5);',result.index(b'fn gdn_completed()'))
 last=result.index(b'comptime {',first)
 result=result[:first]+b''' }else{@assert(false);}
 gdn_phase+=1;receive_operand();
}
fn gdn_return_sent() void {
 if(comptime gdn_columns==1){gdn_phase=0;}
 else{gdn_output+=1;if(gdn_output<gdn_columns){gdn_send();}else{gdn_phase=0;}}
}
'''+result[last:]
 result=one(result,b' @bind_local_task(gdn_completed,gdn_task);\n',b'')
 # Dynamic length is shared by both protocols. Header also arrives into the
 # decoder and is copied before V replaces it; no extra header RX arena.
 result=one(result,b'const m=@get_dsd(mem1d_dsd,.{.base_address=operand,.extent=input_words});',
  b'const length:u16=if(gdn_phase==1) 3 else if(gdn_phase>1) 128 else input_words;\n const m0=@get_dsd(mem1d_dsd,.{.base_address=operand,.extent=1});const m=@set_dsd_length(m0,length);')
 result=one(result,b'const f=@get_dsd(fabin_dsd,.{.input_queue=iq,.extent=input_words});',
  b'const f0=@get_dsd(fabin_dsd,.{.input_queue=iq,.extent=1});const f=@set_dsd_length(f0,length);\n const g0=@get_dsd(fabin_dsd,.{.input_queue=gdn_iq,.extent=1});const g=@set_dsd_length(g0,length);')
 result=one(result,b'@load_to_dsr(id,m);@load_to_dsr(is,f,.{.async=true,.ut_id=@get_ut_id(2),.activate=it});@mov32(id,is,.{.async=true});',
  b'@load_to_dsr(id,m);\n if(gdn_phase>0){@load_to_dsr(is,g,.{.async=true,.ut_id=@get_ut_id(2),.activate=it});}\n else{@load_to_dsr(is,f,.{.async=true,.ut_id=@get_ut_id(2),.activate=it});}\n @mov32(id,is,.{.async=true});')
 result=one(result,b'task operand_arrived() void {',b'task operand_arrived() void {\n if(gdn_phase>0){gdn_completed();return;}')
 result=one(result,b'task sent() void {native.send_completed',b'task sent() void {if(gdn_phase==5){gdn_return_sent();return;}native.send_completed')
 result=one(result,b' @assert(!running);if(comptime mlp_sender)',b' @assert(gdn_local_idle() and !running);if(comptime mlp_sender)')
 # Both protocols fill the same frame arena before selecting their own OQ.
 first=result.index(b' const m=@get_dsd(mem1d_dsd,.{.base_address=outgoing,.extent=output_words});')
 last=result.index(b'\n}\nfn receive_child()',first)
 body=result[first:last]
 body=one(body,b'const m=@get_dsd(mem1d_dsd,.{.base_address=outgoing,.extent=output_words});',
  b'const m0=@get_dsd(mem1d_dsd,.{.base_address=outgoing,.extent=output_words});const m=if(gdn_phase==5) @set_dsd_length(m0,5) else m0;')
 body=one(body,b' if(comptime fused_root and boundary_rows>0){',
  b' if(gdn_phase==5){\n  const g=@get_dsd(fabout_dsd,.{.output_queue=gdn_oq,.extent=5});\n  @load_to_dsr(od,g,.{.async=true,.ut_id=@get_ut_id(5),.activate=ot});\n }else if(comptime fused_root and boundary_rows>0){')
 result=result[:first]+b' send_result_frame();\n}\nfn send_result_frame() void {\n const outgoing=@ptrcast([*]u32,&native.decoded);\n'+body+result[last:]
 proof.update(adapted_sha256=hashlib.sha256(result).hexdigest(),shared_native_transport=True,
  reused_resources=['native decoder512B','operand RX DSR2/UT2/task8','result TX DSR6/UT5/task11'],
  separate_new_queues=['GDN IQ7','GDN OQ6'],old_protocol_unchanged=True)
 return result,proof
