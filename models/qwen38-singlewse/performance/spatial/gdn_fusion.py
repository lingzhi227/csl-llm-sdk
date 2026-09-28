"""Direct frontend/GDN composition with distinct source and sink leases.

This does not admit the complete resident layer. Layout lowering must separately
prove routes, switch capability, scratch ownership and the full ELF census.
"""
import hashlib
from spatial.gdn_columns import compose_frontend_packet, paired_port


def replace_one(raw, old, new):
    if raw.count(old) != 1:
        raise ValueError('Frozen fusion composition anchor changed')
    return raw.replace(old, new)


def frontend(source):
    """Permit early returns, but retain the packet until its DMA callback."""
    result, proof = compose_frontend_packet(source)
    result = replace_one(result, b'fn packet_consumed(token:u32,head:u16) void {', b'''fn packet_launched(token:u32,head:u16) void {
 @assert(active and packet_busy and token==invocation and head==packet_head);
 const bit:u16=@as(u16,1)<<head;@assert((consumed&bit)==0);
 consumed|=bit;audit[5]+=1;
}
fn maybe_retire() void {
 if(retired==7 and !packet_busy){@assert(audit[1]==710 and consumed==7);active=false;audit[7]=1;}
}
fn packet_consumed(token:u32,head:u16) void {''')
    result = replace_one(result, b' packet_busy=false;consumed|=@as(u16,1)<<head;audit[5]+=1;',
                         b' @assert((consumed&(@as(u16,1)<<head))!=0);packet_busy=false;maybe_retire();')
    result = replace_one(result, b' if(retired==7){@assert(audit[1]==710 and consumed==7 and !packet_busy);active=false;audit[7]=1;}',
                         b' maybe_retire();')
    proof.update(adapted_sha256=hashlib.sha256(result).hexdigest(),
                 source_launch_precedes_return_eligibility=True,
                 source_dma_and_output_retirement_may_arrive_in_either_order=True,
                 routing_and_lifecycle_qualified=False)
    return result, proof


def switch_worker():
    """Emit own pairs, then open the hardware path for the following worker.

The switch pops only its own ADV. Upstream routers see NOP, so their selected
child input stays open. The receiver must count every terminal control wavelet
as well as every data frame before issuing another invocation. A next header
is then a downstream fence for resetting the local return switch.
"""
    result = paired_port()
    result = replace_one(result, b'var gdn_header=@zeros([3]u32);', b'''const gdn_tile=@import_module("<tile_config>");const gdn_control=@import_module("<control>");
const gdn_return_task=@get_control_task_id(40);const gdn_header_task=@get_data_task_id(gdn_iq);
var gdn_header_cursor:u16=0;
var gdn_header=@zeros([3]u32);''')
    result = replace_one(result, b'fn gdn_local_idle() bool {return gdn_phase==0;}',
                         b'fn gdn_local_idle() bool {return gdn_phase==0 and gdn_header_cursor==0;}')
    result = replace_one(result, b' gdn_phase=1;gdn_receive(&gdn_header,3);', b''' @unblock(gdn_header_task);
}
task gdn_header_arrived(value:u32) void {
 @assert(gdn_phase==0 and gdn_header_cursor<3);gdn_header[gdn_header_cursor]=value;gdn_header_cursor+=1;
 if(gdn_header_cursor==3){@block(gdn_header_task);gdn_phase=1;@activate(gdn_task);}''')
    result = replace_one(result, b'  @assert(gdn_header[0]>gdn_invocation);gdn_invocation=gdn_header[0];', b'''  @assert(gdn_header[0]>gdn_invocation);gdn_invocation=gdn_header[0];
  // The producer issued this header only after all prior data AND markers.
  gdn_tile.switch_config.clear_current_position(@get_color(gdn_output_color));''')
    result = replace_one(result, b'  @assert(gdn_phase==5);gdn_output+=2;\n  if(gdn_output<gdn_columns){gdn_send();}else{gdn_phase=0;}return;', b'''  if(gdn_phase==6){gdn_phase=0;gdn_admit();return;}
  @assert(gdn_phase==5);gdn_output+=2;
  if(gdn_output<gdn_columns){gdn_send();}else{
   gdn_phase=6;
   const f=@get_dsd(fabout_dsd,.{.output_queue=gdn_oq,.extent=1,.control=true});
   const word=@ptrcast([*]u32,workspace);
   word[0]=gdn_control.encode_payload(1,[1]gdn_control.opcode{gdn_control.opcode.SWITCH_ADV},[1]bool{false},false,gdn_return_task);
   const m=@get_dsd(mem1d_dsd,.{.base_address=word,.extent=1});@load_to_dsr(gdn_ds,m);
   @load_to_dsr(gdn_dd,f,.{.async=true,.ut_id=@get_ut_id(7),.activate=gdn_task});
   @mov32(gdn_dd,gdn_ds,.{.async=true});
  }return;''')
    result = replace_one(result, b'if(gdn_phase==6){gdn_phase=0;gdn_admit();return;}',
                         b'if(gdn_phase==6){gdn_phase=0;gdn_header_cursor=0;gdn_admit();return;}')
    result = replace_one(result, b' @bind_local_task(gdn_completed,gdn_task);',
                         b' @bind_local_task(gdn_completed,gdn_task);\n @bind_data_task(gdn_header_arrived,gdn_header_task);@block(gdn_header_task);')
    return result
