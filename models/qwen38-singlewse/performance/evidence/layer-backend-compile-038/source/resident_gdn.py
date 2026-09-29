"""Compose the P47 wire protocol with the P45 resident cohosts.

This is code/resource lowering. It deliberately does not claim routed graph
execution: the production broadcast/gather paths must be admitted separately.
Outputs remain leased until their explicit successor-consumption event.
"""
import hashlib
from pathlib import Path

from spatial.gdn_columns import compose_controlled_port
from spatial.gdn_fusion import frontend, replace_one, switch_worker

ROOT = Path(__file__).resolve().parents[1]


def worker(base, previous, role):
    """Replace only the scalar port, keeping original projection/control code."""
    expected, _ = compose_controlled_port(base, role)
    if expected != previous:
        raise ValueError('Resident controlled-port source differs from P45')
    if role == 'compact_layer_projection.csl':
        workspace = b'@ptrcast([*]f32,&native.decoded)'
        values = b'@ptrcast([*]f32,&native.inputs)'
        idle = b'@assert(!running and !finish_requested);'
    elif role == 'device_mixer_layer_mlp_standby.csl':
        workspace = b'@ptrcast([*]f32,&mixer.native.decoded)'
        values = b'@ptrcast([*]f32,&mixer.native.encoded)'
        idle = b'@assert(mixer.drained() and mixer_status[2]==0);'
    else:
        raise ValueError('Unsupported resident cohost')
    def bind(raw):
        return (raw.replace(b'GDN_WORKSPACE', workspace)
                .replace(b'GDN_VALUES', values).replace(b'GDN_COHOST_IDLE', idle))
    scalar = bind((ROOT/'runtime/gdn_columns_port.cslpart').read_bytes())
    result = replace_one(previous, scalar, bind(switch_worker()))
    return result, dict(
        original_sha256=hashlib.sha256(previous).hexdigest(),
        adapted_sha256=hashlib.sha256(result).hexdigest(),
        projection_prefix_unchanged=True, bank_allocation_unchanged=True,
        scalar_port_replaced_by_qualified_paired_switch_port=True,
        state_arithmetic_unchanged=True,
        scratch_lease='Decoder and values are exclusive to GDN between predecessor '
                      'projection drain and final return-source DMA completion.',
        requires_global_admission_fence=True, routed=False, executed=False)


FRONTEND_TRANSPORT = b'''
// The packet DMA lease and the successor output lease retire independently.
param gdn_worker_count:u16;param gdn_send_color:u16;param gdn_return_color:u16;
const gdn_source_oq=@get_output_queue(3);const gdn_core_iq=@get_input_queue(3);
const gdn_tx_s=@get_dsr(dsr_src0,1);const gdn_tx_d=@get_dsr(dsr_dest,1);
const gdn_tx_done=@get_local_task_id(8);const gdn_marker_task=@get_control_task_id(40);
var gdn_next_head:u16=0;var gdn_markers:u16=0;var gdn_core_cursor:u16=0;
var gdn_core_frames:u16=0;var gdn_core_frame=@zeros([5]u32);
fn gdn_graph_drained() bool {
 return frontend.drained() and frontend_transport_drained() and gdn_core_cursor==0 and
  (frontend.invocation==0 or (gdn_markers==gdn_worker_count and gdn_next_head==3 and gdn_core_frames==192));
}
fn gdn_pump() void {
 if(gdn_next_head==3 or frontend.packet_busy){return;}
 const bit:u16=@as(u16,1)<<gdn_next_head;if((frontend.notified&bit)==0){return;}
 frontend.prepare(gdn_next_head);
 const source=@get_dsd(mem1d_dsd,.{.base_address=frontend.packet_pointer,.extent=387});
 const target=@get_dsd(fabout_dsd,.{.output_queue=gdn_source_oq,.extent=387});
 frontend.packet_launched(frontend.invocation,gdn_next_head);
 @load_to_dsr(gdn_tx_s,source);
 @load_to_dsr(gdn_tx_d,target,.{.async=true,.ut_id=@get_ut_id(1),.activate=gdn_tx_done});
 @mov32(gdn_tx_d,gdn_tx_s,.{.async=true});
}
task gdn_packet_sent() void {
 frontend.packet_consumed(frontend.invocation,gdn_next_head);gdn_next_head+=1;gdn_pump();
}
task gdn_returned(value:u32) void {
 @assert(gdn_core_cursor<5);gdn_core_frame[gdn_core_cursor]=value;gdn_core_cursor+=1;
 if(gdn_core_cursor==5){
  @assert(gdn_core_frame[0]==frontend.invocation and gdn_core_frame[2]==2 and gdn_core_frame[4]==5);
  @assert(gdn_core_frame[1]<6144);
  const row=@as(u16,gdn_core_frame[1]);
  @assert(row>=384*frontend_group and row<384*frontend_group+384 and row%2==0);
  frontend.accept(gdn_core_frame[0],16480+row,@as(u16,gdn_core_frame[3]&65535),@as(u16,gdn_core_frame[3]>>16));
  gdn_core_cursor=0;gdn_core_frames+=1;@assert(gdn_core_frames<=192);
  frontend_wire_packets+=1;@assert(frontend_wire_packets<=@as(u32,frontend_projected_packets)+192);
  if(frontend_wire_packets==@as(u32,frontend_projected_packets)+192){frontend_wire_active=false;}
 }
}
task gdn_marker() void {gdn_markers+=1;@assert(gdn_markers<=gdn_worker_count);}
comptime {
 @initialize_queue(gdn_source_oq,.{.color=@get_color(gdn_send_color)});
 @initialize_queue(gdn_core_iq,.{.color=@get_color(gdn_return_color)});
 @bind_local_task(gdn_packet_sent,gdn_tx_done);
 @bind_data_task(gdn_returned,@get_data_task_id(gdn_core_iq));
 @bind_control_task(gdn_marker,gdn_marker_task);
}
'''


def producer(cohost, arithmetic):
    """Automatic GDN launch, separate return queue, retained successor outputs."""
    arithmetic, proof = frontend(arithmetic)
    raw = cohost
    # Packet launch/consumption is now owned by this port, never a host command.
    for old in (
        b' else if(c[1]==9){@assert(c[2]<3);frontend.prepare(@as(u16,c[2]));sys.unblock_cmd_stream();}\n',
        b' else if(c[1]==10){@assert(c[3]<3);frontend.packet_consumed(c[2],@as(u16,c[3]));sys.unblock_cmd_stream();}\n',
    ):
        raw = replace_one(raw, old, b'')
    raw = replace_one(raw, b'fn frontend_begin(token:u32) void {\n @assert(frontend_transport_drained());',
                      b'fn frontend_begin(token:u32) void {\n @assert(gdn_graph_drained());gdn_next_head=0;gdn_markers=0;gdn_core_frames=0;')
    raw = replace_one(raw, b'fn frontend_reset() void {\n @assert(frontend_transport_drained());',
                      b'fn frontend_reset() void {\n @assert(gdn_graph_drained());gdn_next_head=0;gdn_markers=0;gdn_core_frames=0;')
    raw = raw.replace(b'frontend.drained() and frontend_transport_drained()', b'gdn_graph_drained()')
    raw = replace_one(raw, b'task frontend_packet_ready() void {frontend_snapshot();}',
                      b'task frontend_packet_ready() void {gdn_pump();}')
    # Gated outputs stay live. The retained explicit consume is the integration
    # port for output projection; a notification alone must not free this data.
    raw = replace_one(raw, b'(frontend_frame[4]<4 or frontend_frame[4]==5)', b'frontend_frame[4]<4')
    raw = replace_one(raw, b'}else if(matrix==1 or matrix==5){', b'}else if(matrix==1){')
    raw = replace_one(raw, b'index=10240+row;if(matrix==5){@assert(wanted);index=16480+row;}', b'index=10240+row;')
    raw += FRONTEND_TRANSPORT
    proof.update(
        original_cohost_sha256=hashlib.sha256(cohost).hexdigest(),
        adapted_cohost_sha256=hashlib.sha256(raw).hexdigest(),
        automatic_packet_launch=True, host_packet_lifetime_commands_removed=True,
        projected_input_rejects_core_frames=True, gated_output_lease_retained=True,
        queues={'projected_input': 1, 'core_return': 3, 'packet_output': 3},
        packet_tx={'dsr': 1, 'ut': 1, 'task': 8},
        extra_core_frame_bytes=20, diagnostic_packet_capture_bytes=0,
        routed=False, executed=False, full_layer=False)
    return raw, arithmetic, proof


def compact_worker(source):
    """Retain P47 arithmetic/wire format while sharing TX descriptor setup.

The first header word wakes a data task. Only then may a two-word header DMA
wait on fabric; there is no idle bulk receive. Source leases are unchanged.
"""
    raw = replace_one(source, b'var gdn_header_cursor:u16=0;\n', b'')
    raw = replace_one(raw, b'return gdn_phase==0 and gdn_header_cursor==0;', b'return gdn_phase==0;')
    raw = replace_one(raw,
        b' @assert(gdn_phase==0 and gdn_header_cursor<3);gdn_header[gdn_header_cursor]=value;gdn_header_cursor+=1;\n if(gdn_header_cursor==3){@block(gdn_header_task);gdn_phase=1;@activate(gdn_task);}',
        b' @assert(gdn_phase==0);@block(gdn_header_task);gdn_header[0]=value;gdn_phase=1;\n gdn_receive(@ptrcast([*]u32,&gdn_header[1]),2);')
    raw = replace_one(raw,
        b'if(gdn_phase==6){gdn_phase=0;gdn_header_cursor=0;gdn_admit();return;}',
        b'if(gdn_phase==6){gdn_phase=0;@unblock(gdn_header_task);return;}')
    raw = replace_one(raw,
        b' const m=@get_dsd(mem1d_dsd,.{.base_address=frame,.extent=5});\n const f=@get_dsd(fabout_dsd,.{.output_queue=gdn_oq,.extent=5});\n @load_to_dsr(gdn_ds,m);@load_to_dsr(gdn_dd,f,.{.async=true,.ut_id=@get_ut_id(7),.activate=gdn_task});\n @mov32(gdn_dd,gdn_ds,.{.async=true});',
        b' gdn_flush();')
    raw = replace_one(raw,
        b'   const f=@get_dsd(fabout_dsd,.{.output_queue=gdn_oq,.extent=1,.control=true});\n', b'')
    raw = replace_one(raw,
        b'   const m=@get_dsd(mem1d_dsd,.{.base_address=word,.extent=1});@load_to_dsr(gdn_ds,m);\n   @load_to_dsr(gdn_dd,f,.{.async=true,.ut_id=@get_ut_id(7),.activate=gdn_task});\n   @mov32(gdn_dd,gdn_ds,.{.async=true});',
        b'   gdn_flush();')
    raw += b'''
fn gdn_flush() void {
 const word=@ptrcast([*]u32,gdn_workspace());
 const m0=@get_dsd(mem1d_dsd,.{.base_address=word,.extent=5});
 const f=@get_dsd(fabout_dsd,.{.output_queue=gdn_oq,.extent=5});
 const c=@get_dsd(fabout_dsd,.{.output_queue=gdn_oq,.extent=1,.control=true});
 const control=gdn_phase==6;
 @load_to_dsr(gdn_ds,if(control) @set_dsd_length(m0,1) else m0);
 @load_to_dsr(gdn_dd,if(control) c else f,.{.async=true,.ut_id=@get_ut_id(7),.activate=gdn_task});
 @mov32(gdn_dd,gdn_ds,.{.async=true});
}
'''
    return raw


def compact_producer(source):
    """Outstanding debts make the initial idle state explicit and smaller."""
    raw = source.replace(b'gdn_core_frames', b'gdn_core_left')
    raw = replace_one(raw, b'var gdn_next_head:u16=0;', b'var gdn_next_head:u16=3;')
    raw = replace_one(raw,
        b'(frontend.invocation==0 or (gdn_markers==gdn_worker_count and gdn_next_head==3 and gdn_core_left==192))',
        b'gdn_markers==0 and gdn_next_head==3 and gdn_core_left==0')
    raw = replace_one(raw,
        b'fn frontend_begin(token:u32) void {\n @assert(gdn_graph_drained());gdn_next_head=0;gdn_markers=0;gdn_core_left=0;',
        b'fn frontend_begin(token:u32) void {\n @assert(gdn_graph_drained());gdn_next_head=0;gdn_markers=gdn_worker_count;gdn_core_left=192;')
    raw = replace_one(raw,
        b'fn frontend_reset() void {\n @assert(gdn_graph_drained());gdn_next_head=0;gdn_markers=0;gdn_core_left=0;',
        b'fn frontend_reset() void {\n @assert(gdn_graph_drained());gdn_next_head=3;gdn_markers=0;gdn_core_left=0;')
    raw = replace_one(raw, b'gdn_core_cursor=0;gdn_core_left+=1;@assert(gdn_core_left<=192);',
                      b'gdn_core_cursor=0;@assert(gdn_core_left>0);gdn_core_left-=1;')
    raw = replace_one(raw, b'task gdn_marker() void {gdn_markers+=1;@assert(gdn_markers<=gdn_worker_count);}',
                      b'task gdn_marker() void {@assert(gdn_markers>0);gdn_markers-=1;}')
    return raw
