// One host stage call drives the whole spatial fragment. Other tiles are
// initialized once and accept device BEGIN messages, not per-operator commands.
const BEGIN:u32=1;const READY:u32=2;const DATA:u32=3;
const DONE:u32=4;const RELEASE:u32=5;const RELEASED:u32=6;
const participant_mask:u32=0xef;
const matrix=rank<4 or rank>=6;
const root=matrix and rank%2==0;
const ingress=rank==4;const nonlinear_tile=rank==5;
const columns:u16=if(rank>=6) @as(u16,64) else @as(u16,96);
const sys=@import_module("<memcpy/memcpy>",memcpy_params);
const clock=@import_module("<time>");
const kernel=@import_module(if(matrix) "gemv_kernel.csl" else "<empty>",if(matrix) .{.rows=128,.columns=columns} else .{});
const nonlinear=@import_module(if(nonlinear_tile) "nonlinear_kernel.csl" else "<empty>");
const line=@import_module(if(matrix) "line.csl" else "<empty>",if(matrix) .{.ordinal=rank%2,.participants=2,.length=128,.negative=WEST,.positive=EAST,.colors=[3]color{@get_color(0),@get_color(1),@get_color(2)},.task_id=@get_local_task_id(10),.on_complete=reduction_complete} else .{});
const input_task=@get_local_task_id(11);
const receiver=@import_module(if(matrix) "input.csl" else "<empty>",if(matrix) .{.length=columns,.input_color=@get_color(if(rank<4) 3 else 4),.completion_task=input_task} else .{});
const packets=@import_module("packets.csl",.{.sent_task_id=@get_local_task_id(12),.received_task_id=@get_local_task_id(13),.on_sent=sent,.on_received=received});
const dest=@get_dsr(dsr_dest,5);const source=@get_dsr(dsr_src1,5);
const broadcast_queue=@get_output_queue(if(ingress) 4 else 5);
var tx=@zeros([31]u32);
var started=@zeros([3]u16);var ended=@zeros([3]u16);
var tx_kind:u32=0;var tx_count:u16=0;
var begin_cursor:u16=0;var release_cursor:u16=0;
var output_count:u16=0;var gate_count:u16=0;var up_count:u16=0;
var data_offset:u16=0;
var ready_pending:bool=false;var data_ready:bool=false;
var work_done:bool=false;var done_sent:bool=false;var done_complete:bool=false;
var release_pending:bool=false;var next_begin_pending:bool=false;
var broadcast_started:bool=false;var command_waiting:bool=false;
var host_generation:u32=0;

fn expand(raw:u16) f32 {return @bitcast(f32,@as(u32,raw)<<16);}
fn round_bf16(raw:u32) u16 {return @as(u16,(raw+0x7fff+((raw>>16)&1))>>16);}
fn guard16(base:[*]u16,n:u16) void {if(n>1){base[0]=0xa55a;base[n-1]=0x5aa5;}}
fn guard32(base:[*]f32,n:u16) void {if(n>1){base[0]=1234.5;base[n-1]=-4321.25;}}
fn guards_ok() bool {
 if(comptime matrix){if(weights[0]!=0xa55a or weights[weights_count-1]!=0x5aa5 or input[0]!=1234.5 or input[input_count-1]!=-4321.25 or partial[0]!=1234.5 or partial[129]!=-4321.25 or reduced[0]!=1234.5 or reduced[129]!=-4321.25){return false;}}
 if(comptime root){if(rounded[0]!=0xa55a or rounded[129]!=0x5aa5){return false;}}
 if(comptime ingress){if(request[0]!=0xa55a or request[193]!=0x5aa5 or output[0]!=0xa55a or output[129]!=0x5aa5){return false;}}
 if(comptime nonlinear_tile){
  if(gate[0]!=0xa55a or gate[129]!=0x5aa5 or up[0]!=0xa55a or up[129]!=0x5aa5 or silu[0]!=0xa55a or silu[129]!=0x5aa5 or product[0]!=0xa55a or product[129]!=0x5aa5){return false;}
  if(exponential[0]!=1234.5 or exponential[129]!=-4321.25 or sigmoid[0]!=1234.5 or sigmoid[129]!=-4321.25 or activation_fp32[0]!=1234.5 or activation_fp32[129]!=-4321.25 or product_fp32[0]!=1234.5 or product_fp32[129]!=-4321.25){return false;}
 }
 return true;
}
fn initialize() void {
 @assert(state[31]==0);packets.init();state[31]=1;
 guard16(&weights,weights_count);guard16(&request,request_count);guard16(&output,output_count_storage);
 guard16(&rounded,rounded_count);guard16(&gate,gate_count_storage);guard16(&up,up_count_storage);
 guard16(&silu,silu_count);guard16(&product,product_count);
 guard32(&input,input_count);guard32(&partial,partial_count);guard32(&reduced,reduced_count);
 guard32(&exponential,exponential_count);guard32(&sigmoid,sigmoid_count);
 guard32(&activation_fp32,activation_fp32_count);guard32(&product_fp32,product_fp32_count);
 sys.unblock_cmd_stream();
}
fn clear_generation() void {
 @assert(guards_ok());
 if(comptime matrix){
  for(@range(u16,columns))|i|{input[1+i]=0.0;}
  for(@range(u16,128))|i|{partial[1+i]=0.0;reduced[1+i]=0.0;}
 }
 if(comptime root){for(@range(u16,128))|i|{rounded[1+i]=0;}}
 if(comptime ingress){for(@range(u16,128))|i|{output[1+i]=0;}}
 if(comptime nonlinear_tile){for(@range(u16,128))|i|{gate[1+i]=0;up[1+i]=0;silu[1+i]=0;product[1+i]=0;exponential[1+i]=0.0;sigmoid[1+i]=0.0;activation_fp32[1+i]=0.0;product_fp32[1+i]=0.0;}}
 output_count=0;gate_count=0;up_count=0;data_offset=0;
 ready_pending=false;data_ready=false;work_done=false;done_sent=false;done_complete=false;
 release_pending=false;broadcast_started=false;
 state[2]=0;state[3]=0;state[4]=0;state[5]=0;
 clock.enable_tsc();clock.get_timestamp(&started);
}
fn start_participant() void {
 @assert(!ingress and state[0]==0 and !packets.sending);
 state[1]+=1;clear_generation();state[0]=1;next_begin_pending=false;
 if(comptime matrix){receiver.arm(@ptrcast([*]f32,&input[1]));}
 ready_pending=true;
}
fn finish_participant_command() void {
 // BEGIN can arrive before the broadcast host RPC on this PE, and delivery of
 // RELEASED can precede its local send callback. Join both orderings before
 // admitting this PE's subsequent host diagnostics or next host stage RPC.
 if(comptime !ingress){
  if(command_waiting and state[0]==0 and state[9]==host_generation){command_waiting=false;sys.unblock_cmd_stream();}
 }
}
fn stage() void {
 @assert(state[31]==1 and !command_waiting);host_generation+=1;
 if(comptime ingress){
  @assert(state[0]==0 and !packets.sending and !command_waiting and state[1]<0xfffffffe);
  state[1]+=1;clear_generation();state[0]=1;begin_cursor=0;release_cursor=0;command_waiting=true;pump();
 }else{command_waiting=true;finish_participant_command();}
}
fn control(kind:u32,destination:u16) void {
 tx_kind=kind;tx_count=0;tx[0]=kind;tx[1]=state[1];tx[2]=@as(u32,rank);tx[3]=0;
 packets.send(destination%4,destination/4,&tx,4);
}
fn data_packet() void {
 if(comptime root){
  @assert(data_offset<64);tx_count=if(64-data_offset<27) 64-data_offset else 27;
  tx_kind=DATA;tx[0]=DATA;tx[1]=state[1];tx[2]=@as(u32,rank);tx[3]=@as(u32,data_offset);
  for(@range(u16,27))|i|{if(i<tx_count){const j:u16=1+2*(data_offset+i);tx[4+i]=@as(u32,rounded[j])|(@as(u32,rounded[j+1])<<16);}}
  const target:u16=if(rank==6) @as(u16,4) else @as(u16,5);
  packets.send(target%4,target/4,&tx,tx_count+4);
 }
}
fn start_broadcast() void {
 if(comptime ingress or nonlinear_tile){
  @assert(!broadcast_started);broadcast_started=true;
  if(comptime ingress){for(@range(u16,192))|i|{@assert((request[1+i]&0x7f80)!=0x7f80);broadcast[i]=expand(request[1+i]);}}
  else{for(@range(u16,128))|i|{broadcast[i]=expand(product[1+i]);}}
  const n:u16=if(ingress) @as(u16,192) else @as(u16,128);
  const mem=@get_dsd(mem1d_dsd,.{.base_address=&broadcast,.extent=n});
  const out=@get_dsd(fabout_dsd,.{.output_queue=broadcast_queue,.extent=n});
  @load_to_dsr(dest,out,.{.async=true,.activate=input_task});@load_to_dsr(source,mem);@mov32(dest,source,.{.async=true});
 }
}
fn finish_clock() void {
 clock.get_timestamp(&ended);clock.disable_tsc();
 for(@range(u16,3))|i|{timing[i]=started[i];timing[3+i]=ended[i];}
 @assert(guards_ok());state[9]+=1;
}
fn pump() void {
 if(packets.sending){return;}
 if(comptime ingress){
  if(state[0]==1){
   if(begin_cursor==4){begin_cursor+=1;}
   if(begin_cursor<8){const t=begin_cursor;begin_cursor+=1;control(BEGIN,t);return;}
   if(state[2]==participant_mask and !broadcast_started){start_broadcast();}
   if(state[5]==1 and state[3]==participant_mask and output_count==64){state[0]=2;}
  }
  if(state[0]==2){
   if(release_cursor==4){release_cursor+=1;}
   if(release_cursor<8){const t=release_cursor;release_cursor+=1;control(RELEASE,t);return;}
   if(state[4]==participant_mask){finish_clock();state[0]=0;command_waiting=false;sys.unblock_cmd_stream();}
  }
 }else{
  if(next_begin_pending and state[0]==0){start_participant();}
  if(ready_pending){ready_pending=false;control(READY,4);return;}
  if(data_ready and data_offset<64){data_packet();return;}
  if(work_done and !done_sent){done_sent=true;control(DONE,4);return;}
  // A delivery can precede its sender's callback. Keep RELEASE pending until
  // DONE's local completion; keep a following BEGIN pending during RELEASED.
  if(release_pending and done_complete and state[0]==1){
   finish_clock();state[0]=2;control(RELEASED,4);return;
  }
 }
}
task input_complete() void {
 @assert(state[0]==1);
 if(comptime matrix){kernel.compute(@ptrcast([*]u16,&weights[1]),@ptrcast([*]f32,&input[1]),@ptrcast([*]f32,&partial[1]));state[6]+=1;line.start(@ptrcast([*]f32,&partial[1]),@ptrcast([*]f32,&reduced[1]));}
 else{state[5]=1;if(comptime nonlinear_tile){work_done=true;}pump();}
}
fn reduction_complete() void {
 @assert(state[0]==1);state[7]+=1;
 if(comptime root){for(@range(u16,128))|i|{@assert((@bitcast(u32,reduced[1+i])&0x7f800000)!=0x7f800000);rounded[1+i]=round_bf16(@bitcast(u32,reduced[1+i]));@assert((rounded[1+i]&0x7f80)!=0x7f80);}data_ready=true;}
 else{work_done=true;}
 pump();
}
fn sent() void {
 if(tx_kind==DATA){data_offset+=tx_count;state[10]+=1;if(data_offset==64){work_done=true;}}
 else if(tx_kind==DONE){done_complete=true;}
 else if(tx_kind==RELEASED){state[0]=0;finish_participant_command();}
 pump();
}
fn receive_payload(data:[*]u32,length:u16,owner:u16) void {
 @assert(state[0]==1 and length>=5 and length<=31);
 const begin=@as(u16,data[3]);const n=length-4;@assert(data[3]<64 and begin+n<=64);
 if(comptime ingress){
  @assert(owner==6 and begin==output_count);
  for(@range(u16,27))|i|{if(i<n){output[1+2*(begin+i)]=@as(u16,data[4+i]&0xffff);output[2+2*(begin+i)]=@as(u16,data[4+i]>>16);}}
  output_count+=n;
 }else if(comptime nonlinear_tile){
  @assert((owner==0 and begin==gate_count) or (owner==2 and begin==up_count));
  for(@range(u16,27))|i|{if(i<n){const j:u16=1+2*(begin+i);if(owner==0){gate[j]=@as(u16,data[4+i]&0xffff);gate[j+1]=@as(u16,data[4+i]>>16);}else{up[j]=@as(u16,data[4+i]&0xffff);up[j+1]=@as(u16,data[4+i]>>16);}}}
  if(owner==0){gate_count+=n;}else{up_count+=n;}
  if(gate_count==64 and up_count==64){
   @assert(!broadcast_started);
   const error=nonlinear.compute(@ptrcast([*]u16,&gate[1]),@ptrcast([*]u16,&up[1]),@ptrcast([*]f32,&exponential[1]),@ptrcast([*]f32,&sigmoid[1]),@ptrcast([*]f32,&activation_fp32[1]),@ptrcast([*]f32,&product_fp32[1]),@ptrcast([*]u16,&silu[1]),@ptrcast([*]u16,&product[1]));
   state[30]=error;@assert(error==0);state[8]+=1;start_broadcast();
  }
 }else{@assert(false);}
 state[11]+=1;
}
fn received(data:[*]u32,length:u16) void {
 @assert(length>=4 and data[2]<8);const kind=data[0];const owner=@as(u16,data[2]);
 if(kind==BEGIN){
  @assert(!ingress and length==4 and owner==4 and data[3]==0 and data[1]==state[1]+1 and !next_begin_pending);
  @assert(state[0]==0 or (state[0]==2 and packets.sending and tx_kind==RELEASED));
  next_begin_pending=true;pump();return;
 }
 @assert(data[1]==state[1]);
 if(kind==DATA){receive_payload(data,length,owner);}
 else{
  @assert(length==4 and data[3]==0);
  if(comptime ingress){
   @assert(owner!=4);const bit:u32=@as(u32,1)<<owner;
   if(kind==READY){@assert(state[0]==1 and (state[2]&bit)==0);state[2]|=bit;}
   else if(kind==DONE){@assert(state[0]==1 and (state[2]&bit)!=0 and (state[3]&bit)==0);state[3]|=bit;}
   else if(kind==RELEASED){@assert(state[0]==2 and (state[3]&bit)!=0 and (state[4]&bit)==0);state[4]|=bit;}
   else{@assert(false);}
  }else{@assert(kind==RELEASE and owner==4 and state[0]==1 and done_sent and !release_pending);release_pending=true;}
 }
 pump();
}
comptime {
 if(ingress or nonlinear_tile){@initialize_queue(broadcast_queue,.{.color=@get_color(if(ingress) 3 else 4)});}
 @bind_local_task(input_complete,input_task);
 @export_symbol(initialize);@export_symbol(stage);
}
