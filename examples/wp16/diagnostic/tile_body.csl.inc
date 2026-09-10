// Source proposal. Indices and exported lengths are defined in interfaces.json.
const receive_task=@get_local_task_id(10);const send_task=@get_local_task_id(11);
const transfer_ut=@get_ut_id(2);
const transfer_dst=@get_dsr(dsr_dest,5);const transfer_src=@get_dsr(dsr_src1,5);
var started=@zeros([3]u16);var ended=@zeros([3]u16);

fn fail(code:u32) void {
 state[0]=3;state[8]=code;handoff_state[0]=9;handoff_state[12]=code;
 mlp_state[12]=code;sys.unblock_cmd_stream();
}
fn mark(slot:u16) void {handoff_state[16]+=1;handoff_state[slot]=handoff_state[16];}
fn identity() bool {return control[0]==1 and control[1]==0 and control[2]==1 and control[7]==0;}
fn round_bf16(raw:u32) u16 {const odd:u32=(raw>>16)&1;return @as(u16,(raw+0x00007fff+odd)>>16);}
fn expand(raw:u16) f32 {return @bitcast(f32,@as(u32,raw)<<16);}
fn source(edge:u32) u16 {return @as(u16,edge);}
fn destination(edge:u32) u16 {return if(edge<2) @as(u16,2) else @as(u16,3);}
fn initialize() void {
 const total:u32=if(rank<2) @as(u32,5120) else @as(u32,128);
 if(state[1]!=0 or !identity() or control[3]!=total or control[4]!=0 or control[5]!=0 or control[6]!=0){fail(1);return;}
 state[1]=1;state[2]=0;state[3]=1;state[4]=total;state[14]=@as(u32,rank);
 handoff_state[1]=1;handoff_state[2]=0;handoff_state[3]=1;
 mlp_state[1]=1;mlp_state[2]=0;mlp_state[3]=1;mlp_state[14]=@as(u32,rank);
 wire_data[0]=0xa55aa55a;wire_data[141]=0x5aa55aa5;
 wire_ack[0]=0xa55aa55a;wire_ack[14]=0x5aa55aa5;
 if(rank!=2){
  ptr_weights_storage[0]=0xa55a;ptr_weights_storage[weight_words-1]=0x5aa5;
  ptr_input_storage[0]=77.5;ptr_input_storage[input_words-1]=-88.25;
  ptr_output_storage[0]=1234.5;ptr_output_storage[129]=-4321.25;
  ptr_bf16_storage[0]=0xa55a;ptr_bf16_storage[129]=0x5aa5;
  for(@range(u16,128))|i|{ptr_output_storage[1+i]=17.0;ptr_bf16_storage[1+i]=0x3f00;}
 }
 if(rank==2){
  ptr_mlp_gate[0]=0xa55a;ptr_mlp_gate[129]=0x5aa5;
  ptr_mlp_up[0]=0xa55a;ptr_mlp_up[129]=0x5aa5;
  ptr_mlp_silu[0]=0xa55a;ptr_mlp_silu[129]=0x5aa5;
  ptr_mlp_exp[0]=1234.5;ptr_mlp_exp[129]=-4321.25;
  ptr_mlp_sigmoid[0]=1234.5;ptr_mlp_sigmoid[129]=-4321.25;
  ptr_mlp_silu_fp32[0]=1234.5;ptr_mlp_silu_fp32[129]=-4321.25;
  ptr_mlp_product_fp32[0]=1234.5;ptr_mlp_product_fp32[129]=-4321.25;
 }
 if(rank>=2){ptr_mlp_product[0]=0xa55a;ptr_mlp_product[129]=0x5aa5;}
 sys.unblock_cmd_stream();
}
fn reset_sum(total:u32) void {
 state[0]=1;state[4]=total;state[5]=0;state[6]=0;state[7]=0;state[8]=0;
 state[9]=0;state[11]+=1;
 const yd=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]f32,&ptr_output_storage[1]),.extent=128});
 @fmovs(yd,0.0);
 for(@range(u16,timing_words))|i|{timing[i]=0;}
}
fn begin_projections() void {
 if(rank>=2){sys.unblock_cmd_stream();return;}
 if(!identity() or state[0]!=0 or state[11]!=0 or control[3]!=5120 or control[4]!=0 or control[5]!=0 or control[6]!=0){fail(2);return;}
 reset_sum(5120);sys.unblock_cmd_stream();
}
fn validate_tile(total:u32,columns:u32,tiles:u32) bool {
 if(!identity() or state[0]!=1 or control[3]!=total or state[4]!=total or
    control[4]!=state[5] or control[5]!=state[6] or state[5]>=tiles or state[6]>=total){return false;}
 const remain:u32=total-state[6];const valid:u32=if(remain>columns) columns else remain;
 return control[6]==valid;
}
fn apply_tile() bool {
 for(@range(u16,12))|i|{ptr_tile_guard_snapshot[i]=0;}
 ptr_tile_guard_snapshot[0]=@as(u32,ptr_weights_storage[0]);
 ptr_tile_guard_snapshot[1]=@as(u32,ptr_weights_storage[weight_words-1]);
 ptr_tile_guard_snapshot[4]=state[1];ptr_tile_guard_snapshot[5]=state[5];
 ptr_tile_guard_snapshot[6]=state[6];ptr_tile_guard_snapshot[7]=control[6];
 if(ptr_tile_guard_snapshot[0]!=0xa55a or ptr_tile_guard_snapshot[1]!=0x5aa5){fail(60);return false;}
 ptr_tile_guard_snapshot[10]=1;
 clock.enable_tsc();clock.get_timestamp(&started);
 gemv.compute(@ptrcast([*]u16,&ptr_weights_storage[1]),@ptrcast([*]f32,&ptr_input_storage[1]),
              @ptrcast([*]f32,&ptr_output_storage[1]),@as(u16,control[6]));
 clock.get_timestamp(&ended);clock.disable_tsc();
 const offset:u16=@as(u16,state[5])*6;
 for(@range(u16,3))|i|{timing[offset+i]=started[i];timing[offset+i+3]=ended[i];}
 ptr_tile_guard_snapshot[2]=@as(u32,ptr_weights_storage[0]);
 ptr_tile_guard_snapshot[3]=@as(u32,ptr_weights_storage[weight_words-1]);
 if(ptr_tile_guard_snapshot[2]!=0xa55a or ptr_tile_guard_snapshot[3]!=0x5aa5){fail(61);return false;}
 state[5]+=1;state[6]+=control[6];state[9]=control[6];state[10]+=1;
 ptr_tile_guard_snapshot[8]=state[5];ptr_tile_guard_snapshot[9]=state[6];
 ptr_tile_guard_snapshot[11]=1;return true;
}
fn accumulate_projections() void {
 if(rank>=2){sys.unblock_cmd_stream();return;}
 if(!validate_tile(5120,112,46)){fail(3);return;}
 if(!apply_tile()){return;}sys.unblock_cmd_stream();
}
fn commit_sum() void {
 for(@range(u16,128))|i|{ptr_bf16_storage[1+i]=round_bf16(@bitcast(u32,ptr_output_storage[1+i]));}
 ptr_weight_guard_snapshot[0]=@as(u32,ptr_weights_storage[0]);
 ptr_weight_guard_snapshot[1]=@as(u32,ptr_weights_storage[weight_words-1]);
 ptr_weight_guard_snapshot[2]=state[3];ptr_weight_guard_snapshot[3]=1;
 state[0]=2;state[7]=state[3];state[12]+=1;
 ptr_weight_guard_snapshot[4]=2;handoff_state[31]=1;
}
fn finalize_projections() void {
 if(rank>=2){sys.unblock_cmd_stream();return;}
 if(!identity() or state[0]!=1 or state[4]!=5120 or state[5]!=46 or state[6]!=5120 or control[3]!=5120){fail(4);return;}
 commit_sum();sys.unblock_cmd_stream();
}
fn reset_transfer() void {
 for(@range(u16,9))|i|{handoff_state[6+i]=0;}
 for(@range(u16,7))|i|{handoff_state[17+i]=0;}
 handoff_state[30]=0;handoff_state[4]=handoff_latched[3];
 for(@range(u16,5))|i|{handoff_state[24+i]=handoff_latched[4+i];}
}
fn receive_data() void {
 if(rank==2 and handoff_latched[3]==1){
  const fabric=@get_dsd(fabin_dsd,.{.input_queue=@get_input_queue(3),.extent=140});
  const memory=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]u32,&wire_data[1]),.extent=140});
  @load_to_dsr(transfer_dst,memory);
  @load_to_dsr(transfer_src,fabric,.{.async=true,.activate=receive_task,.ut_id=transfer_ut});
  @mov32(transfer_dst,transfer_src,.{.async=true});
 }else{
  const fabric=@get_dsd(fabin_dsd,.{.input_queue=@get_input_queue(2),.extent=140});
  const memory=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]u32,&wire_data[1]),.extent=140});
  @load_to_dsr(transfer_dst,memory);
  @load_to_dsr(transfer_src,fabric,.{.async=true,.activate=receive_task,.ut_id=transfer_ut});
  @mov32(transfer_dst,transfer_src,.{.async=true});
 }
}
fn arm_handoff() void {
 const edge:u32=handoff_control[3];
 if(!protocol.valid_descriptor(ptr_handoff_control,edge) or handoff_state[12]!=0){fail(10);return;}
 if(rank!=source(edge) and rank!=destination(edge)){sys.unblock_cmd_stream();return;}
 if(handoff_state[0]!=0 and handoff_state[0]!=4){fail(11);return;}
 if(rank==destination(edge)){
  const mask:u32=if(edge==1) @as(u32,1) else @as(u32,0);
  if(handoff_state[5]!=mask){fail(12);return;}
 }else{
  if(rank<2){if(state[0]!=2 or state[7]!=1){fail(13);return;}}
  else{if(mlp_state[0]!=2 or mlp_state[4]!=3 or mlp_state[11]!=1){fail(14);return;}}
 }
 protocol.latch(ptr_handoff_latched,ptr_handoff_control);reset_transfer();
 handoff_state[0]=1;mark(17);
 if(rank==destination(edge)){receive_data();}
 sys.unblock_cmd_stream();
}
fn finish_receiver() void {
 if(handoff_state[6]==1 and handoff_state[7]==1 and handoff_state[8]==1 and handoff_state[11]==0){
  handoff_state[11]=1;handoff_state[0]=if(handoff_state[12]==0) @as(u32,4) else @as(u32,9);
  mark(23);sys.unblock_cmd_stream();
 }
}
fn send_data() void {
 const memory=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]u32,&wire_data[1]),.extent=140});
 if(rank==2){
  const fabric=@get_dsd(fabout_dsd,.{.output_queue=@get_output_queue(4),.extent=140});
  @load_to_dsr(transfer_dst,fabric,.{.async=true,.activate=send_task,.ut_id=transfer_ut});
  @load_to_dsr(transfer_src,memory);@mov32(transfer_dst,transfer_src,.{.async=true});
 }else{
  const fabric=@get_dsd(fabout_dsd,.{.output_queue=@get_output_queue(2),.extent=140});
  @load_to_dsr(transfer_dst,fabric,.{.async=true,.activate=send_task,.ut_id=transfer_ut});
  @load_to_dsr(transfer_src,memory);@mov32(transfer_dst,transfer_src,.{.async=true});
 }
}
fn handoff() void {
 const edge:u32=handoff_control[3];
 if(rank!=source(edge) and rank!=destination(edge)){sys.unblock_cmd_stream();return;}
 if(handoff_state[6]!=0 or handoff_state[4]!=edge){fail(15);return;}
 if(rank==destination(edge)){
  if(handoff_state[0]!=1 and handoff_state[0]!=3 and handoff_state[0]!=9){fail(16);return;}
  handoff_state[6]=1;mark(18);finish_receiver();return;
 }
 if(handoff_state[0]!=1){fail(17);return;}
 handoff_state[0]=2;handoff_state[6]=1;mark(18);
 if(rank<2){protocol.make_data(ptr_wire_data,ptr_handoff_latched,@ptrcast([*]u16,&ptr_bf16_storage[1]));}
 else{protocol.make_data(ptr_wire_data,ptr_handoff_latched,@ptrcast([*]u16,&ptr_mlp_product[1]));}
 send_data();
}
fn receive_ack() void {
 const memory=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]u32,&wire_ack[1]),.extent=13});
 if(rank==2){
  const fabric=@get_dsd(fabin_dsd,.{.input_queue=@get_input_queue(4),.extent=13});
  @load_to_dsr(transfer_dst,memory);
  @load_to_dsr(transfer_src,fabric,.{.async=true,.activate=receive_task,.ut_id=transfer_ut});
  @mov32(transfer_dst,transfer_src,.{.async=true});
 }else{
  const fabric=@get_dsd(fabin_dsd,.{.input_queue=@get_input_queue(2),.extent=13});
  @load_to_dsr(transfer_dst,memory);
  @load_to_dsr(transfer_src,fabric,.{.async=true,.activate=receive_task,.ut_id=transfer_ut});
  @mov32(transfer_dst,transfer_src,.{.async=true});
 }
}
fn send_ack() void {
 const memory=@get_dsd(mem1d_dsd,.{.base_address=@ptrcast([*]u32,&wire_ack[1]),.extent=13});
 if(rank==2 and handoff_latched[3]==1){
  const fabric=@get_dsd(fabout_dsd,.{.output_queue=@get_output_queue(3),.extent=13});
  @load_to_dsr(transfer_dst,fabric,.{.async=true,.activate=send_task,.ut_id=transfer_ut});
  @load_to_dsr(transfer_src,memory);@mov32(transfer_dst,transfer_src,.{.async=true});
 }else{
  const fabric=@get_dsd(fabout_dsd,.{.output_queue=@get_output_queue(2),.extent=13});
  @load_to_dsr(transfer_dst,fabric,.{.async=true,.activate=send_task,.ut_id=transfer_ut});
  @load_to_dsr(transfer_src,memory);@mov32(transfer_dst,transfer_src,.{.async=true});
 }
}
task sent_complete() void {
 const edge:u32=handoff_latched[3];
 if(rank==source(edge)){
  if(handoff_state[0]!=2 or handoff_state[10]!=0){fail(18);return;}
  handoff_state[10]=1;handoff_state[13]=140;receive_ack();
 }else{
  if(rank!=destination(edge) or handoff_state[7]!=1 or handoff_state[8]!=0){fail(19);return;}
  handoff_state[8]=1;handoff_state[14]=13;mark(21);finish_receiver();
 }
}
task received_complete() void {
 const edge:u32=handoff_latched[3];
 if(rank==source(edge)){
  if(handoff_state[0]!=2 or handoff_state[10]!=1 or !protocol.valid_ack(ptr_wire_ack,ptr_handoff_latched)){fail(20);return;}
  handoff_state[9]=1;handoff_state[14]=13;mark(22);
  handoff_state[0]=4;handoff_state[11]=1;mark(23);sys.unblock_cmd_stream();return;
 }
 if(rank!=destination(edge) or handoff_state[0]!=1 or handoff_state[7]!=0){fail(21);return;}
 const error:u32=protocol.data_error(ptr_wire_data,ptr_handoff_latched);
 handoff_state[7]=1;handoff_state[13]=140;
 handoff_state[30]=if(handoff_state[6]==0) @as(u32,1) else @as(u32,0);mark(19);
 if(error==0){
  for(@range(u16,128))|i|{
   const value:u16=@as(u16,wire_data[13+i]);
   if(edge==0){ptr_mlp_gate[1+i]=value;}
   else if(edge==1){ptr_mlp_up[1+i]=value;}
   else{ptr_mlp_product[1+i]=value;}
  }
  handoff_state[5]|=@as(u32,1)<<@as(u16,edge);handoff_state[15]+=1;
  handoff_state[31]=1;mark(20);handoff_state[0]=3;
  mlp_state[4]=handoff_state[5];mlp_state[0]=1;
  if(rank==3){mlp_state[7]=128;}
 }else{handoff_state[12]=error;handoff_state[0]=9;mlp_state[12]=error;}
 protocol.make_ack(ptr_wire_ack,ptr_handoff_latched,error);send_ack();
}
fn evaluate_mlp() void {
 if(rank!=2){sys.unblock_cmd_stream();return;}
 if(handoff_state[0]!=4 or handoff_state[5]!=3 or handoff_state[12]!=0 or mlp_state[0]!=1 or mlp_state[11]!=0){fail(30);return;}
 state[0]=1;state[11]+=1;clock.enable_tsc();clock.get_timestamp(&started);
 const error:u32=nonlinear.compute(@ptrcast([*]u16,&ptr_mlp_gate[1]),@ptrcast([*]u16,&ptr_mlp_up[1]),
  @ptrcast([*]f32,&ptr_mlp_exp[1]),@ptrcast([*]f32,&ptr_mlp_sigmoid[1]),
  @ptrcast([*]f32,&ptr_mlp_silu_fp32[1]),@ptrcast([*]f32,&ptr_mlp_product_fp32[1]),
  @ptrcast([*]u16,&ptr_mlp_silu[1]),@ptrcast([*]u16,&ptr_mlp_product[1]));
 clock.get_timestamp(&ended);clock.disable_tsc();
 for(@range(u16,3))|i|{timing[i]=started[i];timing[i+3]=ended[i];}
 if(error!=0){fail(30+error);return;}
 state[0]=2;state[5]=1;state[6]=128;state[7]=1;state[9]=128;state[10]+=1;state[12]+=1;
 mlp_state[0]=2;mlp_state[5]=128;mlp_state[6]=128;mlp_state[10]=128;mlp_state[11]=1;
 sys.unblock_cmd_stream();
}
fn begin_down() void {
 if(rank!=3){sys.unblock_cmd_stream();return;}
 if(!identity() or state[0]!=0 or state[11]!=0 or control[3]!=128 or control[4]!=0 or control[5]!=0 or control[6]!=0 or
    handoff_state[0]!=4 or handoff_state[5]!=4 or handoff_state[12]!=0 or mlp_state[7]!=128){fail(40);return;}
 reset_sum(128);sys.unblock_cmd_stream();
}
fn accumulate_down() void {
 if(rank!=3){sys.unblock_cmd_stream();return;}
 if(!validate_tile(128,64,2) or mlp_state[7]!=128 or handoff_state[5]!=4){fail(41);return;}
 const offset:u16=@as(u16,state[6]);
 for(@range(u16,64))|i|{ptr_input_storage[1+i]=expand(ptr_mlp_product[1+offset+i]);}
 if(!apply_tile()){return;}mlp_state[8]=state[5];mlp_state[9]=state[6];sys.unblock_cmd_stream();
}
fn finalize_down() void {
 if(rank!=3){sys.unblock_cmd_stream();return;}
 if(!identity() or state[0]!=1 or state[4]!=128 or state[5]!=2 or state[6]!=128 or control[3]!=128){fail(42);return;}
 commit_sum();mlp_state[0]=2;mlp_state[10]=128;mlp_state[11]=1;sys.unblock_cmd_stream();
}
fn release_all() void {
 if(release_control[0]!=1 or release_control[1]!=0 or release_control[2]!=1 or release_control[3]!=1 or
    state[0]!=2 or state[7]!=1 or state[13]!=0 or handoff_state[0]!=4 or handoff_state[12]!=0){fail(50);return;}
 if(rank<2){if(handoff_state[9]!=1){fail(51);return;}}
 else if(rank==2){if(handoff_state[5]!=3 or handoff_state[9]!=1 or mlp_state[11]!=1){fail(52);return;}}
 else{if(handoff_state[5]!=4 or mlp_state[11]!=1 or state[6]!=128){fail(53);return;}}
 state[0]=4;state[13]+=1;handoff_state[0]=5;handoff_state[29]+=1;mlp_state[13]+=1;
 sys.unblock_cmd_stream();
}
