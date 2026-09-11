// Included in the role-specialized tile after the common utility functions.
fn first_edge() u32 {return if(rank==1) @as(u32,1) else if(rank==3) @as(u32,2) else @as(u32,0);}
fn required_mask() u32 {return if(rank<2) @as(u32,7) else if(rank==2) @as(u32,31) else @as(u32,25);}
fn expected_input(generation:u32) u32 {return if(generation==1) @as(u32,0) else if(generation==2) @as(u32,1) else @as(u32,3);}
fn scope_valid(value:[*]u32,generation:u32) bool {
 return generation>=1 and generation<=3 and value[0]==generation and value[1]==expected_input(generation) and
  value[2]==generation and value[3]==0 and value[4]==112 and value[5]==0 and value[6]==128 and value[7]==0 and value[8]==128;
}
fn zero16(base:[*]u16,n:u16) bool {var i:u16=0;while(i<n){if(base[i]!=0){return false;}i+=1;}return true;}
fn zero32(base:[*]u32,n:u16) bool {var i:u16=0;while(i<n){if(base[i]!=0){return false;}i+=1;}return true;}
fn zerof32(base:[*]f32,n:u16) bool {var i:u16=0;while(i<n){if(@bitcast(u32,base[i])!=0){return false;}i+=1;}return true;}
fn payload16zero(base:[*]u16,n:u16) bool {return if(n==1) zero16(base,1) else zero16(@ptrcast([*]u16,&base[1]),n-2);}
fn payloadf32zero(base:[*]f32,n:u16) bool {return if(n==1) zerof32(base,1) else zerof32(@ptrcast([*]f32,&base[1]),n-2);}
fn guarded16(base:[*]u16,n:u16) bool {return if(n==1) base[0]==0 else base[0]==0xa55a and base[n-1]==0x5aa5;}
fn guardedf32(base:[*]f32,n:u16,left:f32,right:f32) bool {return if(n==1) @bitcast(u32,base[0])==0 else base[0]==left and base[n-1]==right;}
fn no_error() bool {return state[8]==0 and handoff_state[12]==0 and mlp_state[12]==0 and lifecycle_state[19]==0;}
fn completion_ok() bool {
 const g:u32=state[1];const receives:u32=if(rank==2) @as(u32,2) else if(rank==3) @as(u32,1) else @as(u32,0);
 const sends:u32=if(rank<3) @as(u32,1) else @as(u32,0);
 return no_error() and lifecycle_state[20]==0 and lifecycle_state[21]==0 and lifecycle_state[15]==3 and
  lifecycle_state[28]==required_mask() and lifecycle_state[22]==sends*g and lifecycle_state[23]==sends*g and
  lifecycle_state[24]==receives*g and lifecycle_state[25]==receives*g;
}
fn bind_empty_states(g:u32,input:u32,contraction:u32) void {
 state[1]=g;state[2]=input;state[3]=contraction;state[4]=if(rank<2) @as(u32,112) else @as(u32,128);
 state[14]=@as(u32,rank);state[16]=0;state[17]=112;state[18]=128;
 handoff_state[1]=g;handoff_state[2]=input;handoff_state[3]=contraction;
 handoff_state[25]=0;handoff_state[26]=128;handoff_state[27]=0;handoff_state[28]=128;
 handoff_state[32]=0;handoff_state[33]=112;
 mlp_state[1]=g;mlp_state[2]=input;mlp_state[3]=contraction;mlp_state[14]=@as(u32,rank);
 mlp_state[16]=0;mlp_state[17]=112;mlp_state[18]=0;mlp_state[19]=128;
}
fn weight_snapshot(reset:bool) void {
 ptr_weight_guard_snapshot[0]=@as(u32,ptr_weights_storage[0]);
 ptr_weight_guard_snapshot[1]=@as(u32,ptr_weights_storage[weight_words-1]);
 ptr_weight_guard_snapshot[2]=state[1];ptr_weight_guard_snapshot[3]=state[2];ptr_weight_guard_snapshot[4]=state[3];
 ptr_weight_guard_snapshot[5]=state[4];ptr_weight_guard_snapshot[6]=0;ptr_weight_guard_snapshot[7]=128;
 ptr_weight_guard_snapshot[8]=state[0];ptr_weight_guard_snapshot[9]=state[5];ptr_weight_guard_snapshot[10]=state[6];
 ptr_weight_guard_snapshot[11]=state[7];ptr_weight_guard_snapshot[12]=2*state[1]-(if(reset) @as(u32,1) else @as(u32,0));
 ptr_weight_guard_snapshot[13]=1;ptr_weight_guard_snapshot[14]=@as(u32,rank);ptr_weight_guard_snapshot[15]=0;
}
fn reset_check_mask() u32 {
 var mask:u32=0;
 if(payloadf32zero(ptr_output_storage,output_storage_count)){mask|=1;}
 if(payload16zero(ptr_bf16_storage,bf16_storage_count)){mask|=2;}
 if(payloadf32zero(ptr_input_storage,input_storage_count)){mask|=4;}
 if(payload16zero(ptr_mlp_gate,mlp_gate_count) and payload16zero(ptr_mlp_up,mlp_up_count)){mask|=8;}
 if(payload16zero(ptr_mlp_silu,mlp_silu_count) and payload16zero(ptr_mlp_product,mlp_product_count)){mask|=16;}
 if(payloadf32zero(ptr_mlp_exp,mlp_exp_count) and payloadf32zero(ptr_mlp_sigmoid,mlp_sigmoid_count) and
    payloadf32zero(ptr_mlp_silu_fp32,mlp_silu_fp32_count) and payloadf32zero(ptr_mlp_product_fp32,mlp_product_fp32_count)){mask|=32;}
 if(zero32(@ptrcast([*]u32,&state[5]),9) and zero32(@ptrcast([*]u32,&mlp_state[4]),10)){mask|=64;}
 if(zero32(@ptrcast([*]u32,&handoff_state[5]),19) and lifecycle_state[20]==0 and lifecycle_state[21]==0){mask|=128;}
 if(zero32(@ptrcast([*]u32,&wire_data[1]),142) and zero32(@ptrcast([*]u32,&wire_ack[1]),15) and
    zero32(ptr_handoff_latched,12) and zero32(ptr_handoff_control,12) and zero32(ptr_release_control,12)){mask|=256;}
 if(zero16(&timing,timing_count) and zero16(&started,3) and zero16(&ended,3)){mask|=512;}
 if(zero32(ptr_tile_guard_snapshot,tile_guard_snapshot_count)){mask|=1024;}
 if(state[7]==0 and mlp_state[11]==0 and handoff_state[15]==0 and handoff_state[29]==0 and mlp_state[13]==0){mask|=2048;}
 if(guarded16(ptr_bf16_storage,bf16_storage_count) and guarded16(ptr_mlp_gate,mlp_gate_count) and
    guarded16(ptr_mlp_up,mlp_up_count) and guarded16(ptr_mlp_silu,mlp_silu_count) and guarded16(ptr_mlp_product,mlp_product_count) and
    guardedf32(ptr_input_storage,input_storage_count,77.5,-88.25) and guardedf32(ptr_output_storage,output_storage_count,1234.5,-4321.25) and
    guardedf32(ptr_mlp_exp,mlp_exp_count,1234.5,-4321.25) and guardedf32(ptr_mlp_sigmoid,mlp_sigmoid_count,1234.5,-4321.25) and
    guardedf32(ptr_mlp_silu_fp32,mlp_silu_fp32_count,1234.5,-4321.25) and guardedf32(ptr_mlp_product_fp32,mlp_product_fp32_count,1234.5,-4321.25) and
    guarded16(ptr_weights_storage,weights_storage_count) and wire_data[0]==0xa55aa55a and wire_data[143]==0x5aa55aa5 and
    wire_ack[0]==0xa55aa55a and wire_ack[16]==0x5aa55aa5){mask|=4096;}
 return mask;
}
fn initialize() void {
 const total:u32=if(rank<2) @as(u32,112) else @as(u32,128);
 if(lifecycle_state[11]!=0 or control[0]!=0 or control[1]!=0xffffffff or control[2]!=0 or control[3]!=total or
    control[4]!=0 or control[5]!=0 or control[6]!=0 or control[7]!=0){fail(1);return;}
 clear_generation_buffers();
 if(rank!=2){ptr_weights_storage[0]=0xa55a;ptr_weights_storage[weight_words-1]=0x5aa5;}
 bind_empty_states(0,0xffffffff,0);
 lifecycle_state[1]=0;lifecycle_state[2]=0xffffffff;lifecycle_state[3]=0;
 lifecycle_state[4]=0;lifecycle_state[5]=112;lifecycle_state[6]=0;lifecycle_state[7]=128;
 lifecycle_state[8]=0;lifecycle_state[9]=128;lifecycle_state[11]=1;
 lifecycle_state[15]=first_edge();lifecycle_state[31]=@as(u32,rank);sys.unblock_cmd_stream();
}
fn begin_generation() void {
 const g:u32=generation_control[0];
 if(g<1 or g>3){fail(63);return;}
 if(!scope_valid(ptr_generation_control,g) or g!=lifecycle_state[11] or generation_control[9]!=g-1 or
    generation_control[10]!=1 or generation_control[11]!=0 or !no_error() or lifecycle_state[20]!=0 or lifecycle_state[21]!=0){fail(63);return;}
 if(g==1){
  if(lifecycle_state[0]!=0 or lifecycle_state[12]!=0 or lifecycle_state[14]!=0 or state[1]!=0){fail(64);return;}
 }else{
  if(lifecycle_state[0]!=3 or lifecycle_state[27]!=g-1 or lifecycle_state[14]!=g-1 or
     state[0]!=4 or state[13]!=1 or handoff_state[0]!=5 or handoff_state[29]!=1 or mlp_state[13]!=1 or !completion_ok()){fail(65);return;}
 }
 if(!guarded16(ptr_weights_storage,weights_storage_count)){fail(66);return;}
 clear_generation_buffers();bind_empty_states(g,generation_control[1],g);
 lifecycle_state[0]=1;
 for(@range(u16,9))|i|{lifecycle_state[1+i]=generation_control[i];}
 lifecycle_state[10]=g-1;lifecycle_state[11]=g+1;lifecycle_state[12]+=1;lifecycle_state[13]+=1;
 lifecycle_state[15]=first_edge();lifecycle_state[16]=0;lifecycle_state[26]=0;lifecycle_state[28]=0;
 lifecycle_state[29]=lifecycle_state[30]+1;
 if(rank<2){lifecycle_state[18]+=1;}
 if(rank!=2){weight_snapshot(true);}
 const mask:u32=reset_check_mask();lifecycle_state[16]=mask;lifecycle_state[17]+=1;
 if(mask!=8191){fail(67);return;}sys.unblock_cmd_stream();
}
fn release_all() void {
 const g:u32=state[1];
 if(!scope_valid(ptr_release_control,g) or release_control[9]!=required_mask() or release_control[10]!=g or release_control[11]!=0 or
    state[0]!=2 or state[7]!=g or state[13]!=0 or handoff_state[0]!=4 or lifecycle_state[0]!=2 or
    lifecycle_state[14]!=g-1 or lifecycle_state[27]!=g-1 or !completion_ok()){fail(50);return;}
 if(rank<2){if(handoff_state[9]!=1 or handoff_state[10]!=1){fail(51);return;}}
 else if(rank==2){if(handoff_state[5]!=3 or handoff_state[9]!=1 or handoff_state[10]!=1 or mlp_state[11]!=g){fail(52);return;}}
 else{if(handoff_state[5]!=4 or handoff_state[7]!=1 or handoff_state[8]!=1 or mlp_state[11]!=g or state[6]!=128){fail(53);return;}}
 state[0]=4;state[13]+=1;handoff_state[0]=5;handoff_state[29]+=1;mlp_state[13]+=1;
 lifecycle_state[0]=3;lifecycle_state[14]+=1;lifecycle_state[27]=g;lifecycle_state[30]=lifecycle_state[29]+1;
 sys.unblock_cmd_stream();
}
