"""Compose pinned qualified arithmetic with the candidate global resident bus.

Generated source is not acceptance evidence. The representative layout compiles
all roles; it cannot execute the whole model or qualify whole-wafer routing.
"""
import hashlib,json,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'resident/generated'

def actor(kind):
    source=ROOT/'experiments'/kind/'pe.csl';raw=source.read_text()
    prefix='g' if kind=='gdn_head' else 'a'
    raw=raw[raw.index('const kernel=' if prefix=='g' else 'const qwen='):]
    raw=re.sub(r'const qwen=@import_module\("qwen_math.csl"\);\n','',raw)
    raw=re.sub(r'@export_symbol\([^;]+;','',raw)
    names=set(re.findall(r'\b(?:const|var|fn|task)\s+(\w+)',raw))|{'role'}
    # Names inside functions also get a prefix, preserving lexical binding.
    raw=re.sub(r'(?<!\.)\b(?:'+ '|'.join(sorted(names,key=len,reverse=True))+r')\b',lambda m:prefix+'_'+m[0],raw)
    raw=raw.replace('@get_input_queue(2)','@get_input_queue(3)').replace('@get_input_queue(3);\nconst '+prefix+'_oq','@get_input_queue(3);\nconst '+prefix+'_oq')
    raw=raw.replace('@get_output_queue(2)','@get_output_queue(3)')
    result_name='g_result_iq' if prefix=='g' else 'a_return_iq'
    raw=raw.replace('const '+result_name+'=@get_input_queue(3);','const '+result_name+'=@get_input_queue(6);')
    for old,new in [(8,14),(9,15),(10,16)]:raw=raw.replace(f'@get_local_task_id({old})',f'@get_local_task_id({new})')
    raw=raw.replace('@get_dsr(dsr_dest,6)','@get_dsr(dsr_dest,7)').replace('@get_dsr(dsr_src1,6)','@get_dsr(dsr_src1,7)')
    if prefix=='g':
        raw=raw.replace('g_calls[0]+=1;sys.unblock_cmd_stream();','g_calls[0]+=1;actor_finished();')
        raw=raw.replace('g_calls[0]=0;sys.unblock_cmd_stream();','g_calls[0]=0;')
        raw=raw.replace('if(g_role==0) 3 else if(g_role==1) 5 else 4','if(g_role==0) 12 else if(g_role==1) 14 else 13')
        raw=raw.replace('@get_color(3)','@get_color(12)').replace('if(g_role==0) 5 else 4','if(g_role==0) 14 else 13')
        extra='''
fn actor_scratch() [*]f32 {return &g_packet;}
fn actor_compute(position:u16) void {@assert(g_calls[0]==@as(u32,position));g_compute();}
fn actor_reset() void {g_reset();}
fn actor_output() [*]u16 {return &g_gated;}fn actor_output_capacity() u16 {return if(role==6) 128 else 0;}
fn data_buffer() [*]u16 {return &g_input;}fn data_capacity() u16 {return if(role==6) 514 else 0;}
fn conv_data() [*]u16 {return &g_conv_weights;}fn gate_data() [*]u16 {return &g_gate_parameters;}
fn gain_data() [*]u16 {return &g_gain;}fn frequency_data() [*]f32 {return &stub_f32;}
'''
        raw='const g_role:u16=role-6;\n'+raw
    else:
        raw=raw.replace('sys.unblock_cmd_stream();','actor_finished();')
        raw=raw.replace('for(@range(u16,6))|a_i|{a_trace[a_i]=0;}actor_finished();','for(@range(u16,6))|a_i|{a_trace[a_i]=0;}')
        # Standalone SDK startup is unused; reset/compute are driven by global frames.
        raw=re.sub(r'fn a_start\(\) void \{[^\n]+\}\n','',raw)
        raw=raw.replace('if(a_role==0) 3 else 4+a_role%2','if(a_role==0) 12 else 13+a_role%2')
        raw=raw.replace('@get_color(3)','@get_color(12)').replace('4+(a_role+1)%2','13+(a_role+1)%2')
        # Shards arm their local receiver on global compute, then stay on the
        # local protocol until its END. All group members hold global receive.
        raw=raw.replace('a_pending=true;a_finish_request();','a_pending=true;a_arm();')
        raw=raw.replace('a_done=true;a_arm();a_finish_request();','a_done=true;a_finish_request();')
        extra='''
fn actor_scratch() [*]f32 {return &a_joined;}
fn actor_compute(position:u16) void {if(comptime role==9){@assert(a_position==position);}a_compute();}
fn actor_reset() void {a_reset();}
fn actor_output() [*]u16 {return &a_output;}fn actor_output_capacity() u16 {return if(role==9) 1536 else 0;}
fn data_buffer() [*]u16 {return &a_input;}fn data_capacity() u16 {return if(role==9) 3584 else 0;}
fn conv_data() [*]u16 {return &stub_u16;}fn gate_data() [*]u16 {return &stub_u16;}
fn gain_data() [*]u16 {return &a_gain;}fn frequency_data() [*]f32 {return &a_frequencies;}
'''
        raw='const a_role:u16=role-9;\n'+raw
    assert 'sys.unblock_cmd_stream' not in raw
    return raw+extra,dict(path=str(source.relative_to(ROOT)),sha256=hashlib.sha256(source.read_bytes()).hexdigest())

ROOT_STUB='''
fn root_buffer() [*]f32 {return &stub_f32;}fn root_tick() void {}fn root_sent() void {}
fn root_begin() void {}fn root_step() void {}
comptime {
  @export_symbol(@ptrcast([*]u16,&stub_u16),"program");@export_symbol(@ptrcast([*]u16,&stub_u16),"matrices");
  @export_symbol(@ptrcast([*]u16,&stub_u16),"io");@export_symbol(@ptrcast([*]u32,&stub_u32),"prompt");
  @export_symbol(@ptrcast([*]u32,&stub_u32),"controls");@export_symbol(@ptrcast([*]u32,&stub_u32),"generated");
  @export_symbol(@ptrcast([*]u32,&stub_u32),"result");
}
'''
ACTOR_STUB='''
fn actor_scratch() [*]f32 {return &stub_f32;}fn actor_compute(position:u16) void {}fn actor_reset() void {}
fn actor_output() [*]u16 {return &stub_u16;}fn actor_output_capacity() u16 {return 0;}
fn conv_data() [*]u16 {return &stub_u16;}fn gate_data() [*]u16 {return &stub_u16;}
fn gain_data() [*]u16 {return &stub_u16;}fn frequency_data() [*]f32 {return &stub_f32;}
'''
ACTOR_STORAGE='''
fn weight_data() [*]u16 {return &stub_u16;}fn scale_data() [*]f32 {return &stub_f32;}
fn matrix_output() [*]f32 {return &stub_f32;}
'''
EXPORTS='''
comptime {
 @export_symbol(conv_data(),"conv");@export_symbol(gate_data(),"gate");
 @export_symbol(gain_data(),"gain");@export_symbol(frequency_data(),"freq");
}
'''

def layout(cells,width,height):
    assert len(cells)==width*height
    lines=[f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={width},.height={height}}});','layout {',f'  @set_rectangle({width},{height});']
    for y in range(height):
        for x in range(width):
            s=cells[y*width+x];role=s['role'];name='gdn' if 6<=role<=8 else 'attention' if 9<=role<=13 else 'controller' if role==14 else 'storage'
            args=dict(role=role,phase=s.get('phase',0),first=s.get('first',False),last=s.get('last',False),west=x==0,east=x==width-1,north=y==0,south=y==height-1,parity_x=x%2,parity_y=y%2)
            params=','.join('.'+k+'='+str(v).lower() for k,v in args.items())
            lines.append(f'  @set_tile_code({x},{y},"{name}.csl",.{{.memcpy_params=memcpy.get_params({x}),{params}}});')
            def route(color,rx,tx):lines.append(f'  @set_color_config({x},{y},@get_color({color}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}}});')
            if y==height-1:
                if x==width-1:route(3,'RAMP','WEST,NORTH')
                elif x==0:route(3,'EAST','RAMP,NORTH')
                else:route(3,'EAST','RAMP,WEST,NORTH')
            elif y==0:route(3,'SOUTH','RAMP')
            else:route(3,'SOUTH','RAMP,NORTH')
            if role in (1,2):
                if not s['first']:route(4+(x+1)%2,'WEST','RAMP')
                if not s['last']:route(4+x%2,'RAMP','EAST')
            if x>0:route(8+(x+1)%2,'WEST','RAMP')
            if x<width-1:route(8+x%2,'RAMP','EAST')
            if x==width-1:
                if y>0:route(10+(y+1)%2,'NORTH','RAMP')
                if y<height-1:route(10+y%2,'RAMP','SOUTH')
            if 6<=role<=8:
                n=role-6
                route(12,'RAMP' if n==0 else 'WEST','EAST' if n==0 else 'RAMP,EAST' if n==1 else 'RAMP')
                if n>0:route(15-n,'RAMP','WEST')
                if n<2:route(14-n,'EAST','RAMP')
            if 9<=role<=13:
                n=role-9;route(12,'RAMP' if n==0 else 'WEST','EAST' if n==0 else 'RAMP' if n==4 else 'RAMP,EAST')
                if n>0:route(13+n%2,'RAMP','WEST')
                if n<4:route(13+(n+1)%2,'EAST','RAMP')
    for name in ['weights','data','conv','gate','gain','program','matrices','io']:lines.append(f'  @export_name("{name}",[*]u16,false);')
    for name in ['scales','freq']:lines.append(f'  @export_name("{name}",[*]f32,false);')
    for name in ['config','trace','prompt','controls','generated','result']:lines.append(f'  @export_name("{name}",[*]u32,false);')
    for name in ['start','begin','step']:lines.append(f'  @export_name("{name}",fn()void);')
    return '\n'.join(lines+['}'])+'\n'

def build():
    OUT.mkdir(parents=True,exist_ok=True);common=(ROOT/'resident/common.cslpart').read_text()
    storage=(ROOT/'resident/storage.cslpart').read_text();stub='var stub_u16=@zeros([1]u16);var stub_u32=@zeros([1]u32);var stub_f32=@zeros([1]f32);\n'
    provenance={}
    for name in ['storage','controller','gdn','attention']:
        body=common+stub
        if name in ['gdn','attention']:
            raw,receipt=actor('gdn_head' if name=='gdn' else 'attention_group');body+=raw+ACTOR_STORAGE;provenance[name]=receipt
        else:body+=storage+ACTOR_STUB
        body+=(ROOT/'resident/controller.cslpart').read_text() if name=='controller' else ROOT_STUB
        body+=EXPORTS
        aliases=[]
        def export_alias(match):
            expression,symbol=match.groups()
            expression=re.sub(r'@ptrcast\(\[\*\](?:u16|u32|f32),(&\w+)\)',r'\1',expression)
            kind='f32' if symbol in ['scales','freq'] else 'u32' if symbol in ['config','trace','prompt','controls','generated','result'] else 'u16'
            alias='export_'+symbol;aliases.append(f'const {alias}:[*]{kind}={expression};')
            return f'@export_symbol({alias},"{symbol}")'
        body=re.sub(r'@export_symbol\((.*?),"(\w+)"\)',export_alias,body)
        body+='\n'+'\n'.join(aliases)+'\n';(OUT/(name+'.csl')).write_text(body)
    cells=[dict(role=0) for _ in range(144)]
    for y in range(9):
        for x in range(4):cells[y*12+x]=dict(role=1 if y<8 else 2,phase=(y*16)%128,first=x==0,last=x==3)
    for x in range(6):cells[9*12+x]=dict(role=6+x%3)
    for x in range(5):cells[10*12+x]=dict(role=9+x)
    for x,r in [(5,4),(6,4),(7,5),(8,5),(9,3),(10,3)]:cells[10*12+x]=dict(role=r)
    cells[-1]=dict(role=14)
    (OUT/'layout.csl').write_text(layout(cells,12,12))
    (OUT/'roles.json').write_text(json.dumps(dict(scope='Representative role SRAM compile only',width=12,height=12,cells=cells),indent=2)+'\n')
    (ROOT/'evidence/resident-composition-provenance.json').write_text(json.dumps(dict(source=provenance,qualifications=['bank-chain-hw-001','resident-bf16-hw-001','gdn-head-hw-003','attention-group-hw-001'],changes='Global frame70 bus; runtime coordinates; local colors12/13/14,queues3/6,DSR7,tasks14..16; global receive held during complete actor request. New composition remains unqualified.'),indent=2)+'\n')
    print(json.dumps(dict(generated=str(OUT),representative_pes=144,full_model_complete=False)))

if __name__=='__main__':build()
