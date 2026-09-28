"""Optional direct root/merge/frontend cohosts; old qualified programs stay intact.

This emits real local packet leases and queues. Recurrent workers and automatic
output-projection ingress are still absent, so it never claims a neural stage.
"""
import hashlib
import json
from pathlib import Path
from spatial.mixer_frontend_network import lower_frontend_network

ROOT = Path(__file__).resolve().parents[1]


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Qualified composition anchor changed: '+before[:80].decode())
    return source.replace(before, after)


def adapt(source, *, producer=False, merger=False, consumer=False):
    if not any((producer, merger, consumer)) or (merger and consumer):
        raise ValueError('Unsupported frontend cohost')
    if producer:
        source = b'param frontend_source_color:u16;\n'+source
        source = replace_once(source, b'@import_module("mixer_projection.csl",.{.rank_parity=',
                              b'@import_module("mixer_projection_frontend.csl",.{.frontend_color=frontend_source_color,.rank_parity=')
    if merger:
        source += b'''\nparam frontend_horizontal0:bool;param frontend_horizontal1:bool;param frontend_horizontal2:bool;
param frontend_vertical:bool;param frontend_incoming:u16;param frontend_outgoing:u16;
const packet_merge=@import_module("mixer_packet_merge.csl",.{
 .horizontal0=frontend_horizontal0,.horizontal1=frontend_horizontal1,.horizontal2=frontend_horizontal2,
 .vertical=frontend_vertical,.incoming_vertical=frontend_incoming,.outgoing_vertical=frontend_outgoing});
'''
        source = replace_once(source, b'fn arm(epoch:u32) void {',
                              b'fn arm(epoch:u32) void {@assert(packet_merge.drained());')
    if consumer:
        source += b'\n'+(ROOT/'runtime/mixer_frontend_port.cslpart').read_bytes()
        source = replace_once(source, b'fn arm(epoch:u32) void {',
                              b'fn arm(epoch:u32) void {@assert(frontend.drained() and frontend_cursor==0);')
        source = replace_once(source, b'const c=sys.command;const segment=c[2];',
                              b'const c=sys.command;const segment=c[2];frontend_snapshot();')
        segments = b'''else if(segment==9){half=frontend.wp;extent=2560;bits=16;writable=true;}
 else if(segment==10){half=frontend.hp;extent=1920;bits=16;}
 else if(segment==11){half=frontend.gp;extent=128;bits=16;writable=true;}
 else if(segment==12){half=frontend.pp;extent=6;bits=16;writable=true;}
 else if(segment==13){pointer=@ptrcast([*]u32,frontend.vp);extent=256;}
 else if(segment==14){half=frontend.zp;extent=384;bits=16;}
 else if(segment==15){pointer=@ptrcast([*]u32,&frontend.gates);extent=6;}
 else if(segment==16){pointer=frontend.packet_pointer;extent=387;}
 else if(segment==17){half=frontend.op;extent=384;bits=16;}
 else if(segment==18){pointer=frontend.ap;extent=8;}
 else if(segment==19){pointer=fsp;extent=8;}
 else if(segment==20){half=frontend.vvp;extent=384;bits=16;}
 '''
        source = replace_once(source, b'else if(segment==8){pointer=ap;extent=4;}',
                              b'else if(segment==8){pointer=ap;extent=4;}\n '+segments)
        source = replace_once(source, b'@assert(writable and mixer.drained() and mixer_status[2]==0);',
                              b'@assert(writable and mixer.drained() and mixer_status[2]==0 and frontend.drained() and frontend_cursor==0);')
        commands = b'''else if(c[1]==7){@assert(frontend_cursor==0 and mixer.drained());frontend.begin(c[2]);sys.unblock_cmd_stream();}
 else if(c[1]==8){@assert(frontend_cursor==0 and mixer.drained());frontend.reset();sys.unblock_cmd_stream();}
 else if(c[1]==9){@assert(c[2]<3);frontend.prepare(@as(u16,c[2]));sys.unblock_cmd_stream();}
 else if(c[1]==10){@assert(c[3]<3);frontend.packet_consumed(c[2],@as(u16,c[3]));sys.unblock_cmd_stream();}
 else if(c[1]==11){@assert(c[3]<3);frontend.output_consumed(c[2],@as(u16,c[3]));sys.unblock_cmd_stream();}
 '''
        source = replace_once(source, b'else if(c[1]==6){@assert(c[3]<65536);mixer_consume(c[2],@as(u16,c[3]));}',
                              b'else if(c[1]==6){@assert(c[3]<65536);mixer_consume(c[2],@as(u16,c[3]));}\n '+commands)
    return source


def fuse_projection(source):
    """Reuse the native projection's leased output, DSRs and completion task."""
    source = b'param frontend_color:u16;const frontend_oq=@get_output_queue(6);\n'+source
    old = b'''  outgoing[2]=@as(u32,rows);outgoing[3]=@as(u32,math.round(partial[0]));
  if(!use_bf16){outgoing[3]|=@as(u32,math.round(partial[1]))<<16;}outgoing[4]=@as(u32,matrix);
  @activate(root_ready_task);'''
    source = replace_once(source, old, b'  if(matrix<4){frontend_encode();emit_packet(true);}else{\n'+old+b'\n  }')
    source = replace_once(source, b'''  const m=@get_dsd(mem1d_dsd,.{.base_address=&outgoing,.extent=5});const f=@get_dsd(fabout_dsd,.{.output_queue=oq,.extent=5});
  @load_to_dsr(os,m);@load_to_dsr(od,f,.{.async=true,.ut_id=@get_ut_id(5),.activate=ot});@mov32(od,os,.{.async=true});''',
                          b'  emit_packet(false);')
    source = replace_once(source, b'@assert(output_busy and key!=0);',
                          b'@assert(output_busy and (key!=0 or matrix<4));')
    source += b'\n'+(ROOT/'runtime/mixer_root_fused.cslpart').read_bytes()
    source += b'\ncomptime {@initialize_queue(frontend_oq,.{.color=@get_color(frontend_color)});@comptime_assert(frontend_color==0 or frontend_color==1 or frontend_color==19); }\n'
    return source


def compose_frontend(files, profiles, region, setups):
    plan = lower_frontend_network(region, profiles, setups)
    files['mixer_projection_frontend.csl'] = fuse_projection(files['mixer_projection.csl'])
    producers = {tuple(x['pe']): x for x in plan['producers']}
    mergers = {tuple(x['pe']): x for x in plan['mergers']}
    consumers = {tuple(x['pe']): x for x in plan['consumers']}
    bindings = {}
    for p in profiles:
        pe = tuple(p['pe']); flags = (pe in producers, pe in mergers, pe in consumers)
        if not any(flags): continue
        original = p['source']; name = 'frontend_'+''.join(str(int(x)) for x in flags)+'_'+original
        if name not in files:
            files[name] = adapt(files[original], producer=flags[0], merger=flags[1], consumer=flags[2])
            bindings[name] = dict(original=original, original_sha256=hashlib.sha256(files[original]).hexdigest(),
                                  composed_sha256=hashlib.sha256(files[name]).hexdigest(), producer=flags[0], merger=flags[1], consumer=flags[2])
        p['source'] = name; params = p['parameters']
        if flags[0]: params['frontend_source_color'] = producers[pe]['color']
        if flags[1]:
            m = mergers[pe]
            params.update(frontend_horizontal0=0 in m['horizontal'], frontend_horizontal1=1 in m['horizontal'],
                          frontend_horizontal2=19 in m['horizontal'], frontend_vertical=m['vertical'],
                          frontend_incoming=m['incoming'], frontend_outgoing=m['outgoing'])
        if flags[2]: params['frontend_group'] = consumers[pe]['group']
    for name in ('mixer_frontend', 'mixer_packet_merge'):
        files[name+'.csl'] = (ROOT/'csl'/(name+'.csl')).read_bytes()
    files['frontend-composition.json'] = (json.dumps(dict(bindings=bindings, compiled=False, physical=False,
        recurrent_core_connected=False, complete_neural_stage=False), indent=2)+'\n').encode()
    files['frontend-network.json'] = (json.dumps(plan, separators=(',', ':'))+'\n').encode()
    return plan
