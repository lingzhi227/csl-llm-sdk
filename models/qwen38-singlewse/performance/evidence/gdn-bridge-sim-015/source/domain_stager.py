"""Five-PE two-domain switch/reentry fixture; no neural or physical qualification."""
import hashlib,json,re
from pathlib import Path
from tools.build_frontend_stage import verified_frozen,encoded

ROOT=Path(__file__).resolve().parents[1]


def build():
    base=ROOT/'evidence/layer-backend-compile-045'
    m=json.loads((base/'source-manifest.json').read_text())['files']
    files={n:verified_frozen(base/'source'/n,h)for n,h in m.items()
           if n.endswith('.csl')or n in ['check_sram.py','placement.py','elf_inventory.py','source_gate.py']}
    samples=json.loads(verified_frozen(base/'source/profiles.json',m['profiles.json']))['samples']
    selected=[next(s for s in samples if s['original_pe']==pe and s['variant']=='candidate')for pe in [[95,58],[109,145]]]
    name='stream_bridge_compact_layer_projection.csl';original=files[name]
    files[name]+=b'''\nfn probe_arm(token:u32) void {bridge_stream_arm(token);}
fn probe_send() void {sys.unblock_cmd_stream();}
fn probe_wait() void {sys.unblock_cmd_stream();}
comptime {@export_symbol(probe_arm);@export_symbol(probe_send);@export_symbol(probe_wait);}
'''
    source=(ROOT/'probes/gdn_bridge/source.csl').read_bytes()
    source=source.replace(b'fn probe_arm() void',b'fn probe_arm(token:u32) void')
    source=source.replace(b'@assert(count==total_workers and audit[2]==0);audit[2]=count;',
                          b'@assert(count>0 and audit[2]+@as(u32,count)<=@as(u32,total_workers));audit[2]+=count;')
    files['source.csl']=source
    files['sink.csl']=(ROOT/'probes/gdn_bridge/domain_sink.csl').read_bytes()
    files['run.py']=(ROOT/'probes/gdn_bridge/domain_run.py').read_bytes()
    files['backend.py']=(ROOT.parent/'runtime/backend.py').read_bytes()
    lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=5,.height=1});','layout {',' @set_rectangle(5,1);',
           ' @set_tile_code(0,0,"source.csl",.{.memcpy_params=memcpy.get_params(0),.total_frames=192,.total_workers=51,.send_color=1,.return_color=9});']
    children=[];offset=0
    for i,s in enumerate(selected):
        x=i+1;sx=i+3;p=dict(s['parameters']);p.update(bridge_group=14,bridge_input=1,bridge_return_output=9,bridge_switch=i==0)
        incoming,returned=(18,13)if i==0 else(12,16)
        p.update(bridge_output=incoming,bridge_return_input=returned)
        fields=','.join('.%s=%s'%(k,v if isinstance(v,str)else str(v).lower())for k,v in p.items())
        lines.append(f' @set_tile_code({x},0,"{name}",.{{.memcpy_params=memcpy.get_params({x}),{fields}}});')
        lines.append(f' @set_tile_code({sx},0,"sink.csl",.{{.memcpy_params=memcpy.get_params({sx}),.group=14,.first_frame={offset},.total_frames={p["bridge_frames"]},.total_workers={p["bridge_workers"]},.segments={p["bridge_segments"]},.input_color={incoming},.output_color={returned}}});')
        children.append(dict(bridge_x=x,sink_x=sx,original_pe=s['original_pe'],bank_words=p['bank_words'],
                             frames=p['bridge_frames'],workers=p['bridge_workers'],parameters=p))
        offset+=p['bridge_frames']
        for color,start,end,forward in [(incoming,x,sx,True),(returned,x,sx,False)]:
            for q in range(start,end+1):
                rx='RAMP'if(q==start if forward else q==end)else('WEST'if forward else'EAST')
                tx='RAMP'if(q==end if forward else q==start)else('EAST'if forward else'WEST')
                extra=',.switches=.{.pos1=.{.rx=RAMP},.pop_mode=.{.pop_on_advance=true}}'if not forward and q==end else''
                lines.append(f' @set_color_config({q},0,@get_color({color}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}{extra}}});')
    for x,c,rx,tx,extra in [(0,1,'RAMP','EAST',''),(1,1,'WEST','RAMP,EAST',''),(2,1,'WEST','RAMP',''),
        (0,9,'EAST','RAMP',''),(1,9,'RAMP','WEST',',.switches=.{.pos1=.{.rx=EAST},.pop_mode=.{.pop_on_advance=true}}'),(2,9,'RAMP','WEST','')]:
        lines.append(f' @set_color_config({x},0,@get_color({c}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}{extra}}});')
    lines+=[' '+d for d in re.findall(r'@export_name\("[^"]+"[^;]+;',files['layout.csl'].decode())]
    for n in ('wire','received','probe_audit'):lines.append(f' @export_name("{n}",[*]u32,false);')
    lines+=[' @export_name("probe_arm",fn(u32)void);',' @export_name("probe_send",fn()void);',' @export_name("probe_wait",fn()void);','}']
    files['layout.csl']=('\n'.join(lines)+'\n').encode()
    files['probe-plan.json']=encoded(dict(application=[5,1],group=14,frames=192,workers=51,children=children,
        original_cohost_body_sha256=hashlib.sha256(original).hexdigest(),resource_compile='layer-backend-compile-045',
        synthetic_wire=True,physical=False,neural_execution=False,
        scope='Two unchanged selected045 cohost bodies/bank extents with remapped bridge group/colors and explicit switch. '
              'Synthetic producer cohorts receive multicast concurrently with ordered reverse segments.'))
    files['domain_stager.py']=Path(__file__).read_bytes()
    return files
