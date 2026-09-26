"""Emit compact whole-wafer layout loops; no binary or model payload is local."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from resident_plan import load

def tile(name,role,phase='0',first='false',last='false'):
    return f'@set_tile_code(x,y,"{name}.csl",.{{.memcpy_params=memcpy.get_params(x),.role={role},.phase={phase},.first={first},.last={last},.west=x==0,.east=x==749,.north=y==0,.south=y==1159,.parity_x=@as(u16,x%2),.parity_y=@as(u16,y%2)}});'

def main():
    plan=load(ROOT/'configs');lines=['// Complete resident layout; each source is transferred as one bounded SDK message.','const memcpy=@import_module("<memcpy/get_params>",.{.width=750,.height=1160});',
        '''fn strip_x(width:u16,index:u16) u16 {
 if(width==136){return (index%4)*136;}
 if(width==48){if(index==1215){return 0;}return (index%3)*48;}
 if(index<1520){return 544+(index%5)*40;}
 if(index<7595){return 144+((index-1520)%15)*40;}
 if(index<15623){return ((index-7595)%18)*40;}
 return 48+(index-15623)*40;
}
fn strip_y(width:u16,index:u16) u16 {
 if(width==136){return index/4;}
 if(width==48){if(index==1215){return 1155;}return 304+index/3;}
 if(index<1520){return index/5;}
 if(index<7595){return 304+(index-1520)/15;}
 if(index<15623){return 709+(index-7595)/18;}return 1155;
}''','layout {','@set_rectangle(750,1160);']
    for name,a in plan['atlas']['assignments'].items():
        if 'strips' not in a:continue
        d=plan['matrices']['table'][plan['matrices']['matrix_ids'][name]];kind,lo,hi,width,base,rows,count,_=d
        role=1 if kind==1 else 2
        lines+=['// '+name,f'for(@range(u16,{count}))|tr|{{',f'const origin_x:u16=strip_x({width},{base}+tr);const origin_y:u16=strip_y({width},{base}+tr);',
                f'for(@range(i16,{width}))|column|{{const x:i16=@as(i16,origin_x)+column;const y:i16=@as(i16,origin_y);',
                tile('storage',role,'(tr*272)%128' if role==1 else '0','column==0',f'column=={width-1}'),
                f'if(column>0){{@set_color_config(x,y,@get_color(4+@as(u16,(x+1)%2)),.{{.routes=.{{.rx=.{{WEST}},.tx=.{{RAMP}}}}}});}}',
                f'if(column<{width-1}){{@set_color_config(x,y,@get_color(4+@as(u16,x%2)),.{{.routes=.{{.rx=.{{RAMP}},.tx=.{{EAST}}}}}});}}','}}']
    # Original matrix rectangles are already covered. Other contiguous role
    # runs need only a few thousand loop descriptions, never870000 source copies.
    for y in range(1160):
        x=0
        while x<750:
            role=int(plan['role'][y,x]);stop=x+1
            while stop<750 and int(plan['role'][y,stop])==role:stop+=1
            if role not in (1,2):
                name='gdn' if 6<=role<=8 else 'attention' if 9<=role<=13 else 'controller' if role==14 else 'storage'
                lines.append(f'for(@range(i16,{stop-x}))|delta|{{const x:i16={x}+delta;const y:i16={y};'+tile(name,role)+'}')
            x=stop
    lines+=['for(@range(i16,1160))|y|{for(@range(i16,750))|x|{',
        'if(y==1159){',
        'if(x==749){@set_color_config(x,y,@get_color(3),.{.routes=.{.rx=.{RAMP},.tx=.{WEST,NORTH}}});}',
        'else if(x==0){@set_color_config(x,y,@get_color(3),.{.routes=.{.rx=.{EAST},.tx=.{RAMP,NORTH}}});}',
        'else{@set_color_config(x,y,@get_color(3),.{.routes=.{.rx=.{EAST},.tx=.{RAMP,WEST,NORTH}}});}',
        '}else if(y==0){@set_color_config(x,y,@get_color(3),.{.routes=.{.rx=.{SOUTH},.tx=.{RAMP}}});}',
        'else{@set_color_config(x,y,@get_color(3),.{.routes=.{.rx=.{SOUTH},.tx=.{RAMP,NORTH}}});}',
        'if(x>0){@set_color_config(x,y,@get_color(8+@as(u16,(x+1)%2)),.{.routes=.{.rx=.{WEST},.tx=.{RAMP}}});}',
        'if(x<749){@set_color_config(x,y,@get_color(8+@as(u16,x%2)),.{.routes=.{.rx=.{RAMP},.tx=.{EAST}}});}',
        'if(x==749){',
        'if(y>0){@set_color_config(x,y,@get_color(10+@as(u16,(y+1)%2)),.{.routes=.{.rx=.{NORTH},.tx=.{RAMP}}});}',
        'if(y<1159){@set_color_config(x,y,@get_color(10+@as(u16,y%2)),.{.routes=.{.rx=.{RAMP},.tx=.{SOUTH}}});}','}','}}']
    for kind,groups,start,count in [('gdn',plan['roles']['gdn'],6,3),('attention',plan['roles']['attention'],9,5)]:
        for group in groups:
            gx,gy=group['controller']
            for n in range(count):
                x,y=gx+n,gy
                def route(color,rx,tx):lines.append(f'@set_color_config({x},{y},@get_color({color}),.{{.routes=.{{.rx=.{{{rx}}},.tx=.{{{tx}}}}}}});')
                route(12,'RAMP' if n==0 else 'WEST','EAST' if n==0 else 'RAMP' if n==count-1 else 'RAMP,EAST')
                if n>0:route(15-n if count==3 else 13+n%2,'RAMP','WEST')
                if n<count-1:route(14-n if count==3 else 13+(n+1)%2,'EAST','RAMP')
    representative=(ROOT/'resident/generated/layout.csl').read_text()
    lines+=representative[representative.index('  @export_name'):].splitlines()
    raw=('\n'.join(lines)+'\n').encode();path=ROOT/'resident/full-layout.csl';path.write_bytes(raw)
    record=dict(full_model_complete=False,scope='Candidate full750x1160 resident layout source, uncompiled',application_pes=870000,
        source_bytes=len(raw),source_sha256=hashlib.sha256(raw).hexdigest(),role_sha256=plan['role_sha256'],config_sha256=plan['config_sha256'])
    (ROOT/'evidence/full-layout-source.json').write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record))

if __name__=='__main__':main()
