"""Compact CSL layout for the full original MLP plus full-model bank addresses.

The source is a candidate executable composition, not full-wafer compiler or
runtime evidence. Worker's Y register assumes the documented fabric offset4,1.
"""
from .contraction import tree


def emit(residency):
    r=residency;assert r.flow.spec.columns==9
    low=[i for i in range(870000) if r.roles[i]==2 and r.fp8[i]==104]
    assert len(low)==190608;cutoff=max(low)
    for i in range(870000):
        if r.roles[i] in (1,2,3):
            expected=129 if r.roles[i]==1 else 86 if r.roles[i]==3 else 104 if i<=cutoff else 102
            assert r.fp8[i]==expected
    nodes=tree(40);left=[];right=[];outs=[]
    color=lambda n:4+2*(n['depth']-1)+n['side']
    for n in nodes:
        ch=n['children'];left.append(color(nodes[ch[0]]) if ch else 4)
        right.append(color(nodes[ch[1]]) if len(ch)>1 else 5);outs.append(color(n) if n['rank'] else 16)
    arrays=dict(parent=[n['parent'] or 0 for n in nodes],children=[len(n['children']) for n in nodes],subtree=[n['size'] for n in nodes],
                left=left,right=right,native_out=outs)
    lines=['// Full750x1160 bank footprint; fixed compiler fabric offsets4,1 required.',
           'const memcpy=@import_module("<memcpy/get_params>",.{.width=750,.height=1160});']
    for name,values in arrays.items():lines.append('const '+name+'=[40]u16{'+','.join(map(str,values))+'};')
    # All combinations used below are explicit DSL route literals. No assumed
    # bit encoding is written to hardware; the SDK compiles these directions.
    ports=['RAMP','NORTH','EAST','SOUTH','WEST']
    combinations=[(0,2),(0,4),(0,8),(0,16),
                  (1,1),(1,8),(1,9),(2,1),(2,16),(2,17),(2,8),(2,24),
                  (3,1),(3,2),(3,3),(3,16),(4,1),(4,4),(4,5)]
    lines.append('fn route(x:u16,y:u16,comptime c:u16,comptime rx:u16,comptime tx:u16) void {')
    for ri,rx in enumerate(range(5)):
        lines.append((' if' if ri==0 else ' else if')+f'(rx=={rx})'+'{')
        for ti,(_,tx) in enumerate(v for v in combinations if v[0]==rx):
            outputs=','.join(ports[i] for i in range(5) if tx&(1<<i))
            lines.append(('  if' if ti==0 else '  else if')+f'(tx=={tx})'+'{@set_color_config(x,y,@get_color(c),.{.routes=.{.rx=.{'+ports[rx]+'},.tx=.{'+outputs+'}}});}')
        lines+=['  else{@comptime_assert(false);}',' }']
    lines+=[' else{@comptime_assert(false);}','}',
    'fn line(x0:u16,y0:u16,x1:u16,y1:u16,comptime c:u16,comptime multicast:bool) void {',
    ' const vertical=x0==x1;const forward=if(vertical) y1>y0 else x1>x0;',
    ' const length=if(vertical) (if(forward) y1-y0 else y0-y1) else (if(forward) x1-x0 else x0-x1);',
    ' const tx=if(vertical) (if(forward) 8 else 2) else (if(forward) 4 else 16);',
    ' const rx=if(vertical) (if(forward) 1 else 3) else (if(forward) 4 else 2);',
    ' for(@range(u16,length+1))|i|{',
    '  const x=if(vertical) x0 else (if(forward) x0+i else x0-i);',
    '  const y=if(!vertical) y0 else (if(forward) y0+i else y0-i);',
    '  route(x,y,c,if(i==0) 0 else rx,(if(i<length) tx else 0)|(if(i==length or (multicast and i>0)) 1 else 0));',
    ' }','}', 'layout {',' @set_rectangle(750,1160);']
    lines+=[' for(@range(u16,750))|x|{const column_memcpy=memcpy.get_params(x);for(@range(u16,1160))|y|{',
    '  const bx=x/83;const lx=x%83;',
    '  if(y<40 and x==0){@set_tile_code(x,y,"mlp_norm.csl",.{.memcpy_params=column_memcpy,.rank=y});}',
    '  else if(y<40 and bx<9 and (lx==40-y or lx==42+y)){',
    '   const up=lx==42+y;@set_tile_code(x,y,"mlp_join.csl",.{.memcpy_params=column_memcpy,.header=true,.relay_only=up,.has_child=!up and bx<8,.own_color=13,.child_color=17+(bx+1)%2,.out_color=17+bx%2});',
    '  }else if(y>=40 and bx<9 and ((y-40)/65)*9+bx<136){',
    '   const by=(y-40)/65;const ly=(y-40)%65;const g=by*9+bx;',
    '   if(ly==0 and lx>=1 and lx<=40){',
    '    @set_tile_code(x,y,"mlp_join.csl",.{.memcpy_params=column_memcpy,.header=false,.relay_only=false,.has_child=g+9<136,.own_color=2,.child_color=13+(by+1)%2,.out_color=13+by%2});',
    '   }else if(ly>0 and lx==41){@set_tile_code(x,y,"mlp_activation.csl",.{.memcpy_params=column_memcpy,.pair=ly-1});}',
    '   else if(ly>0 and ((lx>=1 and lx<=40) or (lx>=42 and lx<=81))){',
    '    const gate=lx<=40;const k=if(gate) 40-lx else lx-42;',
    '    const state=!gate and g*160+((ly-1)/4)*10+k/4<2304;',
    f'    const fp8_slots=if(gate) 129 else if(state) 86 else if(@as(u32,y)*750+@as(u32,x)<={cutoff}) 104 else 102;',
    '    @set_tile_code(x,y,"mlp_worker.csl",.{.memcpy_params=column_memcpy,.gate=gate,.rank=k,.fp8_slots=fp8_slots,.children=children[k],.subtree=subtree[k],.left_color=left[k],.right_color=right[k],.native_out=if(k==0 and !gate) 17 else native_out[k]});',
    '   }else{@set_tile_code(x,y,"mlp_idle.csl",.{.memcpy_params=column_memcpy});}',
    '  }else{@set_tile_code(x,y,"mlp_idle.csl",.{.memcpy_params=column_memcpy});}',
    ' }}',
    ' // Shared40 input packets: horizontal header multicast, then real relays.',
    ' for(@range(u16,40))|k|{',
    '  const last=8*83+42+k;',
    '  for(@range(u16,last+1))|x|{route(x,k,2,if(x==0) 0 else 4,(if(x<last) 4 else 0)|(if(x%83==40-k or x%83==42+k) 1 else 0));}',
    '  for(@range(u16,9))|bx|{for(@range(u16,2))|b|{',
    '   const x=83*bx+(if(b==0) 40-k else 42+k);const last_y=40+((135-bx)/9)*65+64;',
    '   for(@range(u16,last_y-k+1))|offset|{const y=k+offset;const consume=y>40 and (y-40)%65>0;',
    '    route(x,y,3,if(offset==0) 0 else 1,(if(y<last_y) 8 else 0)|(if(consume) 1 else 0));',
    '   }',
    '  }}',
    ' }',
    ' for(@range(u16,136))|g|{const bx=g%9;const by=g/9;const x=83*bx;const y=40+65*by;',
    '  for(@range(u16,64))|p|{',
    '   for(@range(u16,39))|idx|{const k=idx+1;',
    '    line(x+40-k,y+1+p,x+40-parent[k],y+1+p,native_out[k],false);',
    '    line(x+42+k,y+1+p,x+42+parent[k],y+1+p,native_out[k],false);',
    '   }',
    '   line(x+40,y+1+p,x+41,y+1+p,16,false);line(x+42,y+1+p,x+41,y+1+p,17,false);',
    '   if(p>0){line(x+41,y+1+p,x+41,y+p,4+p%2,false);}',
    '   for(@range(u16,40))|j|{line(x+40-j,y+1+p,x+40-j,y+p,if(p%2==0) 2 else 15,false);}',
    '  }',
    '  line(x+41,y+1,x+41,y+64,6,true);',
    '  route(x+41,y+1,18,0,2);route(x+41,y,18,3,16);',
    '  for(@range(u16,40))|j|{const xx=x+40-j;',
    '   route(xx,y,18,2,8|(if(j<39) 16 else 0));',
    '   for(@range(u16,64))|p|{route(xx,y+1+p,18,1,1|(if(p<63) 8 else 0));}',
    '   line(xx,y,xx,if(by>0) y-65 else j,13+by%2,false);',
    '  }',
    ' }',
    ' for(@range(u16,9))|bx|{for(@range(u16,40))|j|{line(bx*83+40-j,j,if(bx>0) (bx-1)*83+40-j else 0,j,17+bx%2,false);}}',
    ' for(@range(u16,39))|i|{const k=i+1;line(0,k,0,k-1,4+k%2,false);}',
    ' line(0,0,0,39,6,true);']
    for name,ty in [('bank','u32'),('packet','u32'),('audit','u32'),('control','u16'),('native_local','f32'),('down_local','f32'),('value','u32'),('scale','u32'),('gains','u16'),('residual','u16'),('successor','u16')]:
        lines.append(f' @export_name("{name}",[*]{ty},false);')
    lines+=[' @export_name("arm",fn(u16)void);',' @export_name("start",fn()void);',' @export_name("embedding_lookup",fn(u16,u16)void);','}']
    return '\n'.join(lines)+'\n'
