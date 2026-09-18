"""Emit a full resident MLP with bounded rectangular weight upload descriptors.

Source generation is not compile, memory-fit, numerical or hardware acceptance.
"""
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import shutil


@dataclass(frozen=True)
class Shape:
    hidden: int = 5120
    intermediate: int = 17408
    rows: int = 128
    columns: int = 96
    diagnostic_padding: bool = False

    def __post_init__(self):
        if self.rows != 128 or self.columns != 96 or min(self.hidden, self.intermediate) < 128:
            raise ValueError("Qualified kernel shape is 128x96")
        if self.hidden % 128 or self.intermediate % 128:
            raise ValueError("Whole 128-row output groups required")

    @property
    def projection_shards(self): return (self.hidden+95)//96
    @property
    def down_shards(self): return (self.intermediate+95)//96
    @property
    def channels(self): return self.intermediate//128
    @property
    def outputs(self): return self.hidden//128
    @property
    def width(self): return max(self.projection_shards, self.down_shards)+2
    @property
    def down_y(self): return 2*self.channels+2
    @property
    def height(self): return self.down_y+max(self.outputs, 2)


def upload_batches(shape=Shape(), max_host_bytes=16 << 20):
    """Uniform symbol weights, extent 12288 u16, native16 uint32 host slots.

    Every rectangle is entirely matrix PEs. Projection rows alternate gate/up;
    each PE uses column-major [96,128] raw BF16 words. Tail columns are zero.
    """
    bytes_per_pe = 128*96*4
    result = []
    for role, width, height, y0 in (("gate_up", shape.projection_shards, 2*shape.channels, 0),
                                  ("down", shape.down_shards, shape.outputs, shape.down_y)):
        rows_per_call = max_host_bytes//(width*bytes_per_pe)
        if rows_per_call < 1:
            raise ValueError("Host buffer cannot hold one matrix row")
        for start in range(0, height, rows_per_call):
            count = min(rows_per_call, height-start)
            result.append(dict(role=role, x=0, y=y0+start, width=width, height=count,
                               source_row=start, count_per_PE=12288, host_bytes=width*count*bytes_per_pe,
                               native_bytes=width*count*bytes_per_pe//2))
    return result


def layout(shape=Shape()):
    s = shape
    lines = [f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={s.width},.height={s.height}}});',
             f'layout{{@set_rectangle({s.width},{s.height});']
    # Compact compile-time loops permit specialization coalescing instead of a
    # generated Python object or file for every physical PE.
    lines += [f'for(@range(u16,{s.height}))|y|{{for(@range(u16,{s.width}))|x|{{',
              f'const proj=y<{2*s.channels} and x<{s.projection_shards};',
              f'const down=y>={s.down_y} and y<{s.down_y+s.outputs} and x<{s.down_shards};',
              f'const non=x=={s.width-2} and y<{2*s.channels} and y%2==0;',
              f'const role:u16=if(proj) (if(y%2==0) @as(u16,1) else @as(u16,2)) else if(down) @as(u16,3) else if(non) @as(u16,4) else if(x=={s.width-1} and y==0) @as(u16,5) else if(x=={s.width-1} and y=={s.down_y}) @as(u16,6) else if(x=={s.width-1} and y=={s.down_y+1}) @as(u16,7) else @as(u16,0);',
              f'const group:u16=if(proj or non) y/2 else if(down) y-{s.down_y} else @as(u16,0);',
              'const matrix=proj or down;',
              f'@set_tile_code(x,y,"pe.csl",.{{.memcpy_params=memcpy.get_params(x),.role=role,.group=if((matrix and x==0) or non) group else @as(u16,0),.ordinal=if(matrix) x else @as(u16,0),.participants=if(down) @as(u16,{s.down_shards}) else @as(u16,{s.projection_shards}),.hidden={s.hidden},.intermediate={s.intermediate},.width={s.width},.down_y={s.down_y},.diagnostic_padding={"true" if s.diagnostic_padding else "false"} and x=={s.width-2} and y<{2*s.channels} and y%2==1}});']
    for color, row0, height, width, period in ((3,0,2*s.channels,s.projection_shards,s.projection_shards*96),
                                              (4,s.down_y,s.outputs,s.down_shards,s.down_shards*96)):
        lines += [f'if(y=={row0}){{',
                  f'if(x=={s.width-1}){{@set_color_config(x,y,@get_color({color}),.{{.routes=.{{.rx=.{{RAMP}},.tx=.{{WEST}}}}}});}}',
                  f'else if(x>={width}){{@set_color_config(x,y,@get_color({color}),.{{.routes=.{{.rx=.{{EAST}},.tx=.{{WEST}}}}}});}}',
                  'else{',
                  f'@set_color_config(x,y,@get_color({color}),.{{.routes=.{{.rx=.{{EAST}},.tx=if(x==0) .{{RAMP{",SOUTH" if height>1 else ""}}} else .{{RAMP,WEST{",SOUTH" if height>1 else ""}}}}},.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter=({period}-x*96)%{period},.max_counter=95,.limit1={period-1}}}}});',
                  '}',
                  f'}}else if(y>{row0} and y<{row0+height} and x<{width}){{',
                  f'@set_color_config(x,y,@get_color({color}),.{{.routes=.{{.rx=.{{NORTH}},.tx=if(y=={row0+height-1}) .{{RAMP}} else .{{RAMP,SOUTH}}}},.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter=({period}-x*96)%{period},.max_counter=95,.limit1={period-1}}}}});',
                  '}']
    lines.append('}}')
    for name, kind in (('weights','u16'),('input','f32'),('result','f32'),('gate','u16'),('up','u16'),('product','u16'),('output','u16'),('state','u32'),
                       ('partial','f32'),('rounded','u16'),('silu','u16'),('exponential','f32'),('sigmoid','f32'),('activation_fp32','f32'),('product_fp32','f32')):
        lines.append(f'@export_name("{name}",[*]{kind},false);')
    for name in ('initialize','prepare','compute'):
        lines.append(f'@export_name("{name}",fn()void);')
    return '\n'.join(lines+['}'])+'\n'


def emit(destination, accepted_source, shape=Shape()):
    destination=Path(destination);destination.mkdir(parents=True,exist_ok=False)
    source=Path(accepted_source)
    files={}
    for name in ('gemv_kernel.csl','line.csl','input.csl','packets.csl','nonlinear_kernel.csl','exp_kernel.csl'):
        raw=(source/name).read_bytes();(destination/name).write_bytes(raw)
        files[name]={'sha256':hashlib.sha256(raw).hexdigest(),'origin':'accepted resident SDK002 '+name}
    shutil.copyfile(Path(__file__).with_name('pe.csl'),destination/'pe.csl')
    (destination/'layout.csl').write_text(layout(shape))
    calls=upload_batches(shape)
    info=dict(shape=asdict(shape),application=[shape.width,shape.height],matrix_PEs=2*shape.channels*shape.projection_shards+shape.outputs*shape.down_shards,
              projection_tail=shape.hidden-(shape.projection_shards-1)*96,down_tail=shape.intermediate-(shape.down_shards-1)*96,
              weight_uploads=calls,weight_native_bytes=sum(x['native_bytes'] for x in calls),weight_host_slot_bytes=sum(x['host_bytes'] for x in calls),
              kernel_origins=files,FP32_reduction_order='right-to-left linear sum of forward local 96-column FMA contractions; one final BF16 RNE at root',
              nonlinear_domain='accepted abs(gate),abs(up)<=24; must be checked by original-weight reference before hardware admission',
              status='source_only_uncompiled',full_model=False)
    (destination/'layout.json').write_text(json.dumps(info,indent=2)+'\n')
    return info
