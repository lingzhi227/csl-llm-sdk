"""Equal-work native tile candidates; no complete-model ownership reassignment."""
from dataclasses import dataclass


@dataclass(frozen=True)
class NativeShape:
    rows: int
    columns: int

    def validate(self):
        if (self.rows, self.columns) not in ((2, 128), (4, 64), (8, 32), (16, 16)):
            raise ValueError('Supported 256-weight native tile required')
        if self.rows * self.columns != 256 or 128 % self.columns or 128 % self.rows:
            raise ValueError('Tile must stay inside one original 128x128 scale block')
        return self

    def document(self):
        self.validate()
        return dict(rows=self.rows, columns=self.columns, weight_elements=256,
                    fp8_weight_bytes=256, bf16_weight_bytes=512,
                    input_packet_words=self.columns // 2 + 1,
                    partial_output_words=self.rows + 1,
                    scalar_map_steps=self.columns,
                    fp32_bf16_expansion_bytes=self.rows * 4)


def document(compare_scaling=False,input_loop=None):
    if input_loop not in [None,"pair","unroll"]:raise ValueError("Unsupported native loop")
    if compare_scaling and input_loop:raise ValueError("Choose one matched comparison")
    shapes=[dict(NativeShape(r, 256 // r).document(),vector_scales=vector)
            for vector in ([False,True] if compare_scaling else [False]) for r in (2,4,8,16)]
    if input_loop:
        shapes=[dict(NativeShape(r,256//r).document(),vector_scales=True,input_loop=input_loop if paired else "single",
                     input_map_steps=((256//r)//2 if input_loop=="pair" else 0) if paired else 256//r)
                for paired in [False,True] for r in (2,4,8,16)]
    return dict(schema='wse-equal-work-native-shapes-v1', application=[len(shapes), 1],
                shapes=shapes,compare_scaling=compare_scaling,compare_input_loop=input_loop,
                repetitions=32, fp8_slots=110, bf16_slots=13, bank_words=8814,
                selected_fp8_slot=0, selected_bf16_slot=6, full_model=False,
                ownership='Component-only repacking of selected original tiles; remaining original background slots retain their 2x128 format. No P15/P17 global ownership identity is reused for retiled values.',
                scope='Four original-weight 256-MAC shapes with equal resident bank capacity; optional matched scalar/vector scaling or single versus paired/unrolled input loops with identical per-shape operands and banks. Local arithmetic and FP8 decode-plus-arithmetic timings only; excludes input distribution, reduction, matrix coverage and dependent model inference.')


def emit_layout(compare_scaling=False,input_loop=None):
    plan=document(compare_scaling,input_loop);width=plan['application'][0]
    lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=%d,.height=1});'%width,
           'layout {', ' @set_rectangle(%d,1);'%width]
    for x, shape in enumerate(plan['shapes']):
        lines.append(' @set_tile_code(%d,0,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),.rows=%d,.columns=%d,.vector_scales=%s,.input_loop=%s});' % (x,x,shape['rows'],shape['columns'],str(shape['vector_scales']).lower(),{'single':0,'pair':1,'unroll':2}[shape.get('input_loop','single')]))
    for name, dtype in [('bank','u32'),('packet','u32'),('output','f32'),('ticks','u16'),('audit','u32')]:
        lines.append(' @export_name("%s",[*]%s,false);' % (name,dtype))
    lines.extend([' @export_name("measure",fn(u16,u16)void);', '}'])
    return '\n'.join(lines)+'\n'
