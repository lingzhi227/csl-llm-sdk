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


def document(compare_scaling=False):
    shapes=[dict(NativeShape(r, 256 // r).document(),vector_scales=vector)
            for vector in ([False,True] if compare_scaling else [False]) for r in (2,4,8,16)]
    return dict(schema='wse-equal-work-native-shapes-v1', application=[len(shapes), 1],
                shapes=shapes,compare_scaling=compare_scaling,
                repetitions=32, fp8_slots=110, bf16_slots=13, bank_words=8814,
                selected_fp8_slot=0, selected_bf16_slot=6, full_model=False,
                ownership='Component-only repacking of selected original tiles; remaining original background slots retain their 2x128 format. No P15/P17 global ownership identity is reused for retiled values.',
                scope='Four original-weight 256-MAC shapes with equal resident bank capacity; optional paired scalar/vector output scaling with identical per-shape operands and banks. Local arithmetic and FP8 decode-plus-arithmetic timings only; excludes input distribution, reduction, matrix coverage and dependent model inference.')


def emit_layout(compare_scaling=False):
    plan=document(compare_scaling);width=plan['application'][0]
    lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=%d,.height=1});'%width,
           'layout {', ' @set_rectangle(%d,1);'%width]
    for x, shape in enumerate(plan['shapes']):
        lines.append(' @set_tile_code(%d,0,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),.rows=%d,.columns=%d,.vector_scales=%s});' % (x,x,shape['rows'],shape['columns'],str(shape['vector_scales']).lower()))
    for name, dtype in [('bank','u32'),('packet','u32'),('output','f32'),('ticks','u16'),('audit','u32')]:
        lines.append(' @export_name("%s",[*]%s,false);' % (name,dtype))
    lines.extend([' @export_name("measure",fn(u16,u16)void);', '}'])
    return '\n'.join(lines)+'\n'
