"""Two-stage counter windows feeding representative maximum columnar banks."""
class FilteredBankPlan:
    def document(self):
        return dict(schema='wse-filtered-bank-component-v1', application=[4,5], full_model=False,
                    colors=dict(horizontal=2,vertical=20), input_queue=2,output_queue=2,
                    input_microthread=2,output_microthread=3,workers=self.workers(),
                    cases=10,packet_words=65,pair_words=130,horizontal_periods=[260,780],
                    scope='Three horizontal pair selectors and twelve vertical packet selectors, then original FP8/BF16 native dots in representative maximum mixed banks. Independent host-provided operands; no complete matrix, autonomous neural feedback or model speed claim.')
    def workers(self):
        return [dict(x=x,y=y,rank=(y-1)*3+x-1,fp8_slots=110+((y-1)&1),bf16_slots=13-((y-1)&1)) for y in range(1,5) for x in range(1,4)]
    def emit_layout(self):
        lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=5});',
               'const filter=.{.kind=.{.counter=true},.count_data=true,.count_control=false,.init_counter=0,.limit1=779,.max_counter=129};','layout {',' @set_rectangle(4,5);']
        workers={(d['x'],d['y']):d for d in self.workers()}
        for y in range(5):
            for x in range(4):
                role=0 if (x,y)==(0,0) else 1 if y==0 else 2 if x else 3
                d=workers.get((x,y),dict(fp8_slots=0,bf16_slots=0))
                lines.append(' @set_tile_code(%d,%d,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),.role=%d,.fp8_slots=%d,.bf16_slots=%d});'%(x,y,x,role,d['fp8_slots'],d['bf16_slots']))
                if y==0:
                    tx='EAST' if x==0 else 'RAMP' if x==3 else 'RAMP,EAST'
                    lines.append(' @set_color_config(%d,%d,@get_color(2),.{.routes=.{.rx=.{%s},.tx=.{%s}},.teardown=true%s});'%(x,y,'RAMP' if x==0 else 'WEST',tx,',.filter=filter' if x else ''))
                if x:
                    tx='SOUTH' if y==0 else 'RAMP' if y==4 else 'RAMP,SOUTH'
                    lines.append(' @set_color_config(%d,%d,@get_color(20),.{.routes=.{.rx=.{%s},.tx=.{%s}},.teardown=true%s});'%(x,y,'RAMP' if y==0 else 'NORTH',tx,',.filter=filter' if y else ''))
        for n,t in [('bank','u32'),('control','u16'),('packet','u32'),('local_result','f32'),('audit','u32'),('ticks','u16')]:
            lines.append(' @export_name("%s",[*]%s,false);'%(n,t))
        lines += [' @export_name("arm",fn()void);',' @export_name("start",fn()void);','}']
        return '\n'.join(lines)+'\n'
