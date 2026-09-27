"""Bounded route-word and two-color teardown/rebind qualification plan."""
from .forest import PORTS


class RouteEpochPlan:
    width = 4
    height = 5
    rounds = 64
    bank_bytes = 35256

    def xy(self, rank):
        row, col = divmod(rank, self.width)
        return (col if row % 2 == 0 else self.width - col - 1, row)

    def route(self, rank, reverse):
        source, destination = (19, 0) if reverse else (0, 19)
        step = -1 if reverse else 1
        def direction(other):
            a, b = self.xy(rank), self.xy(other)
            return {(1, 0): 'EAST', (-1, 0): 'WEST', (0, 1): 'SOUTH', (0, -1): 'NORTH'}[(b[0] - a[0], b[1] - a[1])]
        rx = 'RAMP' if rank == source else direction(rank - step)
        tx = 'RAMP' if rank == destination else direction(rank + step)
        return dict(xy=self.xy(rank), rx=rx, tx=tx, word=PORTS[rx] | (PORTS[tx] << 3))

    def document(self):
        return dict(schema='wse-route-epoch-component-v1', application=[4, 5], physical=False, full_model=False,
                    bank_bytes_per_pe=self.bank_bytes, bank_payload='Synthetic retention sentinel with maximum P12 mixed-bank byte capacity; not original weights or a neural kernel.',
                    maximum_rounds=self.rounds, simulation_rounds=16, packet_words=257, colors=[3, 4], disabled_test_color=5,
                    input_queue=2, output_queue=2, input_microthread=2, output_microthread=3,
                    local_tasks=[8, 9, 10, 11], teardown_task=29, dsrs=[2, 3],
                    profiles=[[self.route(r, reverse) for r in range(20)] for reverse in (False, True)],
                    readiness='Strict alternating path reversal. Next source was previous destination and waits for full packet plus teardown. Every next-path router is either already configured or held in teardown on the alternating color. This is not a general arbitrary-forest transition barrier.',
                    timing='Same-endpoint send in epoch e through receive in epoch e+1. Includes one intervening route transition and packet feedback; excludes host arm/readback. No cross-PE clock subtraction or model throughput claim.')

    def emit_layout(self):
        lines=['const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=5});','layout {',' @set_rectangle(4,5);']
        for r in range(20):
            x,y=self.xy(r)
            lines.append(' @set_tile_code(%d,%d,"pe.csl",.{.memcpy_params=memcpy.get_params(%d),.rank=%d,.forward_word=%d,.reverse_word=%d});' %
                         (x,y,x,r,self.route(r,False)['word'],self.route(r,True)['word']))
            for c in (3,4,5):
                lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{RAMP},.tx=.{RAMP}},.teardown=true});' % (x,y,c))
        for name,kind in [('bank','u32'),('packet','u32'),('audit','u32'),('ticks','u16')]:
            lines.append(' @export_name("%s",[*]%s,false);' % (name,kind))
        lines += [' @export_name("initialize",fn(u16)void);',' @export_name("start",fn()void);','}']
        return '\n'.join(lines)+'\n'
