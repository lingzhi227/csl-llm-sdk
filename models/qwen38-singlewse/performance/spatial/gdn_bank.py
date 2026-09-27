"""Complete128x128 GDN state on atlas-compatible4x4 co-resident FP8 banks.

One controller and three passive PEs form the row above the state rectangle.
The prototype's row chain is a fixed qualified-order candidate, not a claim of
optimal routing. Request, prediction/output, delta and result planes are explicit.
"""
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class GdnBankPlan:
    width: int = 4
    height: int = 5
    fp8_slots: int = 111
    batch: int = 8
    request_color: int = 18
    delta_color: int = 19
    chain_colors: tuple = (7, 8)
    result_colors: tuple = (12, 13, 14, 15)

    def routes(self):
        out = []
        def add(x, y, color, rx, tx):
            out.append(dict(x=x, y=y, color=color, rx=rx, tx=tx))
        add(0, 0, self.request_color, 'RAMP', ['SOUTH'])
        for y in range(1, 5):
            for x in range(4):
                add(x, y, self.request_color, 'NORTH' if x == 0 else 'WEST',
                    ['RAMP'] + (['EAST'] if x < 3 else []) + (['SOUTH'] if x == 0 and y < 4 else []))
                add(x, y, self.delta_color, 'RAMP' if x == 0 else 'WEST',
                    (['RAMP'] if x else []) + (['EAST'] if x < 3 else []))
                if x:
                    color = self.chain_colors[x % 2]
                    add(x, y, color, 'RAMP', ['WEST'])
                    add(x - 1, y, color, 'EAST', ['RAMP'])
            color = self.result_colors[y - 1]
            for yy in range(y + 1):
                add(0, yy, color, 'RAMP' if yy == y else 'SOUTH', ['RAMP'] if yy == 0 else ['NORTH'])
        return out

    def validate(self):
        if (self.width, self.height, self.fp8_slots, self.batch) != (4, 5, 111, 8):
            raise ValueError('Pinned complete GDN/cohost profile')
        colors = [self.request_color, self.delta_color, *self.chain_colors, *self.result_colors]
        if len(set(colors)) != 8 or any(not 2 <= c <= 20 for c in colors):
            raise ValueError('Reserved or aliased color')
        routes = self.routes(); mapping = {(r['x'], r['y'], r['color']): r for r in routes}
        if len(mapping) != len(routes):
            raise ValueError('Route collision')
        steps = dict(NORTH=(0, -1, 'SOUTH'), SOUTH=(0, 1, 'NORTH'), WEST=(-1, 0, 'EAST'), EAST=(1, 0, 'WEST'))
        for r in routes:
            for direction in r['tx']:
                if direction != 'RAMP':
                    dx, dy, opposite = steps[direction]
                    target = mapping.get((r['x'] + dx, r['y'] + dy, r['color']))
                    if target is None or target['rx'] != opposite:
                        raise ValueError('Disconnected route')
        return dict(application_pes=20, state_pes=16, fp8_slots_per_state_pe=111,
                    state_elements=128 * 128, state_elements_per_pe=1024,
                    worker_input_queues=dict(chain=2, request=4, delta=5),
                    worker_output_queues=dict(chain_or_result=2, delta=3),
                    worker_microthreads=dict(chain=2, request=4, send=5, delta=6),
                    worker_dsrs=dict(request=2, chain=3, delta=5, send=6, math=[4, 7]),
                    controller_input_queues=[2, 3, 4, 5], controller_receive_microthreads=[2, 3, 4, 5],
                    controller_send_queue=2, controller_send_microthread=6,
                    shared_scratch_bytes=1548,
                    scratch_leases=[dict(name='state_request', byte_start=0, bytes=1548, phase='state'),
                                    dict(name='decoded_fp8', byte_start=260, bytes=512, phase='dot'),
                                    dict(name='encoded_dot_input', byte_start=0, bytes=260, phase='dot')],
                    phase_exclusion='Native FP8 dot requires !running; state request is armed only after dot completion. No state/FP8 overlap on the same PE.',
                    compiled_sram_admitted=False, full_model=False,
                    arithmetic='Each32-key native FMA partial then right-associated4-block FP32 chain; prediction and output separately reduced. FP32 state persists across all batches.',
                    schedule='Controller issues next preprocessed packet only after send callback and four tagged complete32-value results. Every worker send callback grants its next phase receive. Eight state-dependent steps per batch without host access.')

    def document(self):
        return dict(schema='wse-cohost-gdn-state-v1', plan=asdict(self), resources=self.validate(), routes=self.routes())

    def emit_layout(self):
        self.validate()
        lines = ['const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=5});', 'layout {', ' @set_rectangle(4,5);']
        for y in range(1, 5):
            for x in range(4):
                params = dict(key_shard=x, value_shard=y - 1, fp8_slots=self.fp8_slots,
                              request_color=self.request_color, delta_color=self.delta_color,
                              chain_in=self.chain_colors[(x + 1) % 2], chain_out=self.chain_colors[x % 2],
                              result_color=self.result_colors[y - 1])
                lines.append(' @set_tile_code(%d,%d,"worker.csl",.{.memcpy_params=memcpy.get_params(%d),%s});' %
                             (x, y, x, ','.join('.%s=%d' % (k, v) for k, v in params.items())))
        for x in range(4):
            lines.append(' @set_tile_code(%d,0,"controller.csl",.{.memcpy_params=memcpy.get_params(%d),.active=%s,.request_color=%d,.result_colors=.{%s}});' %
                         (x, x, 'true' if x == 0 else 'false', self.request_color, ','.join(map(str, self.result_colors))))
        for r in self.routes():
            lines.append(' @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});' %
                         (r['x'], r['y'], r['color'], r['rx'], ','.join(r['tx'])))
        for name, typ in [('weights', 'u32'), ('scale', 'f32'), ('state', 'f32'), ('dot_input', 'u32'), ('dot_output', 'f32'),
                          ('audit', 'u32'), ('requests', 'u32'), ('results', 'u32'), ('ticks', 'u16')]:
            lines.append(' @export_name("%s",[*]%s,false);' % (name, typ))
        lines += [' @export_name("reset",fn()void);', ' @export_name("initialize",fn(u16,u16)void);',
                  ' @export_name("dot",fn(u16)void);', ' @export_name("start",fn()void);', '}']
        return '\n'.join(lines) + '\n'
