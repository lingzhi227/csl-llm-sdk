"""Explicit multicast / acknowledgement IR and restricted-topology verifier.

The graph is a rooted spanning tree. Its bidirectional physical links are not
resource cycles: data uses one outward route; acknowledgement uses separate
queues/colors inward. Repeated epochs are root-credit controlled.
"""
from dataclasses import asdict, dataclass
import hashlib
import json


@dataclass(frozen=True)
class MeshPlan:
    width: int = 16
    height: int = 16
    max_words: int = 256
    data_color: int = 3
    row_colors: tuple = (4, 5)
    column_colors: tuple = (6, 7)
    input_queues: tuple = (2, 3, 4)
    output_queues: tuple = (2, 3)
    local_tasks: tuple = (8, 9, 10)
    stack_allowance: int = 4096
    code_allowance: int = 8192
    sram_ceiling: int = 48128

    def parent(self, x, y):
        return (x - 1, y) if x else ((0, y - 1) if y else None)

    def children(self, x, y):
        return ([(x + 1, y)] if x + 1 < self.width else []) + (
            [(0, y + 1)] if x == 0 and y + 1 < self.height else [])

    def routes(self):
        result = []
        for y in range(self.height):
            for x in range(self.width):
                tx = ([] if x == y == 0 else ['RAMP'])
                if x + 1 < self.width:
                    tx.append('EAST')
                if x == 0 and y + 1 < self.height:
                    tx.append('SOUTH')
                result.append(dict(x=x, y=y, color=self.data_color,
                                   rx='RAMP' if x == y == 0 else 'WEST' if x else 'NORTH', tx=tx))
                for cx, cy in self.children(x, y):
                    c = self.row_colors[cx % 2] if cx else self.column_colors[cy % 2]
                    result.append(dict(x=x, y=y, color=c,
                                       rx='EAST' if cx else 'SOUTH', tx=['RAMP']))
                parent = self.parent(x, y)
                if parent is not None:
                    c = self.row_colors[x % 2] if x else self.column_colors[y % 2]
                    result.append(dict(x=x, y=y, color=c, rx='RAMP', tx=['WEST' if x else 'NORTH']))
        return result

    def validate(self):
        if not (1 <= self.width <= 750 and 1 <= self.height <= 1160 and self.width * self.height > 1):
            raise ValueError('Mesh outside application geometry or missing transport')
        if self.width * self.height > 65535:
            raise ValueError('Benchmark checksum bound: at most 65535 endpoints')
        if not 1 <= self.max_words <= 8192:
            raise ValueError('Payload bound')
        colors = (self.data_color,) + tuple(self.row_colors) + tuple(self.column_colors)
        if len(self.row_colors) != 2 or len(self.column_colors) != 2 or len(set(colors)) != 5:
            raise ValueError('Independent stream colors required')
        # memcpy owns high colors/tasks and queues zero/one in this backend.
        if any(not 2 <= c <= 20 for c in colors):
            raise ValueError('Color outside unreserved backend pool')
        for ids, count, low, high in [(self.input_queues, 3, 2, 7), (self.output_queues, 2, 2, 7), (self.local_tasks, 3, 8, 20)]:
            if len(ids) != count or len(set(ids)) != count or any(not low <= i <= high for i in ids):
                raise ValueError('Queue/task conflict or reserved identifier')
        if min(self.stack_allowance, self.code_allowance) < 0 or self.sram_ceiling != 48128:
            raise ValueError('Invalid memory policy')
        # Payload + exported result/counters/timestamps + conservative scalar/DSD reserve.
        live_data_bytes = self.max_words * 4 + 512
        estimated = live_data_bytes + self.stack_allowance + self.code_allowance
        if estimated > self.sram_ceiling:
            raise ValueError('Estimated local SRAM overflow')
        routes = self.routes()
        seen = set()
        hops = 0
        for r in routes:
            key = (r['x'], r['y'], r['color'])
            if key in seen:
                raise ValueError('Concurrent route conflict')
            seen.add(key)
        # Explicitly trace both directions, checking each tree edge and root reachability.
        keyed = {(r['x'], r['y'], r['color']): r for r in routes}
        opposite = dict(EAST='WEST', WEST='EAST', SOUTH='NORTH', NORTH='SOUTH')
        steps = dict(EAST=(1, 0), WEST=(-1, 0), SOUTH=(0, 1), NORTH=(0, -1))
        for r in routes:
            for d in r['tx']:
                if d == 'RAMP':
                    continue
                dx, dy = steps[d]
                dest = keyed.get((r['x'] + dx, r['y'] + dy, r['color']))
                if dest is None or dest['rx'] != opposite[d]:
                    raise ValueError('Unmatched physical link')
                hops += 1
        nodes = self.width * self.height
        if hops != 2 * (nodes - 1):
            raise ValueError('Transport does not cover spanning tree exactly')
        return dict(passed=True, scope='restricted rooted multicast/ack tree only',
                    pes=nodes, routes=len(routes), tree_edges=nodes-1,
                    max_one_way_hops=self.width+self.height-2,
                    estimated_live_data_bytes=live_data_bytes,
                    estimated_total_bytes=estimated, compiled_sram_qualified=False,
                    root_window=1, payload_buffers=1,
                    completion='own receive and every child ack; parent send completion before buffer reuse')

    def document(self):
        return dict(schema='wse-stream-plan-v1', plan=asdict(self), resources=self.validate(), routes=self.routes())

    def fingerprint(self):
        return hashlib.sha256(json.dumps(self.document(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    def emit_layout(self):
        self.validate()
        lines = [f'const memcpy=@import_module("<memcpy/get_params>",.{{.width={self.width},.height={self.height}}});', 'layout {',
                 f'  @set_rectangle({self.width},{self.height});']
        for y in range(self.height):
            for x in range(self.width):
                children = len(self.children(x, y))
                params = dict(x=x, y=y, width=self.width, height=self.height, max_words=self.max_words,
                              child_count=children, data_color=self.data_color,
                              east_color=self.row_colors[(x+1)%2], south_color=self.column_colors[(y+1)%2],
                              ack_color=self.row_colors[x%2] if x else self.column_colors[y%2],
                              data_iq=self.input_queues[0], east_iq=self.input_queues[1], south_iq=self.input_queues[2],
                              data_oq=self.output_queues[0], ack_oq=self.output_queues[1],
                              receive_task=self.local_tasks[0], send_task=self.local_tasks[1], next_task=self.local_tasks[2])
                p = ','.join(f'.{k}={v}' for k,v in params.items())
                lines.append(f'  @set_tile_code({x},{y},"pe.csl",.{{.memcpy_params=memcpy.get_params({x}),{p}}});')
        for r in self.routes():
            tx = ','.join(r['tx'])
            lines.append('  @set_color_config(%d,%d,@get_color(%d),.{.routes=.{.rx=.{%s},.tx=.{%s}}});' % (r["x"],r["y"],r["color"],r["rx"],tx))
        lines.extend(['  @export_name("packet",[*]u32,false);', '  @export_name("audit",[*]u32,false);',
                      '  @export_name("ticks",[*]u16,false);', '  @export_name("benchmark",fn(u16,u16)void);', '}'])
        return '\n'.join(lines)+'\n'
