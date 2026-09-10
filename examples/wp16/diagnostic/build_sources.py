"""Generate role-specialized CSL source/layout from the reviewed interface proposal."""
from pathlib import Path
import json
import shutil
import sys

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
from qwen38.wp16_diag_layout import ARRAYS, EDGES, FLOAT32, budget, sequence


def scalar_type(name, bits):
    return 'u16' if bits == 16 else ('f32' if name in FLOAT32 else 'u32')


def main():
    header = ['// Generated source proposal. No SDK compile/simulation has run.',
              'param rank:u16;param memcpy_params;',
              'comptime{@comptime_assert(rank<4);}',
              'const sys=@import_module("<memcpy/memcpy>",memcpy_params);',
              'const clock=@import_module("<time>");',
              'const gemv=@import_module("gemv_kernel.csl",.{.rows=128,.columns=if(rank==3) @as(u16,64) else @as(u16,112)});',
              'const nonlinear=@import_module("nonlinear_kernel.csl");',
              'const protocol=@import_module("protocol.csl");']
    for name, (bits, counts) in ARRAYS.items():
        kind = scalar_type(name, bits)
        expr = str(counts[0]) if len(set(counts)) == 1 else ''.join(
            'if(rank==' + str(i) + ') @as(u16,' + str(counts[i]) + ') else ' for i in range(3)) + '@as(u16,' + str(counts[3]) + ')'
        header += ['const ' + name + '_count:u16=' + expr + ';',
                   'var ' + name + '=@zeros([' + name + '_count]' + kind + ');',
                   'const ptr_' + name + ':[*]' + kind + '=&' + name + ';']
    header += ['const weight_words:u16=weights_storage_count;',
               'const input_words:u16=input_storage_count;',
               'const timing_words:u16=timing_count;']
    launches = list(dict.fromkeys(item['name'] for item in sequence() if item['kind'] == 'launch'))
    footer = ['comptime{', ' @bind_local_task(received_complete,receive_task);',
              ' @bind_local_task(sent_complete,send_task);']
    for name in ARRAYS:
        footer.append(' @export_symbol(ptr_' + name + ',"' + name + '");')
    for name in launches:
        footer.append(' @export_symbol(' + name + ');')
    for pe in range(4):
        footer.append(' if(rank==' + str(pe) + '){')
        queue_bindings = set()
        for edge in EDGES:
            route = edge['forward_route']
            if pe not in route:
                continue
            position = route.index(pe)
            rx = 'RAMP' if position == 0 else 'WEST'
            tx = 'RAMP' if position == len(route) - 1 else 'EAST'
            ack_rx = 'RAMP' if position == len(route) - 1 else 'EAST'
            ack_tx = 'RAMP' if position == 0 else 'WEST'
            for color, source, destination in ((edge['data_color'], rx, tx), (edge['ack_color'], ack_rx, ack_tx)):
                footer.append('  @set_local_color_config(@get_color(' + str(color) + '),.{.routes=.{.rx=.{' + source + '},.tx=.{' + destination + '}}});')
            if pe == edge['sender']:
                queue_bindings.add(('output', edge['sender_queue'], edge['data_color']))
                queue_bindings.add(('input', edge['sender_queue'], edge['ack_color']))
            if pe == edge['receiver']:
                queue_bindings.add(('input', edge['receiver_queue'], edge['data_color']))
                queue_bindings.add(('output', edge['receiver_queue'], edge['ack_color']))
        for direction, queue, color in sorted(queue_bindings):
            if sum(d == direction and q == queue for d, q, _ in queue_bindings) != 1:
                raise ValueError('Conflicting local queue color binding')
            footer.append('  @initialize_queue(@get_' + direction + '_queue(' + str(queue) + '),.{.color=@get_color(' + str(color) + ')});')
        footer.append(' }')
    footer.append('}')
    body = (HERE / 'tile_body.csl.inc').read_text()
    (HERE / 'tile.csl').write_text('\n'.join(header) + '\n' + body + '\n' + '\n'.join(footer) + '\n')
    layout = ['// Source-only four-PE partial MLP diagnostic.',
              'const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=1});',
              'layout{', ' @set_rectangle(4,1);',
              ' for(@range(i16,4))|x|{',
              '  @set_tile_code(x,0,"tile.csl",.{.memcpy_params=memcpy.get_params(x),.rank=@as(u16,x)});', ' }']
    for name, (bits, _) in ARRAYS.items():
        layout.append(' @export_name("' + name + '",[*]' + scalar_type(name, bits) + ',false);')
    for name in launches:
        layout.append(' @export_name("' + name + '",fn()void);')
    layout.append('}')
    (HERE / 'layout.csl').write_text('\n'.join(layout) + '\n')
    interfaces = json.loads((HERE/'interfaces.json').read_bytes())
    interfaces['arrays'] = {name:dict(bits=bits, counts=list(counts)) for name,(bits,counts) in ARRAYS.items()}
    interfaces['field_indices']['tile_guard_snapshot'] = {name:i for i,name in enumerate((
        'head_before','tail_before','head_after','tail_after','generation','tile_index',
        'consumed_before','valid_columns','completed_tiles','consumed_after','pre_guard_ok','post_guard_ok'))}
    for name,value in [('interfaces.json',interfaces),('physical-sequence.json',sequence()),('resource-plan.json',budget())]:
        (HERE/name).write_text(json.dumps(value,indent=2)+'\n')
    for source, target in [('persistent_gemv_bf16_f32.csl', 'gemv_kernel.csl'),
                           ('exp_bounded_f32.csl', 'exp_kernel.csl'),
                           ('mlp_silu_product128_bf16.csl', 'nonlinear_kernel.csl')]:
        shutil.copyfile(ROOT / 'csl/kernels' / source, HERE / target)
    print(dict(exported_arrays=len(ARRAYS), exported_functions=len(launches),
               compute_pes=4, compiled=False, simulator_started=False))


if __name__ == '__main__':
    main()
