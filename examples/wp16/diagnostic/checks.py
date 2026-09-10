"""Unexecuted WP16 diagnostic driver checks; no SDK import or numerical operands H2D."""
import math
import numpy as np

from qwen38.mlp_numerics import FP32_TINY, Interval, intermediate_product, silu
from qwen38.mlp_reference_checks import decode, encode, rounded
from qwen38.mlp_streamed_source import linear_bounds
from qwen38.wp16_diag_protocol import descriptor, packet, verify_ack
from qwen38.wp16_diag_layout import tile_snapshot


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f64_of_f32(array):
    """Read exact finite FP32 bits, including subnormals, without FP32 arithmetic."""
    require(array.dtype == np.float32, 'Observed FP32 array required')
    words = array.view(np.uint32)
    exponent = ((words >> 23) & 255).astype(np.int32)
    require(not np.any(exponent == 255), 'Nonfinite device FP32 value')
    fraction = (words & 0x7fffff).astype(np.int64)
    significant = np.where(exponent == 0, fraction, fraction + (1 << 23))
    power = np.where(exponent == 0, -149, exponent - 150)
    values = np.ldexp(significant.astype(np.float64), power)
    return np.copysign(values, np.where(words & 0x80000000, -1., 1.))


def bits_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left.view(np.uint32), right.view(np.uint32))


def inside(values, bounds, label):
    require(bounds.shape == (values.size, 2), label + ': bound shape')
    require(np.isfinite(values).all() and np.isfinite(bounds).all(), label + ': finite values')
    rejected = (values < bounds[:, 0]) | (values > bounds[:, 1])
    require(not np.any(rejected), label + ': ' + str(int(np.count_nonzero(rejected))) + ' interval violations')


def numeric_state(pe, *, phase, tiles=0, consumed=0, committed=False, released=False):
    total = 5120 if pe < 2 else 128
    last = (80 if consumed == 5120 else 112) if pe < 2 else (128 if pe == 2 else 64)
    active = tiles > 0 or phase == 1
    return [phase, 1, 0, 1, total, tiles, consumed, int(committed), 0,
            last if tiles else 0, tiles, int(active or committed), int(committed),
            int(released), pe, 0]


def key(phase, name, pe, tile=None):
    return phase + '__' + name + '__pe' + str(pe) + ('__tile' + str(tile) if tile is not None else '')


class Checks:
    def __init__(self, oracle, weights, prefixes=None):
        self.oracle = oracle
        self.weights = weights
        self.prefixes = prefixes
        self.observed = {}
        self.uploaded = {}
        self.pending_weights = {}
        self.next_tile = {0: 0, 1: 0, 3: 0}
        self.validated = set()
        self.reports = dict(handoffs=[], nonlinear=None, down=[], casts=[], retention=[], timing=[], tile_groups=[])

    def upload(self, item, values):
        name, pe = item['name'], item['pe']
        if name == 'weights_storage':
            require(pe in self.next_tile and pe not in self.pending_weights, 'Weight overwrite before validated tile observation group')
            require(item['tile'] == self.next_tile[pe], 'Exactly next tile weight upload')
            phase = 'projection_tile_readback' if pe < 2 else 'down_tile_readback'
            self.pending_weights[pe] = (phase, item['tile'])
        self.uploaded[name, pe] = values.copy()

    def get(self, phase, name, pe, tile=None):
        return self.observed[key(phase, name, pe, tile)]

    def bf16_body(self, phase, name, pe, tile=None):
        array = self.get(phase, name, pe, tile)
        require(array.dtype == np.uint32 and array.shape == (130,), 'Guarded BF16 shape/type')
        require(array[0] == 0xa55a and array[-1] == 0x5aa5 and not np.any(array >> 16), 'Guarded BF16 integrity')
        words = array[1:-1].astype('<u2')
        decode(words)
        return words

    def float_body(self, phase, name, pe, tile=None):
        array = self.get(phase, name, pe, tile)
        require(array.dtype == np.float32 and array.shape == (130,), 'Guarded FP32 shape/type')
        require(array[0] == 1234.5 and array[-1] == -4321.25, 'Guarded FP32 integrity')
        return f64_of_f32(array[1:-1])

    def check_cast(self, phase, bf16_name, fp32_name, pe):
        actual = self.bf16_body(phase, bf16_name, pe)
        expected = encode(rounded(self.float_body(phase, fp32_name, pe)))
        require(np.array_equal(actual, expected), phase + ': exact observed FP32 to BF16 RNE')
        self.reports['casts'].append(dict(phase=phase, pe=pe, source=fp32_name,
            destination=bf16_name, count=128, bit_mismatches=0))

    def expected_final_state(self, pe, *, released=False):
        tiles = 46 if pe < 2 else (1 if pe == 2 else 2)
        return numeric_state(pe, phase=4 if released else 2, tiles=tiles,
            consumed=5120 if pe < 2 else 128, committed=True, released=released)

    def check_timing(self, array, pe, phase):
        require(array.dtype == np.uint32 and array.size % 6 == 0 and not np.any(array >> 16), 'Native16 timestamp words')
        cycles = []
        for row in array.reshape(-1, 6).tolist():
            start = row[0] + (row[1] << 16) + (row[2] << 32)
            end = row[3] + (row[4] << 16) + (row[5] << 32)
            delta = (end - start) & ((1 << 48) - 1)
            require(0 < delta < 1 << 40, 'Positive bounded device timestamp delta')
            cycles.append(delta)
        self.reports['timing'].append(dict(pe=pe, phase=phase, cycles=cycles))

    def handoff(self, edge):
        sender, receiver = edge, 2 if edge < 2 else 3
        phase = 'handoff_' + str(edge)
        tensor = ('mlp_gate', 'mlp_up', 'mlp_product')[edge]
        if edge < 2:
            body = self.bf16_body('finalize_projections', 'bf16_storage', sender)
        else:
            body = self.bf16_body('whole_silu_and_product', 'mlp_product', 2)
        expected = np.array(packet(edge, body.astype(np.uint32).tolist()), dtype=np.uint32)
        orders = []
        for pe in (sender, receiver):
            require(np.array_equal(self.get(phase, 'wire_data', pe), expected), 'Exact sender/receiver data frame')
            verify_ack(edge, self.get(phase, 'wire_ack', pe).tolist())
            require(self.get(phase, 'handoff_latched', pe).tolist() == list(descriptor(edge)), 'Latched complete descriptor')
            state = self.get(phase, 'handoff_state', pe)
            require(state.shape == (32,) and state[0] == 4 and state[1:5].tolist() == [1, 0, 1, edge], 'Completed handoff identity')
            require(state[6] == 1 and state[11] == 1 and state[12] == 0 and state[13:15].tolist() == [140, 13], 'Command/word/error handoff gates')
            require(state[24:29].tolist() == [edge, 0, 128, 0, 128] and state[29] == 0 and state[31] == 1, 'Handoff range/retention metadata')
            require(state[17] > 0 and state[18] > state[17] and state[23] > state[18], 'Arm/command/unblock event order')
            require(state[16] == state[23], 'Final event counter equals command unblock')
            if pe == sender:
                require(state[9] == 1 and state[10] == 1 and state[22] > state[18] and state[23] > state[22], 'Sender data/ACK join')
                require(state[7:9].tolist() == [0, 0] and state[19:22].tolist() == [0, 0, 0] and state[30] == 0, 'Sender inactive receiver join fields')
                require(state[5] == (3 if edge == 2 else 0) and state[15] == (2 if edge == 2 else 0), 'Sender retained incoming mask/count')
            else:
                require(state[7] == 1 and state[8] == 1 and state[19] > state[17] and state[20] > state[19] and state[21] > state[20] and state[23] > state[21], 'Receiver receipt/commit/ACK join')
                require(state[9:11].tolist() == [0, 0] and state[22] == 0, 'Receiver inactive sender join fields')
                expected_mask = 1 if edge == 0 else (3 if edge == 1 else 4)
                require(state[5] == expected_mask and state[15] == (2 if edge == 1 else 1), 'Exact incoming commit mask/count')
                receipt_first = int(state[19]) < int(state[18])
                require(int(state[30]) == int(receipt_first), 'Observed join-order flag')
                orders.append('receipt_before_command' if receipt_first else 'command_before_receipt')
        destination = self.bf16_body(phase, tensor, receiver)
        require(np.array_equal(destination, body), 'Exact device-owned consumer operands')
        mlp = self.get(phase, 'mlp_state', receiver)
        require(mlp[0] == 1 and mlp[1:4].tolist() == [1, 0, 1] and mlp[4] == (1 if edge == 0 else (3 if edge == 1 else 4)) and mlp[12] == 0, 'Receiver compute validity after commit')
        if edge == 2:
            require(mlp[7] == 128 and mlp[11] == 0, 'Down input committed before arithmetic')
        self.reports['handoffs'].append(dict(edge=edge, payload_count=128,
            exact_frame_words=142, observed_receiver_orders=orders, protocol_passed=True))

    def nonlinear(self):
        phase = 'whole_silu_and_product'
        for edge, name in ((0, 'mlp_gate'), (1, 'mlp_up')):
            require(bits_equal(self.get(phase, name, 2), self.get('handoff_' + str(edge), name, 2)),
                    'Exact received operand retained through nonlinear compute: ' + name)
        gate = decode(self.bf16_body(phase, 'mlp_gate', 2))
        upper = decode(self.bf16_body(phase, 'mlp_up', 2))
        act = decode(self.bf16_body(phase, 'mlp_silu', 2))
        prod = decode(self.bf16_body(phase, 'mlp_product', 2))
        observed = {name: self.float_body(phase, name, 2) for name in
                    ('mlp_exp', 'mlp_sigmoid', 'mlp_silu_fp32', 'mlp_product_fp32')}
        require(np.max(np.abs(gate)) <= 24 and np.max(np.abs(upper)) <= 24, 'Device nonlinear domain')
        inside(gate, self.oracle['gate_bf16'], 'Original gate source')
        inside(upper, self.oracle['up_bf16'], 'Original up source')
        inside(act, self.oracle['silu_bf16'], 'Original whole SiLU source')
        inside(prod, self.oracle['product_bf16'], 'Original product source')
        inside(observed['mlp_silu_fp32'], self.oracle['silu_fp32'], 'Source FP32 whole SiLU')
        inside(observed['mlp_product_fp32'], self.oracle['product_fp32'], 'Source FP32 product')
        for i, g in enumerate(gate):
            exponential = math.exp(-abs(float(g)))
            e_allowance = exponential * (2.**-20 + 2.**-44) + FP32_TINY
            require(abs(observed['mlp_exp'][i] - exponential) <= math.nextafter(e_allowance, math.inf), 'Actual-gate bounded exp allowance')
            ideal_sigmoid = exponential / (1. + exponential) if g < 0 else 1. / (1. + exponential)
            s_allowance = ideal_sigmoid * (2.**-18 + 2.**-44) + FP32_TINY
            require(abs(observed['mlp_sigmoid'][i] - ideal_sigmoid) <= math.nextafter(s_allowance, math.inf), 'Actual-gate stable sigmoid allowance')
            activation = silu(Interval.point(float(g)))
            multiplication = intermediate_product(Interval.point(float(act[i])), Interval.point(float(upper[i])))
            for stage, value in ((activation, observed['mlp_silu_fp32'][i]), (multiplication, observed['mlp_product_fp32'][i])):
                require(stage['fp32'].lo <= value <= stage['fp32'].hi, 'Conditional actual-operand FP32 arithmetic')
        self.check_cast(phase, 'mlp_silu', 'mlp_silu_fp32', 2)
        self.check_cast(phase, 'mlp_product', 'mlp_product_fp32', 2)
        mlp = self.get(phase, 'mlp_state', 2)
        require(mlp.tolist() == [2, 1, 0, 1, 3, 128, 128, 0, 0, 0, 128, 1, 0, 0, 2, 0], 'Exact nonlinear compute commit/counts')
        self.reports['nonlinear'] = dict(count=128, source_passed=True, conditional_passed=True,
            observed_FP32_stages=4, extra_sigmoid_BF16_boundary=False,
            cpu_silu_bit_mismatches=int(np.count_nonzero(self.bf16_body(phase, 'mlp_silu', 2) != self.oracle['cpu_silu_bits'])),
            cpu_product_bit_mismatches=int(np.count_nonzero(self.bf16_body(phase, 'mlp_product', 2) != self.oracle['cpu_product_bits'])))

    def down_output(self, phase, tile=None):
        columns = 64 * (tile + 1) if tile is not None else 128
        values = self.float_body(phase, 'output_storage', 3, tile)
        name = 'partial_down_prefix64_fp32' if columns == 64 else 'partial_down_fp32'
        inside(values, self.oracle[name], 'Original128-channel partial down source')
        product = decode(self.bf16_body('handoff_2', 'mlp_product', 3))[:columns]
        points = np.column_stack((product, product))
        conditional = np.empty((128, 2), dtype=np.float64)
        conditional_bf16 = np.empty((128, 2), dtype=np.float64)
        for first in (0, 64):
            result = linear_bounds(decode(self.weights['down_proj'][first:first + 64, :columns]), points)
            conditional[first:first + 64] = result['fp32']
            conditional_bf16[first:first + 64] = result['bf16']
        inside(values, conditional, 'Conditional device-product partial down')
        self.reports['down'].append(dict(phase=phase, consumed_columns=columns,
            source_and_conditional_passed=True, outputs=128, BF16_prefix_boundary=False if columns == 64 else None))
        return conditional_bf16

    def receive(self, item, values):
        phase, name, pe, tile = (item[k] for k in ('phase', 'name', 'pe', 'tile'))
        tag = key(phase, name, pe, tile)
        require(tag not in self.observed, 'Duplicate diagnostic observation identity')
        self.observed[tag] = values
        if name == 'weights_storage':
            require(bits_equal(values, self.uploaded[name, pe]), 'Exact original uploaded weight readback')
        if name == 'input_storage':
            if pe < 2:
                require(bits_equal(values, self.uploaded[name, pe]), 'Exact original input/padding/guards')
            else:
                words = self.bf16_body('handoff_2', 'mlp_product', 3)
                offset = 64 * (tile if tile is not None else 1)
                expected = np.zeros(66, dtype=np.float32)
                expected[0], expected[-1] = 77.5, -88.25
                expected.view(np.uint32)[1:-1] = words[offset:offset + 64].astype(np.uint32) << 16
                require(bits_equal(values, expected), 'Exact device BF16 to down input expansion')
        if name == 'state':
            if phase == 'initialize':
                expected = numeric_state(pe, phase=0)
            elif phase in ('projection_tile_readback', 'down_tile_readback'):
                columns = min((tile + 1) * 112, 5120) if pe < 2 else (tile + 1) * 64
                expected = numeric_state(pe, phase=1, tiles=tile + 1, consumed=columns)
            else:
                expected = self.expected_final_state(pe, released=phase == 'released')
            require(values.tolist() == expected, phase + ': exact state/commit/counts')
        if name == 'weight_guard_snapshot':
            require(values.tolist() == [0xa55a, 0x5aa5, 1, 1, 2], 'Final weight guards precede numerical commit')
        if name == 'tile_guard_snapshot':
            require(values.dtype == np.uint32 and values.shape == (12,) and values.tolist() == tile_snapshot(pe, tile),
                    'Exact pre/post weight guards and tile identity/commit snapshot')
            require(self.pending_weights.get(pe) == (phase, tile), 'Matching pending weight observation group')
            required = ['input_storage', 'output_storage', 'state']
            full_payload = pe == 3 or tile in (0, 45)
            if full_payload:
                required.append('weights_storage')
            require(all(key(phase, field, pe, tile) in self.validated for field in required),
                    'All tile observation checks before weight overwrite')
            del self.pending_weights[pe]
            self.next_tile[pe] += 1
            self.reports['tile_groups'].append(dict(pe=pe, tile=tile, consumed_columns=values[9].item(),
                source_FP32_and_exact_input_state_guards=True, full_weight_payload_bitwise_D2H=full_payload))
        if name == 'timing':
            self.check_timing(values, pe, phase)
            if phase == 'whole_silu_and_product':
                self.nonlinear()
        if name == 'output_storage' and phase in ('projection_tile_readback', 'finalize_projections'):
            stage = 'gate' if pe == 0 else 'up'
            if phase == 'projection_tile_readback':
                require(self.prefixes is not None and 0 <= tile < 46, 'Frozen all-prefix source required')
                require(int(self.prefixes['prefix_columns'][tile]) == min(112*(tile+1),5120), 'Exact consumed prefix columns')
                bounds = self.prefixes[stage + '_prefix_fp32'][tile]
            else:
                bounds = self.oracle[stage + '_fp32']
            inside(self.float_body(phase, name, pe, tile), bounds, 'Original projection FP32 source')
        if name == 'bf16_storage' and phase == 'finalize_projections':
            stage = 'gate' if pe == 0 else 'up'
            self.check_cast(phase, name, 'output_storage', pe)
            inside(decode(self.bf16_body(phase, name, pe)), self.oracle[stage + '_bf16'], 'Original projection BF16 source')
        if phase.startswith('handoff_') and name == 'mlp_state':
            self.handoff(int(phase[-1]))
        if pe == 3 and name == 'output_storage' and phase == 'down_tile_readback':
            self.down_output(phase, tile)
        if pe == 3 and name == 'bf16_storage' and phase == 'finalize_down':
            self.check_cast(phase, name, 'output_storage', pe)
            values64 = decode(self.bf16_body(phase, name, pe))
            inside(values64, self.oracle['partial_down_bf16'], 'Final partial source BF16')
            inside(values64, self.down_output(phase), 'Final actual-product conditional BF16')
        if phase == 'retention_after_down':
            if name == 'handoff_state':
                edge = pe if pe < 2 else 2
                original = self.get('handoff_' + str(edge), name, pe)
            elif pe == 2:
                original = self.get('whole_silu_and_product', name, pe)
            elif name in ('weights_storage', 'input_storage'):
                original = self.get('projection_tile_readback' if pe < 2 else 'down_tile_readback', name, pe, 45 if pe < 2 else 1)
            elif name == 'mlp_product':
                original = self.get('handoff_2', name, 3)
            else:
                original = self.get('finalize_projections' if pe < 2 else 'finalize_down', name, pe)
            # Down finalize changes only its numeric state, not the handoff state.
            require(bits_equal(values, original), 'Retained data/state changed after downstream completion')
            self.reports['retention'].append(dict(name=name, pe=pe, words=values.size))
        if phase == 'released' and name == 'handoff_state':
            expected = self.get('retention_after_down', name, pe).copy()
            expected[0] = 5;expected[29] += 1
            require(bits_equal(values, expected), 'Release after global completion and ACK retention')
        if phase == 'initialize' and name == 'handoff_state':
            expected = np.zeros(32, dtype=np.uint32);expected[1] = 1;expected[3] = 1
            require(bits_equal(values, expected), 'Initialized empty handoff state')
        self.validated.add(tag)

    def finish(self):
        require(not self.pending_weights and self.next_tile == {0: 46, 1: 46, 3: 2}, 'Complete validated tile groups before overwrite')
        require(len(self.reports['tile_groups']) == 94, 'All92projection and2down observation groups')
        require(len(self.reports['handoffs']) == 3 and len(self.reports['retention']) == 29, 'Complete handoff/retention coverage')
        require(len(self.reports['casts']) == 5 and len(self.reports['down']) == 3, 'All casts and both down prefixes/final gates')
        return self.reports
