"""Full-chain host fences with family-local protocol coordinates.

The caller injects pinned family Boundary/observer implementations and plans.
No arithmetic reference or runtime allocation is performed by these checks.
"""
import copy
from checkpoint import control, hidden_edges


def require(ok, message):
    if not ok:
        raise ValueError(message)


class Boundary:
    def __init__(self, lowered, families):
        self.lowered = lowered
        self.regions = {r['layer']: r for r in lowered['regions']}
        self.families = families
        self.gates = {layer: families[r['family']]['module'].Boundary(
            families[r['family']]['plan'], families[r['family']]['transport'])
            for layer, r in self.regions.items()}

    def local(self, item):
        region = self.regions[item['layer_id']]
        result = copy.copy(item)
        result['x'] -= region['x']
        require(0 <= result['x'] < result['x'] + result['width'] <= region['width'],
                'Copy stays inside its original layer')
        require(item['family'] == region['family'], 'Exact selected layer family')
        return result, region

    def state(self, item, value, serial, generation, position, phase):
        import numpy as np
        local, region = self.local(item)
        require(phase in ('initialized', 'prepared', 'complete', 'reset'), 'Known command boundary')
        require(value.dtype == np.dtype('<u4') and
                value.shape == (local['height'], local['width'], local['count']), 'Exact state array')
        family = self.families[region['family']]
        plan = family['plan']
        expected = np.zeros(value.shape, dtype='<u4')
        expected[:, :, 0] = 2 * serial - (phase == 'prepared')
        expected[:, :, 1] = serial
        expected[:, :, 4] = generation
        expected[:, :, 5] = position + (phase in ('prepared', 'complete'))
        origin = plan['support']['origin'] if region['family'] == 'linear' else plan['origin']
        x, y = origin
        if local['x'] <= x < local['x'] + local['width'] and local['y'] <= y < local['y'] + local['height']:
            expected[y-local['y'], x-local['x'], 2] = 4 * (serial - (phase == 'prepared'))
        if region['family'] == 'attention':
            for x, y in plan['attention_heads']:
                if local['x'] <= x < local['x'] + local['width'] and local['y'] <= y < local['y'] + local['height']:
                    expected[y-local['y'], x-local['x'], 3] = position + (phase == 'complete')
        require(np.array_equal(value, expected), 'Every original-layer state/error/position: ' + phase)

    def observer(self, item, value, serial, previous_sequence):
        _, region = self.local(item)
        return self.families[region['family']]['module'].observer(value, serial, previous_sequence)

    def transport(self, item, value, serial, resets):
        local, region = self.local(item)
        return self.gates[region['layer']].transport(local, value, serial, resets)

    def lifecycle(self, item, value, serial):
        local, region = self.local(item)
        self.gates[region['layer']].lifecycle(local, value, serial)

    def handoff(self, item, value, serial, generation, position):
        import numpy as np
        _, region = self.local(item)
        point = item['x'], item['y']
        matching = [e for e in self.lowered['hidden_edges'] if
                    point in (tuple(e['source']), tuple(e['destination']))]
        require(len(matching) == 1 and value.shape == (1, 1, 12) and
                value.dtype == np.dtype('<u4'), 'One complete boundary status owner')
        edge = matching[0]
        expected = [0, 2*serial, 2*serial, 5120*serial, generation, position,
                    edge['source_layer'], edge['destination_layer'], 2, 1, 0, 1]
        require(value.reshape(-1).tolist() == expected, 'Header/payload complete and no active boundary transfer')
        return region['layer'], point

    def control(self, value, generation, next_position):
        control(value, self.lowered['regions'], self.lowered['request_id'],
                self.lowered['stage_id'], generation, next_position)

    def hidden_edges(self, retained):
        return hidden_edges(self.lowered['hidden_edges'], retained)

    def attention_metadata(self, retained, serial, generation, position):
        region = self.regions[3]
        family = self.families['attention']
        module, plan = family['module'], family['plan']
        def value(symbol, point):
            return [int(x) for x in retained[symbol, (point[0]+region['x'], point[1])]]
        for head, point in enumerate(plan['head_observers']):
            report = module.decode_head(value('observer', point), head, serial, position)
            require(report['coherent'] and report['field_bounds_valid'] and report['event'] == 11 and
                    report['Q_and_native_retained'] and report['QK_error'] == 0 and
                    report['attention_error'] == 0 and not report['protocol_error_latched'] and
                    not report['sink_protocol_error'] and not report['source_cap_reached'] and
                    report['archive_records_preceding_snapshot'] == serial,
                    'Every final physical attention head observer')
        for head, point in enumerate(plan['head_observers']):
            module.decode_archive(value('qk_archive_status', point), value('qk_archive', point),
                                  head=head, identity=[self.lowered['request_id'], generation, position,
                                                      self.lowered['stage_id'], 0], generation=serial)
        return dict(heads=24, archives=24, serial=serial, generation=generation,
                    position=position, passed=True)
