import copy
import json
from pathlib import Path
import sys
import unittest
from collections import deque
import random
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial.layer_fused_routes import compact_tree_groups, audit_fused_routes, translate_routes, root_parameters
from spatial.layer_routes import contraction_groups, edge_color


class FusedRoutesTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root = Path(__file__).resolve().parents[1]/'evidence'
        cls.plan = json.loads((root/'layer-native-schedule-002/layer-schedule.json').read_text())
        cls.regions = [r for s in cls.plan['stages'] for r in s['regions'] if r['role'] == 'gate_up']
        cls.templates = {tuple(r['rect'][2:]): r for r in
                         (json.loads(f.read_text()) for f in (root/'layer-fused-routes-003').glob('routes-*.json'))}

    def test_all64_composed_routes_keep_original_owners_and_reachable_credits(self):
        originals = {}; flows = blocks = 0
        for region in self.regions:
            shape = tuple(region['rect'][2:]); original = originals.setdefault(shape, region)
            result = translate_routes(self.templates[shape], original, region)
            self.assertTrue(result['audit']['passed']); self.assertFalse(result['executed'])
            flows += result['audit']['flows']; blocks += result['audit']['original_projected_blocks']
        self.assertEqual(flows, 4032); self.assertEqual(blocks, 64*17408//8)

    def test_interval_coloring_does_not_change_arithmetic_tree(self):
        for region in [self.regions[0], self.regions[3]]:
            for before, after in zip(contraction_groups(region), compact_tree_groups(region)):
                for a, b in zip(before['nodes'], after['nodes']):
                    b = dict(b); self.assertLessEqual(b.pop('route_color', 3), 9); self.assertEqual(a, b)

    def test_root_ports_bind_each_prefix_and_return_to_correct_stream(self):
        for region in [self.regions[0], self.regions[3]]:
            routes = self.templates[tuple(region['rect'][2:])]
            for group in compact_tree_groups(region):
                p = root_parameters(routes, group); first = group['output_start']*8
                self.assertEqual(p['boundary_rows'], (128-first%128)//8 if first%128 else 0)
                for f in routes['flows']:
                    if f['producer'] != group['index']: continue
                    main = f['actor'] == group['index']
                    color = f['routes'][0]['color'] if f['kind'] == 'projection' else f['routes'][-1]['color']
                    parameter = ('out_color' if main else 'boundary_out_color') if f['kind'] == 'projection' else ('main_credit_color' if main else 'boundary_credit_color')
                    self.assertEqual(p[parameter], color)

    def test_unsupported_swap_and_missing_projection_are_rejected(self):
        region = self.regions[0]; routes = self.templates[tuple(region['rect'][2:])]
        bad = copy.deepcopy(routes)
        record = bad['flows'][0]['routes'][0]
        record['color_swap_x'] = True
        with self.assertRaises((ValueError, KeyError)): audit_fused_routes(region, bad)
        bad = copy.deepcopy(routes); bad['flows'].pop()
        with self.assertRaises((ValueError, KeyError)): audit_fused_routes(region, bad)

    def test_small_boundary_buffer_breaks_cross_root_head_of_line_chain(self):
        """Independent token-flow model, not numerical CSL or device execution.

        Pause producer0. With one group buffer producer1 cannot release its first
        block. With separate early suffix storage every other producer drains;
        resume0 and all136 original groups contain each original row exactly once.
        """
        region = self.regions[0]; routes = self.templates[tuple(region['rect'][2:])]
        groups = list(contraction_groups(region)); owner = {q:i for i,a in enumerate(routes['actors']) for q in a['groups']}
        def run(early_enabled):
            cursor = [0]*len(groups); index = [0]*len(groups); active = [set() for _ in groups]; early = [set() for _ in groups]
            completed = {}; maximum_early = 0
            def advance(i):
                while index[i] < len(routes['actors'][i]['groups']) and len(active[i]) == 128:
                    q = routes['actors'][i]['groups'][index[i]]
                    self.assertEqual(active[i], set(range(q*128,(q+1)*128)))
                    self.assertNotIn(q, completed); completed[q] = active[i]
                    index[i] += 1; active[i] = set()
                    if index[i] == len(routes['actors'][i]['groups'])-1:
                        active[i], early[i] = early[i], set()
            def drain(paused):
                nonlocal maximum_early
                while True:
                    progressed = False
                    # Reverse arrival order stresses early suffixes before all
                    # of the preceding producer's lower groups exist.
                    for producer in reversed(range(len(groups))):
                        g = groups[producer]
                        if producer in paused or cursor[producer] == g['output_count']: continue
                        first = (g['output_start']+cursor[producer])*8; q = first//128; actor = owner[q]
                        actor_groups = routes['actors'][actor]['groups']; current = actor_groups[index[actor]]
                        values = set(range(first,first+8))
                        if q == current: active[actor] |= values
                        elif early_enabled and q == actor_groups[-1] and producer == actor+1:
                            early[actor] |= values; maximum_early = max(maximum_early, len(early[actor])*2)
                        else: continue
                        cursor[producer] += 1; progressed = True; advance(actor)
                    if not progressed: break
            drain({0}); before = cursor[:]; drain(set())
            return before, cursor, completed, maximum_early
        stalled, _, _, _ = run(False); self.assertEqual(stalled[1], 0)
        before, after, completed, maximum = run(True)
        self.assertEqual(before[0], 0)
        self.assertEqual(before[1:], [g['output_count'] for g in groups[1:]])
        self.assertEqual(after, [g['output_count'] for g in groups]); self.assertEqual(len(completed), 136)
        self.assertLessEqual(maximum, 256); self.assertGreater(maximum, 0)

    def test_private_dma_buffer_reuse_with_bounded_fifo_and_delayed_consumer(self):
        """Ownership model: local copying can finish before the consumer drains.

        Every word is an immutable queue value; queue colors/routes never change.
        Deliberately destroying the source arena after the last copied word must
        preserve all67 eleven-word packets despite a two-word bounded FIFO.
        This tests the protocol assumption, not WSE timing or numeric execution.
        """
        randomizer = random.Random(20260928); queue = deque(); actual = []; partial = []
        iteration = 0; offset = 0; arena = [iteration*100+i for i in range(11)]
        local_done_before_drain = False
        for _ in range(100000):
            if iteration < 67 and len(queue) < 2 and randomizer.randrange(4):
                queue.append(arena[offset]); offset += 1
                if offset == 11:
                    local_done_before_drain |= len(actual) <= iteration
                    arena[:] = [-1]*11
                    iteration += 1; offset = 0; arena[:] = [iteration*100+i for i in range(11)]
            if queue and randomizer.randrange(5) == 0:
                partial.append(queue.popleft())
                if len(partial) == 11: actual.append(partial); partial = []
            self.assertLessEqual(len(queue), 2)
            if len(actual) == 67: break
        self.assertTrue(local_done_before_drain)
        self.assertEqual(actual, [[row*100+i for i in range(11)] for row in range(67)])
        self.assertFalse(queue); self.assertFalse(partial)

    def test_whole_region_compiler_roles_are_bound_to_actual_fabric_endpoints(self):
        evidence = Path(__file__).resolve().parents[1]/'evidence'
        profiles = json.loads((evidence/'layer-projection-compile-012/source/profiles.json').read_text())['profiles']
        routes = self.templates[(78,67)]
        installed = {(*r['pe'], r['color']): r for f in routes['reduction_trees']+routes['flows'] for r in f['routes']}
        seen = set(); actors = 0
        for profile in profiles:
            p = profile['parameters']
            for pe in profile['pes']:
                self.assertNotIn(tuple(pe),seen);seen.add(tuple(pe));self.assertTrue(p['copy_transport'])
                self.assertEqual(installed[(*pe,p['out_color'])]['rx'],'RAMP')
                inbound = []
                if p['children']:inbound.append(p['left_color'])
                if p['children']>1:inbound.append(p['right_color'])
                if p.get('boundary_rows',0):self.assertEqual(installed[(*pe,p['boundary_out_color'])]['rx'],'RAMP')
                if p.get('fusion_actor'):
                    actors+=1;self.assertFalse(p['root']);inbound.append(p['actor_main_color'])
                    if p['actor_has_suffix']:inbound.append(p['actor_suffix_color'])
                self.assertEqual(len(inbound),len(set(inbound)))
                for color in inbound:
                    tx = installed[(*pe,color)]['tx'];self.assertIn('RAMP',tx if isinstance(tx,list) else [tx])
        self.assertEqual(len(seen),5226);self.assertEqual(actors,33)


if __name__ == '__main__': unittest.main()
