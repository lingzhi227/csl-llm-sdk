from dataclasses import replace
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial.gdn_bank import GdnBankPlan


class GdnBankTests(unittest.TestCase):
    def test_broadcast_reaches_every_shard_and_results_reach_controller(self):
        plan = GdnBankPlan(); plan.validate()
        routes = {(r['x'], r['y'], r['color']): r for r in plan.routes()}
        steps = dict(NORTH=(0, -1), SOUTH=(0, 1), WEST=(-1, 0), EAST=(1, 0))
        def delivered(start, color):
            pending = [start]; seen = set(); receivers = set()
            while pending:
                x, y = pending.pop()
                self.assertNotIn((x, y), seen); seen.add((x, y))
                for d in routes[x, y, color]['tx']:
                    if d == 'RAMP':
                        receivers.add((x, y))
                    else:
                        dx, dy = steps[d]; pending.append((x + dx, y + dy))
            return receivers
        self.assertEqual(delivered((0, 0), plan.request_color), {(x, y) for x in range(4) for y in range(1, 5)})
        for y in range(1, 5):
            self.assertEqual(delivered((0, y), plan.delta_color), {(x, y) for x in range(1, 4)})
            self.assertEqual(delivered((0, y), plan.result_colors[y - 1]), {(0, 0)})
        self.assertEqual(plan.emit_layout().count('@set_tile_code'), 20)
        for invalid in [replace(plan, fp8_slots=110), replace(plan, delta_color=18), replace(plan, result_colors=(0, 13, 14, 15))]:
            with self.assertRaises(ValueError):
                invalid.validate()
