"""Complete original dimensions and collision-free hierarchical output ownership."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.full_bf16_matrix import FullBf16MatrixPlan
from spatial.sdk_leases import audit_explicit_sdk_leases

class FullMatrixTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        e=Path(__file__).resolve().parents[1]/'evidence'
        cls.plan=FullBf16MatrixPlan(e/'model-atlas-001.json',e/'columnar-plan-001.json')

    def test_all_original_rows_and_columns_at_frozen_owner_addresses(self):
        p=self.plan
        self.assertEqual(p.matrix['shape'],[48,5120])
        self.assertEqual([(w['group'],w['k']) for w in p.workers],[(g,k) for g in range(24) for k in range(40)])
        self.assertEqual([w['eligible_rank'] for w in p.workers],list(range(502400,503360)))
        self.assertEqual({w['target_slot'] for w in p.workers},{6})
        self.assertEqual(p.workers[0]['global_xy'],[420,705]);self.assertEqual(p.workers[-1]['global_xy'],[120,706])
        self.assertEqual({(w['fp8_slots'],w['bf16_slots']) for w in p.workers},{(84,13),(110,13)})

    def test_routes_do_not_merge_simultaneous_inputs_or_reuse_colors(self):
        routes=self.plan.routes();keys=[(*r['xy'],r['color']) for r in routes]
        self.assertEqual(len(keys),len(set(keys)))
        self.assertTrue(all(r['rx'] in ['RAMP','NORTH','SOUTH','WEST','EAST'] for r in routes))
        self.assertEqual(sum(r['filtered'] for r in routes),960)
        self.assertEqual({r['color'] for r in routes},set(range(2,17))|{18})

    def test_non_power_of_two_gather_keeps_every_output_once_and_ordered(self):
        values={}
        for group in range(23,-1,-1):
            profile=self.plan.gather(group);value=[2*group,2*group+1]
            for stage in range(profile['steps']):value+=values[group+2**stage]
            self.assertEqual(len(value),profile['words']);values[group]=value
        self.assertEqual(values[0],list(range(48)))
        self.assertEqual(self.plan.gather(0),dict(steps=5,words=48,color=18))
        self.assertEqual(self.plan.gather(16),dict(steps=3,words=16,color=16))

    def test_sdk_overlap_in_actual_gather_or_microthread_is_rejected(self):
        p=Path(__file__).resolve().parents[1]/'probes/full_bf16_matrix/pe.csl'
        original=p.read_bytes()
        self.assertTrue(audit_explicit_sdk_leases({'pe.csl':original})['passed'])
        for good,bad in [(b'const gd=@get_dsr(dsr_dest,1)',b'const gd=@get_dsr(dsr_dest,0)'),
                         (b'@get_ut_id(6)',b'@get_ut_id(0)')]:
            self.assertIn(good,original)
            with self.assertRaisesRegex(ValueError,'overlaps SDK reservation'):
                audit_explicit_sdk_leases({'pe.csl':original.replace(good,bad)})
        with self.assertRaisesRegex(ValueError,'nonliteral'):
            audit_explicit_sdk_leases({'pe.csl':original.replace(b'const gd=@get_dsr(dsr_dest,1)',b'const gd=@get_dsr(dsr_dest,gather_index)')})
