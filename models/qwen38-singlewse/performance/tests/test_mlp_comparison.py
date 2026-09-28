"""Reject scope/timing substitutions in a physical component comparison."""
import copy,importlib.util,json,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('compare_mlp',ROOT/'tools/compare_layer_mlp.py')
comparison=importlib.util.module_from_spec(spec);spec.loader.exec_module(comparison)


class MlpComparisonTests(unittest.TestCase):
    def setUp(self):
        self.before=json.loads((ROOT/'evidence/layer-mlp-hw-003/result.json').read_text())
        self.after=copy.deepcopy(self.before)
        for case in self.after['epochs']:
            case['controller_cycles']=600
            case['controller_phase_cycles']=dict(input=100,fused=200,output=300)
            case['completed_output_seconds']=case['host_arm_start_finish_seconds']+.001

    def test_cycle_comparison_does_not_claim_wall_speedup_or_tps(self):
        result=comparison.compare(self.before,self.after)
        self.assertEqual(len(result['cases']),4)
        self.assertEqual(result['cases'][0]['baseline_to_candidate_cycle_ratio'],2844472/600)
        self.assertIsNone(result['wall_time_speedup']);self.assertIsNone(result['full_model_tps'])

    def test_different_oracle_missing_case_and_incomplete_timing_rejected(self):
        for mutation in ('oracle','missing','order','nonphysical','drain','phase','wall'):
            bad=copy.deepcopy(self.after)
            if mutation=='oracle':bad['fixture_metadata_sha256']='different'
            elif mutation=='missing':bad['epochs'].pop()
            elif mutation=='order':bad['epochs'].reverse()
            elif mutation=='nonphysical':bad['physical']=False
            elif mutation=='drain':bad['epochs'][1]['all_pe_drained']=False
            elif mutation=='phase':bad['epochs'][0]['controller_phase_cycles']['input']=99
            else:bad['epochs'][0]['completed_output_seconds']=0
            with self.assertRaises(ValueError,msg=mutation):comparison.compare(self.before,bad)


if __name__=='__main__':unittest.main()
