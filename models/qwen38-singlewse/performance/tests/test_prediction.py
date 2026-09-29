import copy
import json
from pathlib import Path
import sys
import unittest

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'performance'))
from prediction.schedule import simulate
from prediction.model import analyze,boundary_traffic,read_json,evaluate_scenario,service_template


def stage(name,latency,ii=None,slots=2):
    return dict(id=name,latency_cycles=latency,ii_cycles=latency if ii is None else ii,slots=slots)


class QueueModelTests(unittest.TestCase):
    def test_single_request_keeps_autoregressive_feedback_dependency(self):
        r=simulate([stage('a',2),stage('b',3)],1,feedback_cycles=1,warmup=10,samples=100)
        self.assertEqual(r['first_completion_cycles'],5)
        self.assertAlmostEqual(r['generated_completions_per_cycle'],1/6)
        self.assertAlmostEqual(r['mean_request_itl_cycles'],6)
        self.assertIsNone(r['generated_tokens_per_second'])

    def test_many_requests_fill_pipeline_but_slow_stage_limits_rate(self):
        r=simulate([stage('a',2),stage('b',7),stage('c',1)],12,warmup=100,samples=400)
        self.assertAlmostEqual(r['generated_completions_per_cycle'],1/7)
        self.assertGreater(r['stages'][0]['output_blocked_item_cycles'],0)
        self.assertTrue(all(s['peak_slots']<=2 for s in r['stages']))

    def test_two_requests_do_not_fill_64_stage_pipeline(self):
        r=simulate([stage(str(i),10) for i in range(64)],2,warmup=100,samples=1000)
        self.assertAlmostEqual(r['generated_completions_per_cycle'],2/640)
        self.assertEqual(r['first_completion_cycles'],640)

    def test_latency_and_initiation_interval_differ_and_slots_limit_overlap(self):
        r=simulate([stage('a',10,1,2)],20,warmup=100,samples=1000)
        self.assertAlmostEqual(r['generated_completions_per_cycle'],2/10)
        wider=simulate([stage('a',10,1,12)],20,warmup=100,samples=1000)
        self.assertAlmostEqual(wider['generated_completions_per_cycle'],1)

    def test_clock_is_explicit_and_invalid_costs_fail(self):
        r=simulate([stage('a',4)],1,clock_hz=1000,warmup=10,samples=20)
        self.assertEqual(r['generated_tokens_per_second'],250)
        for value in (0,-1,float('nan'),None):
            with self.assertRaises(ValueError):simulate([stage('a',value)],1)


class PlanPredictionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base=ROOT/'performance/evidence'
        cls.plan,cls.digest=read_json(base/'pipeline-stage-map-002/stage-map.json')
        cls.ir,_=read_json(base/'pipeline-stage-map-002/layer-kernel-ir.json')
        cls.calibration,_=read_json(base/'native-shapes-summary-001.json')

    def test_real_plan_never_turns_partial_calibration_into_full_tps(self):
        r=analyze(self.plan,self.ir,self.calibration)
        self.assertEqual(r['concurrency'],2)
        self.assertAlmostEqual(r['requirements']['maximum_mean_dependent_decode_cycle_seconds'],.0005)
        self.assertEqual(r['target_single_conversation_response_tokens_per_second'],2000)
        self.assertFalse(r['independent_request_capacity']['qualifies_single_conversation_target'])
        self.assertIsNone(r['predicted_generated_tokens_per_second'])
        self.assertFalse(r['complete_model_prediction'])
        self.assertGreater(r['coverage']['calibrated_shape_mac_fraction'],.9)
        self.assertIn('BF16 tile 1x128',r['coverage']['uncalibrated_matrix_shapes'])
        self.assertNotIn('embedding/embedding',[r['id'] for r in r['top_native_work_regions']])

    def test_boundary_load_uses_lane_schedule_and_counts_shared_links(self):
        edge=dict(boundary_ports=[dict(source=[0,0],destination=[1,0]),dict(source=[0,1],destination=[1,1])],
                  chunks=3,packet_words=10,chunk_to_lane='chunk_index modulo lane_count')
        r=boundary_traffic(dict(stage_boundaries=[edge,edge]))
        self.assertEqual(r['words_per_complete_pass'],60)
        self.assertEqual(r['maximum_words_per_boundary_link'],40)
        self.assertFalse(r['complete_network'])

    def test_mismatched_placement_and_unadmitted_capacity_fail(self):
        p=copy.deepcopy(self.plan);p['placement_admitted']=False
        with self.assertRaises(ValueError):analyze(p,self.ir,self.calibration)
        ir=copy.deepcopy(self.ir);ir['kernels'][1]['rect'][0]+=1
        with self.assertRaises(ValueError):analyze(self.plan,ir,self.calibration)

    def scenario(self):
        s=service_template(self.plan,self.digest)
        s.update(feedback_cycles=5,feedback_source='synthetic unit test')
        for st in s['stages']:st.update(latency_cycles=10,ii_cycles=2,source='synthetic unit test')
        return s

    def test_stale_profiles_and_unplaced_concurrency_are_rejected(self):
        s=self.scenario();s['plan_sha256']='stale'
        with self.assertRaises(ValueError):evaluate_scenario(self.plan,self.digest,s)

    def test_unknown_template_cannot_silently_assume_zero_latency(self):
        s=service_template(self.plan,self.digest)
        with self.assertRaises(ValueError):evaluate_scenario(self.plan,self.digest,s)
        with self.assertRaises(ValueError):evaluate_scenario(self.plan,self.digest,self.scenario(),concurrency=3)
        s=self.scenario();s['clock_hz']=850e6
        with self.assertRaises(ValueError):evaluate_scenario(self.plan,self.digest,s)

    def test_complete_explicit_profile_runs_without_claiming_acceptance(self):
        r=evaluate_scenario(self.plan,self.digest,self.scenario(),warmup=10,samples=100)
        self.assertIsNone(r['generated_tokens_per_second'])
        self.assertFalse(r['target_achieved'])
        self.assertEqual(r['scenario_concurrency'],1)
        self.assertFalse(r['response_inclusive'])
        self.assertAlmostEqual(r['generated_completions_per_cycle'],1/(66*10+5))
        parallel=evaluate_scenario(self.plan,self.digest,self.scenario(),concurrency=2,warmup=10,samples=100)
        self.assertAlmostEqual(parallel['generated_completions_per_cycle'],2/(66*10+5))
        self.assertIn('Independent-request aggregate',parallel['metric_scope'])


if __name__=='__main__':unittest.main()
