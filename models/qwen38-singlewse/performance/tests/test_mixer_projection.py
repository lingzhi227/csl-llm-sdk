"""Complete original mixer projection ownership; not numerical execution."""
import copy
import json
from pathlib import Path
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
from spatial.mixer_projection import lower_mixer_projections,audit_mixer_projections
from stage_layer_mlp_compile import build


class MixerProjectionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        plan=json.loads((ROOT/'evidence/layer-native-schedule-002/layer-schedule.json').read_text())
        cls.stage=next(s for s in plan['stages'] if s['id']=='layer_00')
        cls.region=next(r for r in cls.stage['regions'] if r['role']=='mix')
        cls.plan=lower_mixer_projections(cls.region,cls.stage['request_controller']['pe'])

    def test_all_original_dimensions_and_every_tile_address(self):
        a=self.plan['audit'];self.assertEqual(a['original_tiles_checked'],454400)
        self.assertEqual([x['shape'] for x in a['original_matrices']],[[10240,5120],[6144,5120],[48,5120],[48,5120],[5120,6144]])
        self.assertEqual([x['rows'] for x in a['original_matrices']],[10240,6144,48,48,5120])
        self.assertEqual(a['fixed_neighbor_colors'],[10,11]);self.assertFalse(a['full_mixer'])
        self.assertEqual(a['input_owner_deliveries'],7072)

    def test_corrupt_original_address_type_extent_and_edge_rejected(self):
        for change in ('address','dtype','extent','edge','port'):
            with self.subTest(change=change):
                p=copy.deepcopy(self.plan)
                if change=='address':p['workers'][0]['descriptors'][0]['base_words']+=1
                elif change=='dtype':p['workers'][0]['descriptors'][0]['bf16']=1
                elif change=='extent':p['workers'][0]['descriptors'][0]['iterations']-=1
                elif change=='edge':p['reduction_routes'][0]['tx']='NORTH'
                else:p['root_ports'][0]['first_row']+=2
                with self.assertRaises(ValueError):audit_mixer_projections(self.region,p)

    def test_incorrect_input_filter_hop_duplicate_and_sdk_task_rejected(self):
        for change in ('filter','hop','duplicate','reserved'):
            with self.subTest(change=change):
                p=copy.deepcopy(self.plan)
                if change=='filter':
                    next(r for r in p['input_routes'] if 'filter_tag' in r)['filter_tag']+=1
                elif change=='hop':p['input_routes'][0]['rx']=['NORTH']
                elif change=='duplicate':p['input_routes'].append(copy.deepcopy(p['input_routes'][0]))
                else:p['resources']['local_tasks'][0]=21
                with self.assertRaises(ValueError):audit_mixer_projections(self.region,p)

    def test_actual_norm_mlp_cohost_keeps_every_original_bank_extent(self):
        old,_,_,before=build('layer_00',shared_inputs=True,norm_bridge=True)
        new,w,h,after=build('layer_00',shared_inputs=True,norm_bridge=True,mixer_projections=True)
        self.assertEqual((w,h),(78,146))
        self.assertEqual({tuple(p['pe']):p['parameters'].get('bank_words',0) for p in before},
                         {tuple(p['pe']):p['parameters'].get('bank_words',0) for p in after})
        self.assertEqual(old['state-page-remap.json'],new['state-page-remap.json'])
        for name in ('layer_projection.csl','layer_norm_bridge.csl','layer_norm_sender.csl','mlp_schedule.csl'):
            self.assertEqual(old[name],new[name])
        self.assertEqual(sum(p['source'].startswith('mixer_') for p in after),3554)
        self.assertTrue(all(p['parameters']['mixer_input48_queue']==4 for p in after if p['source']=='mixer_layer_norm_sender.csl'))
        self.assertTrue(all(p['parameters']['mixer_input48_queue']==5 for p in after if p['source']=='mixer_layer_norm_bridge.csl'))
        for name in ('mixer_layer_norm_bridge.csl','mixer_layer_norm_sender.csl'):
            self.assertIn(b'@assert(!active);',new[name])
            self.assertIn(b'@assert(mixer.drained() and mixer_status[2]==0);',new[name])
        self.assertNotIn(b'MIXER_COHOST_IDLE',new['mixer_layer_mlp_standby.csl'])


if __name__=='__main__':unittest.main()
