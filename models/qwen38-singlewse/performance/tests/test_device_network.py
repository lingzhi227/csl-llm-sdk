"""Validate full-region routing and reject broken credit/address ownership."""
import sys,unittest
from copy import deepcopy
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from spatial.device_network import lower_device_network,audit_device_network


class DeviceNetworkTests(unittest.TestCase):
 def test_full_original_region_has3554_distinct_targets_and_one_gateway(self):
  p=lower_device_network([63,67,45,79]);self.assertEqual(p['gateway'],[107,145]);self.assertEqual(p['audit']['endpoints'],3554)
  self.assertEqual({w['tag'] for w in p['workers']},set(range(3554)))
  self.assertEqual(p['audit']['maximum_response_hops'],3554)

 def test_all_cardinal_turns_and_large_sparse_tags(self):
  p=lower_device_network([0,0,3,3],[0,1,48,72,3419,3458,3479,3504])
  self.assertEqual(p['audit']['endpoints'],8)
  self.assertEqual({w['reply_tx'] for w in p['workers']},{'EAST','WEST','SOUTH'})

 def test_wrong_filter_or_broken_response_link_is_rejected(self):
  original=lower_device_network([0,0,3,3])
  p=deepcopy(original);r=next(r for r in p['routes'] if 'filter_tag' in r);r['filter_tag']=99
  with self.assertRaises(ValueError):audit_device_network(p)
  p=deepcopy(original);r=next(r for r in p['routes'] if r['color']==13 and r['pe']==[2,0]);r['tx']=['NORTH']
  with self.assertRaises(ValueError):audit_device_network(p)

if __name__=='__main__':unittest.main()
