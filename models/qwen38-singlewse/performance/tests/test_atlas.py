from dataclasses import replace
from pathlib import Path
import sys,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.atlas import AtlasPolicy,Geometry,complement_segments,eligible_to_bank,bank_to_eligible,slot_count,intervals_for,value_arena

class AtlasTests(unittest.TestCase):
 def test_actor_holes_and_serpentine_inversion(self):
  p=AtlasPolicy();g=Geometry(p);actors=g.actors();self.assertEqual(len(actors),9120);self.assertEqual(len(set(actors)),9120)
  self.assertTrue(all(g.bank_id(*xy) is None for xy in actors))
  self.assertEqual(g.bank_xy(0),(0,0));self.assertEqual(g.bank_xy(749),(749,0));self.assertEqual(g.bank_xy(750),(749,1));self.assertEqual(g.bank_xy(p.bank_pes-1),(120,1159))
  for r in range(len(g.rows)):
   for c in (0,374,749):
    b=r*750+c
    if b<p.bank_pes:self.assertEqual(g.bank_id(*g.bank_xy(b)),b)
  for bad in [replace(p,bank_pes=860879),replace(p,actor_row_stride=8),replace(p,actor_first_row=100),replace(p,context=2048)]:
   with self.assertRaises(ValueError):bad.validate()
 def test_compact_ring_excludes_state_and_preserves_order(self):
  for excluded in [[],[0],[23],[2,3,4,9,10,23]]:
   seg,n=complement_segments(24,excluded);expected=[i for i in range(24) if i not in excluded]
   self.assertEqual(n,len(expected))
   for i,b in enumerate(expected):self.assertEqual(eligible_to_bank(i,seg),b);self.assertEqual(bank_to_eligible(b,seg),i)
   for b in excluded:self.assertIsNone(bank_to_eligible(b,seg))
 def test_whole_k_dispatch_segments_equal_independent_round_robin(self):
  for n,k,start,length,phase in [(120,40,80,240,60),(4080,136,3944,680,0),(120,40,400,200,60)]:
   m=dict(kind='bf16',k_blocks=k,stream_start=start,tiles=length);classes={'bf16':dict(pes=n,phase=phase)};decoded=[]
   for d in intervals_for(m,classes):
    self.assertEqual(d['class_start']%k,0);self.assertEqual(d['count']%k,0)
    for owner in range(n):
     rank=(owner-phase)%n
     if d['class_start']<=rank<d['class_start']+d['count']:decoded.append((d['tile_start']+rank-d['class_start'],owner,d['slot']))
   self.assertEqual(sorted(decoded),[(t,(phase+start+t)%n,(start+t)//n) for t in range(length)])
   counts=[sum(1 for a in range(start+length) if (phase+a)%n==owner) for owner in range(n)]
   self.assertEqual(counts,[slot_count(start+length,i,n,phase) for i in range(n)])
 def test_closed_lifetimes_cannot_reuse_an_input_at_its_consumer(self):
  graph=dict(nodes=[dict(id=0,inputs=['token'],outputs=['a']),dict(id=1,inputs=['a'],outputs=['b']),dict(id=2,inputs=['b'],outputs=['c']),dict(id=3,inputs=['c'],outputs=['d'])],values={n:dict(shape=[129],dtype='BF16') for n in 'abcd'},selected_token='d')
  r=value_arena(graph,AtlasPolicy());v=r['values']
  self.assertNotEqual(v['a']['base_cell'],v['b']['base_cell']);self.assertEqual(v['a']['base_cell'],v['c']['base_cell'])
  for left in v.values():
   for right in v.values():
    if left is right:continue
    live=max(left['birth'],right['birth'])<=min(left['last_use'],right['last_use'])
    overlap=max(left['base_cell'],right['base_cell'])<min(left['base_cell']+left['reserved_cells'],right['base_cell']+right['reserved_cells'])
    self.assertFalse(live and overlap)
