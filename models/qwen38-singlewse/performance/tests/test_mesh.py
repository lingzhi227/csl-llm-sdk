import importlib.util
from pathlib import Path
import sys
import unittest
from dataclasses import replace
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.mesh import MeshPlan

class MeshTests(unittest.TestCase):
    def test_topology_and_degenerate_rectangles(self):
        for w,h in [(1,2),(2,1),(2,2),(3,5),(16,16),(750,2)]:
            plan=MeshPlan(width=w,height=h)
            report=plan.validate()
            self.assertEqual(report['tree_edges'],w*h-1)
            for y in range(h):
                for x in range(w):
                    node=(x,y);seen=set()
                    while node is not None:
                        self.assertNotIn(node,seen);seen.add(node);node=plan.parent(*node)
                    self.assertIn((0,0),seen)
    def test_resource_conflicts_rejected(self):
        base=MeshPlan()
        for plan in [replace(base,data_color=4),replace(base,row_colors=(4,4)),
                     replace(base,input_queues=(2,2,4)),replace(base,output_queues=(0,3)),
                     replace(base,local_tasks=(8,9,9)),replace(base,data_color=21),
                     replace(base,width=751),replace(base,max_words=8192,code_allowance=20000)]:
            with self.assertRaises(ValueError):plan.validate()
    def test_identity_and_manual_knobs(self):
        p=MeshPlan(width=3,height=4)
        self.assertEqual(p.fingerprint(),MeshPlan(width=3,height=4).fingerprint())
        self.assertNotEqual(p.fingerprint(),replace(p,width=4).fingerprint())
        text=p.emit_layout()
        self.assertEqual(text.count('@set_tile_code'),12)
        self.assertEqual(text.count('@set_color_config'),len(p.routes()))
    def test_independent_ack_simulation(self):
        p=MeshPlan(width=7,height=3)
        for epoch in (0,1,255):
            sums={}
            for y in reversed(range(p.height)):
                for x in reversed(range(p.width)):
                    sums[x,y]=y*p.width+x+1+epoch+sum(sums[c] for c in p.children(x,y))
            n=p.width*p.height
            self.assertEqual(sums[0,0],n*(n+1)//2+n*epoch)

if __name__=='__main__':unittest.main()
