import unittest
from spatial.gdn_domain_routes import Fabric,emit_domain


class DomainRouteTests(unittest.TestCase):
    def fabric(self):
        profiles={(x,y):dict(source='compact_layer_projection.csl',parameters={})for x in range(4)for y in range(3)}
        return Fabric(profiles,{pe:46000 for pe in profiles},[])

    def test_chain_covers_anchors_once_without_revisiting_cells(self):
        f=self.fabric();anchors=[(3,0),(3,2),(0,2)]
        for nearest in (False,True):
            path=f.chain((0,0),anchors,f.cells,nearest=nearest)
            self.assertEqual(len(path),len(set(path)));self.assertTrue(set(anchors)<=set(path))
            self.assertTrue(all(b in f.neighbors[a]for a,b in zip(path,path[1:])))

    def test_disconnected_color_does_not_create_a_route(self):
        f=self.fabric();legal=f.cells-{(1,y)for y in range(3)}
        self.assertIsNone(f.chain((0,0),[(3,0)],legal))
        self.assertIsNone(f.tree((0,0),{(3,0)},legal))

    def test_intermediate_weighted_bridge_advances_but_terminal_does_not(self):
        f=self.fabric();targets={(1,0),(2,0)};tree=f.tree((0,0),targets,f.cells)
        rows=emit_domain((0,0),targets,1,9,tree,[(0,0),(1,0),(2,0)],{},weighted=True)
        returned={tuple(r['pe']):r for r in rows if r['color']==9}
        self.assertEqual(returned[(1,0)]['switch'],dict(next_rx='EAST',pop_on_advance=True))
        self.assertNotIn('switch',returned[(2,0)])


if __name__=='__main__':unittest.main()
