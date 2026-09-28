import copy
import json
from pathlib import Path
import random
import struct
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from spatial.layer_schedule import lower_layers,audit_native_schedule,tile_owner,xy_rank,rank_xy,gdn_state_owner
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from layer_weights import matrix_tiles


class LayerScheduleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        root=Path(__file__).resolve().parents[2]
        cls.tensors=json.loads((root/'configs/tensors.json').read_text())['tensors']
        original=json.loads((root/'performance/evidence/pipeline-stage-map-002/stage-map.json').read_text())
        calibration=json.loads((root/'performance/evidence/native-shapes-hw-002/result.json').read_text())
        cls.plan=lower_layers(original,calibration)

    def test_all_original_tiles_and_layers_still_fit(self):
        a=audit_native_schedule(self.plan,self.tensors)
        self.assertTrue(a['passed']);self.assertEqual(a['matrices'],498)
        self.assertEqual(a['native_tiles'],115077120);self.assertFalse(a['compiled_sram'])
        self.assertEqual(self.plan['failed_regions'],[])
        stage=self.plan['stages'][1]
        self.assertEqual(stage['regions'][1]['native_shape'],[8,32])
        self.assertEqual(stage['regions'][2]['native_shape'],[4,64])

    def test_ragged_tail_keeps_every_k_slice_and_paired_roots(self):
        r=self.plan['stages'][1]['regions'][1];gate,up=r['matrices']
        self.assertEqual(gate['group_partitions'],up['group_partitions'])
        self.assertLess(gate['group_partitions'][-1]['workers'],gate['k_blocks'])
        randomizer=random.Random(314)
        for outrow in [0,gate['output_tiles']-1,*[randomizer.randrange(gate['output_tiles']) for _ in range(10)]]:
            seen=set();pe_slices={}
            for k in range(gate['k_blocks']):
                tile=outrow*gate['k_blocks']+k;a=tile_owner(r,0,tile);b=tile_owner(r,1,tile)
                self.assertEqual(a['pe'],b['pe']);self.assertNotEqual(a['byte_offset'],b['byte_offset'])
                seen.add(k);pe_slices.setdefault(tuple(a['pe']),[]).append(k)
                for o in [a,b]:self.assertLessEqual(o['byte_offset']+o['bytes'],35256)
            self.assertEqual(seen,set(range(gate['k_blocks'])))
            self.assertLessEqual(max(map(len,pe_slices.values()))*32,128)

    def test_serpentine_mapping_and_state_request_isolation(self):
        r=self.plan['stages'][1]['regions'][0];n=r['rect'][2]*r['rect'][3]
        coords=[rank_xy(r,i) for i in range(n)]
        self.assertEqual(len(set(map(tuple,coords))),n)
        for i,xy in enumerate(coords):self.assertEqual(xy_rank(r,xy),i)
        for a,b in zip(coords,coords[1:]):self.assertEqual(sum(abs(x-y) for x,y in zip(a,b)),1)
        seen=set()
        for request in range(2):
            for head in range(48):
                for key in range(32):
                    for value in range(16):
                        p=gdn_state_owner(r,request,head,key,value);address=(*p['pe'],p['byte_offset'])
                        self.assertNotIn(address,seen);seen.add(address)
                        self.assertEqual(p['shape'],[4,8]);self.assertLessEqual(p['byte_offset']+128,35256)
        self.assertEqual(len(seen),49152)

    def test_corrupt_ragged_coverage_rejected(self):
        bad=copy.deepcopy(self.plan)
        bad['stages'][1]['regions'][1]['matrices'][0]['group_partitions'][-1]['output_start']+=1
        with self.assertRaises(ValueError):audit_native_schedule(bad,self.tensors)


class Matrix:
    def __init__(self,first,count,width,kind):self.first=first;self.count=count;self.width=width;self.kind=kind
    def __getitem__(self,ij):
        r,c=ij
        if not 0<=r<self.count or not 0<=c<self.width:raise IndexError()
        if self.kind=='scale':return 0x3f80+(self.first+r)*2+c
        if self.kind=='fp8':return ((self.first+r)*11+c*7)%126
        return ((self.first+r)*257+c)&65535


class Weights:
    def __init__(self,fp8=True):self.fp8=fp8;self.reads=[]
    def rows(self,name,first,count):
        self.reads.append((name,first,count))
        return Matrix(first,count,2 if name.endswith('scale_inv') else 256,'scale' if name.endswith('scale_inv') else 'fp8' if self.fp8 else 'bf16')


class LayerWeightPackingTests(unittest.TestCase):
    def test_original_fp8_bytes_and_scale_across_every_block_boundary(self):
        for rows,columns in [(2,128),(4,64),(8,32),(16,16)]:
            reader=Weights();m=dict(tensor='w',scale_tensor='w_scale_inv',shape=[256,256],tile_shape=[rows,columns],dtype='F8_E4M3')
            observed={}
            for tile,raw in matrix_tiles(reader,m):
                outblock,kblock=divmod(tile,256//columns);r0=outblock*rows;c0=kblock*columns
                self.assertEqual(struct.unpack('<I',raw[256:])[0],(0x3f80+(r0//128)*2+c0//128)<<16)
                for index,byte in enumerate(raw[:256]):
                    c,r=divmod(index,rows);address=(r0+r,c0+c)
                    self.assertNotIn(address,observed);observed[address]=byte
            self.assertEqual(len(observed),256*256)
            for (r,c),byte in observed.items():self.assertEqual(byte,(r*11+c*7)%126)
            self.assertEqual(len([n for n in reader.reads if n[0]=='w']),256//rows)
            self.assertEqual(len([n for n in reader.reads if n[0].endswith('scale_inv')]),2)

    def test_original_bf16_row_tail_bits(self):
        m=dict(tensor='w',shape=[3,256],tile_shape=[1,128],dtype='BF16');reader=Weights(False)
        tiles=list(matrix_tiles(reader,m));self.assertEqual(len(tiles),6)
        for tile,raw in tiles:
            row,col=divmod(tile,2)
            self.assertEqual(struct.unpack('<128H',raw),tuple(row*257+col*128+k for k in range(128)))
