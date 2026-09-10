import unittest
from qwen38.projected_attention_protocol import FRAGMENTS,CommitModel,frame,ack,verify_ack,parse_frame,routes,QK_MASK,ATTENTION_MASK
from qwen38.wp15_layout import sequence,budget,declared_bytes
from qwen38.projected_attention_numerics import Interval,attention
from qwen38.projected_attention_checks import ignored_qk_witness

class ProjectedAttentionProtocolTests(unittest.TestCase):
    def test_fragment_aliases_stale_and_partial_writes(self):
        model=CommitModel(9);identity=(1,1,0);model.begin(*identity)
        f=FRAGMENTS[8];good=frame(f,identity,[0x3f80]*128)
        # All metadata, both guards, version and canonical payload are validated
        # before a single destination word is committed.
        for index in range(13):
            bad=good.copy();bad[index]^=1
            before=model.payload.copy()
            with self.assertRaises(ValueError):model.commit(f,bad)
            self.assertEqual(before,model.payload);self.assertEqual(model.mask,0)
        for index in (13,140):
            bad=good.copy();bad[index]|=1<<16
            with self.assertRaises(ValueError):model.commit(f,bad)
        model.commit(f,good)
        with self.assertRaises(ValueError):model.commit(f,good)
        with self.assertRaises(ValueError):model.commit(FRAGMENTS[9],good)
        with self.assertRaises(ValueError):model.release()
        verify_ack(ack(f,identity),f,identity)
        with self.assertRaises(ValueError):verify_ack(ack(f,(2,1,1)),f,identity)

    def test_complete_tokens_keep_request_and_advance_contraction(self):
        for pe,mask in ((8,QK_MASK),(9,ATTENTION_MASK)):
            model=CommitModel(pe)
            for cid in (1,2):
                identity=(cid,1,cid-1);model.begin(*identity)
                for f in FRAGMENTS:
                    if f.destination==pe:model.commit(f,frame(f,identity,[cid*100+f.identity]*128))
                self.assertEqual(model.mask,mask)
                with self.assertRaises(ValueError):model.begin(cid+1,1,cid)
                model.release()
                with self.assertRaises(ValueError):model.begin(cid,1,cid)
            with self.assertRaises(ValueError):model.begin(3,1,0)
            model.begin(3,2,0)
            self.assertEqual(model.identity,(3,2,0))

    def test_routing_queue_and_operand_coverage(self):
        links=routes();colors=[r[k] for r in links for k in ('data_color','ack_color')]
        self.assertEqual(sorted(colors),list(range(2,20)))
        for pe,size in ((8,512),(9,1024)):
            written=[]
            for f in FRAGMENTS:
                if f.destination==pe:written.extend(range(f.destination_offset,f.destination_offset+128))
            self.assertEqual(sorted(written),list(range(size)))
            self.assertEqual(len(written),len(set(written)))
            queues=[r['receiver_queue'] for r in links if r['destination']==pe]
            self.assertEqual(len(queues),len(set(queues)))
            self.assertTrue(all(2<=q<=6 for q in queues))
        seq=sequence()
        self.assertEqual(sum(r['kind']=='launch' and r['name']=='initialize_request' for r in seq),1)
        for name in ('qk_input','attention_input','attention_cache'):
            self.assertEqual(sum(r['kind']=='copy' and r['name']==name and r['direction']=='h2d' for r in seq),1)
        self.assertEqual(budget()['fabric']['injected_bytes'],14688)
        self.assertEqual(budget()['fabric']['route_byte_hops'],53856)
        self.assertLessEqual(budget()['max_host_buffer_bytes'],65536)
        for role in ('producer','qk','attention'):self.assertLess(declared_bytes(role)['total']+4096,49152)

    def test_uncertain_cached_V_and_score_domain(self):
        z=[Interval.point(0)]*256;zn=[0.]*256
        v0=[Interval(.21875,.28125)]*256;v1=[Interval.point(.5)]*256
        result=attention(z,[z,z],[v0,v1],z,zn,[zn,zn],[[.25]*256,[.5]*256],zn)
        bound=result['output'][0]
        self.assertLessEqual(bound.lo,.1796875);self.assertGreaterEqual(bound.hi,.1953125)
        self.assertLess(bound.hi,.25) # Current-KV-only loses the retained first value.
        self.assertEqual(result['source_shift_magnitude_upper'],0)
        large=[Interval.point(16)]*256
        with self.assertRaises(ValueError):attention(large,[large,[-x for x in large]],[v0,v1],z,[16.]*256,[[16.]*256,[-16.]*256],[[.25]*256,[.5]*256],zn)
        self.assertTrue(ignored_qk_witness([1.]*256,[[1.]*256,[-1.]*256])['rejected'])
        self.assertFalse(ignored_qk_witness(zn,[zn])['rejected'])

class ProjectedAttentionAdmissionTests(unittest.TestCase):
    def test_long_deadline_requires_exact_single_candidate_admission(self):
        import copy
        from qwen38.wp15_admission import validate
        profile=dict(scope='wp15-original-hidden-persistent-attention',pes=10,case_indices=[0,1],positions=[0,1],request_generations=[1,1])
        steps=dict(required_profile='sdk',steps=[dict(name='compile',seconds=300),dict(name='simulate',seconds=1200)])
        admission=dict(package='WP15',sdk_execution_authorized=True,candidate='wp15-001-ready',candidate_limit=1,
                       manifest_sha256='frozen',compile_seconds=300,simulation_seconds=1200)
        self.assertTrue(validate(admission,profile,steps,'frozen','wp15-001-ready'))
        for key,v in [('package','WP14'),('sdk_execution_authorized',False),('candidate_limit',2),
                      ('manifest_sha256','changed'),('compile_seconds',301),('simulation_seconds',1201)]:
            bad={**admission,key:v}
            with self.assertRaises(ValueError):validate(bad,profile,steps,'frozen','wp15-001-ready')
        for key,v in [('pes',5),('request_generations',[1,2]),('case_indices',[0]),('scope','arbitrary')]:
            with self.assertRaises(ValueError):validate(admission,{**profile,key:v},steps,'frozen','wp15-001-ready')
        bad=copy.deepcopy(steps);bad['steps'][0]['seconds']=1200
        with self.assertRaises(ValueError):validate(admission,profile,bad,'frozen','wp15-001-ready')
