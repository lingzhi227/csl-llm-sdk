"""Bounded callback-order proof for distinct receive/packet/grant lifetimes.

This models local ownership, not device performance. Independent hardware
qualification is required because CSL queues and routes are not simulated here.
"""
import copy,random,unittest


class LeaseModel:
    def __init__(self,native,outputs,epoch):
        self.native=native;self.total=native+outputs;self.epoch=epoch;self.cursor=0
        self.grant=None;self.receiving=None;self.frame=None;self.packet=None
        self.events=[];self.retired=[];self.outputs=[];self.max_frames=0;self.overlap=False
        self.pump()

    def pump(self):
        if self.frame is not None:
            if self.cursor<self.native:
                if self.packet is not None:return
                self.packet=tuple(self.frame);self.frame=None
                self.events.append(('packet',self.packet))
            else:self.outputs.append(tuple(self.frame));self.frame=None
            self.cursor+=1
        if self.cursor<self.total and self.grant is None and self.receiving is None and self.frame is None:
            self.grant=(self.epoch,self.cursor);self.receiving=self.grant
            # A response may be observed before the local grant callback runs.
            self.events.extend([('grant',self.grant),('response',self.grant)])
            self.overlap |= self.packet is not None
        self.max_frames=max(self.max_frames,int(self.receiving is not None)+int(self.frame is not None))
        assert self.max_frames<=1

    def event(self,index):
        kind,identity=self.events.pop(index)
        if kind=='grant':
            assert self.grant==identity,'Grant arena changed before DMA callback'
            self.grant=None
        elif kind=='response':
            assert self.receiving==identity and self.frame is None,'Receive arena reused before consumption'
            self.receiving=None;self.frame=identity
        else:
            assert self.packet==identity,'Packet arena changed before DMA callback'
            self.packet=None;self.retired.append(identity)
        self.pump()

    def done(self):
        return self.cursor==self.total and not self.events and all(v is None for v in [self.grant,self.receiving,self.frame,self.packet])

    def identity(self):
        return (self.cursor,self.grant,self.receiving,self.frame,self.packet,tuple(sorted(self.events)),tuple(self.retired),tuple(self.outputs))

    def check(self):
        assert self.done(),'Finish/rearm before all local leases retire'
        assert self.retired==[(self.epoch,i) for i in range(self.native)]
        assert self.outputs==[(self.epoch,i) for i in range(self.native,self.total)]


class PrefetchTests(unittest.TestCase):
    def test_exhaustive_callback_order_including_final_output_before_packet_retirement(self):
        todo=[LeaseModel(3,2,0x89abcdef)];seen=set();ends=0;delayed_final=False
        while todo:
            state=todo.pop();key=state.identity()
            if key in seen:continue
            seen.add(key)
            delayed_final |= state.cursor==state.total and state.packet is not None
            if not state.events:state.check();ends+=1
            for i in range(len(state.events)):
                following=copy.deepcopy(state);following.event(i);todo.append(following)
        self.assertGreater(ends,0);self.assertTrue(delayed_final)

    def test_full_schedule_warm_epochs_with_adversarial_callback_delays(self):
        overlap=False
        for seed in range(50):
            rng=random.Random(seed)
            for epoch in range(1,5):
                state=LeaseModel(176,10,epoch)
                while state.events:state.event(rng.randrange(len(state.events)))
                state.check();overlap |= state.overlap
        self.assertTrue(overlap)

    def test_premature_buffer_reuse_and_rearm_are_detected(self):
        state=LeaseModel(2,1,1);state.event(1)
        self.assertIsNotNone(state.packet)
        with self.assertRaises(AssertionError):state.check()
        broken=copy.deepcopy(state);broken.packet=(99,99)
        with self.assertRaises(AssertionError):broken.event(next(i for i,e in enumerate(broken.events) if e[0]=='packet'))
        broken=copy.deepcopy(state);broken.grant=(99,99)
        with self.assertRaises(AssertionError):broken.event(next(i for i,e in enumerate(broken.events) if e[0]=='grant'))


if __name__=='__main__':unittest.main()
