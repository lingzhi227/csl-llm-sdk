"""Host observations for device-only Q/K handoff; no candidate operand H2D."""
import hashlib
import json
import numpy as np
from qwen38.wp14_layout import ARRAYS, QK_INITIAL, QK_READS, PROBES
from qwen38.projection_handoff import descriptor, data_frame, acknowledgement, WIRE_GUARDS
from qwen38.qk_rope_protocol import expected_state, EVENTS
from qwen38.qk_rope_numerics import validate, expanded, bits


class ConsumerChecks:
    def __init__(self, root, transfer, launch, observed, norm_weights):
        self.root, self.transfer, self.launch, self.observed = root, transfer, launch, observed
        self.reference = np.load(root/'qk-reference-observations.npz', allow_pickle=False)
        self.weights = norm_weights
        self.frequencies = self.reference['inverse_frequencies'].tolist()
        assert np.array_equal(np.array(norm_weights), self.reference['norm_weights'])
        self.initial = {}
        self.input = np.array([0xa55a]+[0x3f00]*512+[0x5aa5], dtype=np.uint32)

    def read(self, name, pe=4):
        count, native = ARRAYS[name]
        integer = native == 16 or name in ('qk_control', 'qk_state', 'qk_events', 'handoff_control', 'handoff_state', 'wire_data', 'wire_ack')
        return self.transfer('d2h', name, pe, count, native, dtype=np.uint32 if integer else np.float32)

    def write(self, name, values, pe=4):
        count, native = ARRAYS[name]
        integer = native == 16 or name in ('qk_control', 'handoff_control')
        return self.transfer('h2d', name, pe, count, native, values=np.array(values,dtype=np.uint32 if integer else np.float32))

    def save(self):
        np.savez(self.root/'device-observations.npz', **self.observed)

    def initialize(self):
        values = {'qk_input': self.input.tolist(),
                  'qk_weights': [0xa55a, *[bits(v)>>16 for v in self.weights], 0x5aa5],
                  'qk_frequencies': [1234.5, *self.frequencies, -4321.25],
                  'qk_probe_inputs': [1234.5, *PROBES, -4321.25], 'qk_control': [1,1,1]}
        for name in QK_INITIAL:
            count, native = ARRAYS[name]
            defaults = [0xa55a]+[0]*(count-2)+[0x5aa5] if native==16 else [1234.5]+[0.]*(count-2)+[-4321.25]
            array = self.write(name, values.get(name, defaults))
            self.initial[name] = array.copy()
        self.launch('initialize_consumer')
        state = self.read('qk_state'); hand = self.read('handoff_state')
        self.observed.update(qk_initial_state=state, qk_initial_handoff_state=hand); self.save()
        assert state.tolist() == expected_state(1,0,0,1,True)
        assert hand.tolist() == [1,1]+[0]*22

    def handoff_and_compute(self):
        for rank in range(4):
            control = descriptor(rank)
            for pe in range(5): self.write('handoff_control', control, pe)
            self.launch('arm_handoff')
            armed = self.read('handoff_state')
            self.observed[f'handoff_{rank}_armed'] = armed; self.save()
            assert armed[2]==1 and armed[3]==(1<<rank)-1 and armed[4]==rank and armed[8]==0
            assert armed[9:12].tolist()==[0,0,0]
            self.launch('handoff')
            for pe in (rank,4):
                back = {name:self.read(name,pe) for name in ('handoff_state','wire_data','wire_ack')}
                for name,a in back.items(): self.observed[f'handoff_{rank}_pe{pe}_{name}']=a
                self.save()
                for name in ('wire_data','wire_ack'):
                    assert (int(back[name][0]),int(back[name][-1]))==WIRE_GUARDS
                words = self.observed[f'1_{rank}_bf16'][1:-1].tolist()
                assert back['wire_data'][1:-1].tolist()==data_frame(control,words)
                assert back['wire_ack'][1:-1].tolist()==acknowledgement(control)
                st=back['handoff_state']
                assert st[0:3].tolist()==[1,1,3] and st[4]==rank and st[8]==0
                assert st[9:13].tolist()==[1,1,1,0] and st[20:23].tolist()==[control[3],control[7],128]
                if pe==rank:
                    assert st[3]==1<<rank and st[5:8].tolist()==[1,1,138] and st[23]==1
                    assert 0<st[16]<st[17]
                else:
                    assert st[3]==(1<<(rank+1))-1 and st[5:8].tolist()==[rank+1,rank+1,(rank+1)*138]
                    assert st[18:20].tolist()==[min(rank+1,2)*128,max(rank-1,0)*128] and st[23]==rank+1
                    assert 0<st[14]<st[15]<st[16]<st[17]
            self.input[1+rank*128:1+(rank+1)*128] = self.observed[f'1_{rank}_bf16'][1:-1]
            actual = self.read('qk_input')
            self.observed[f'handoff_{rank}_input']=actual; self.save()
            assert np.array_equal(actual,self.input)
        self.launch('preprocess')
        back = {name:self.read(name) for name in (*QK_READS,'handoff_state')}
        self.observed.update({'consumer_'+name:array for name,array in back.items()}); self.save()
        assert np.array_equal(back['qk_input'],self.input)
        for name in ('qk_weights','qk_frequencies','qk_probe_inputs','qk_control'):
            assert np.array_equal(back[name].view(np.uint32),self.initial[name].view(np.uint32))
        for name in QK_READS:
            if name in ('qk_stats','qk_state','qk_events','qk_control'): continue
            guards=(0xa55a,0x5aa5) if ARRAYS[name][1]==16 else (1234.5,-4321.25)
            assert (back[name][0],back[name][-1])==guards
        assert back['qk_state'].tolist()==expected_state(1,1,1,1)
        assert back['qk_events'].tolist()==EVENTS
        st=back['handoff_state']
        assert st[0:4].tolist()==[1,1,4,15] and st[5:9].tolist()==[4,4,552,0]
        assert st[12]==1 and st[18:20].tolist()==[256,256] and st[23]==4
        actual = {name: (back['qk_'+name] if name=='stats' else back['qk_'+name][1:-1]).tolist()
                  for name in ('stats','rms_stages','normalized','trig','trig_casts','rotary_stages','product_casts','output','probes')}
        token={'position':1,'q':[expanded(int(w)) for w in self.input[1:257]],'k':[expanded(int(w)) for w in self.input[257:513]]}
        conditional=validate(token,self.weights[:256],self.weights[256:],self.frequencies,actual,PROBES)
        source_checks={}
        case='bounded_dyadic'
        for head,name in enumerate(('q','k')):
            decoded=np.array([expanded(int(w)) for w in actual['output'][head*256:(head+1)*256]])
            normalized=np.array([expanded(int(w)) for w in actual['normalized'][head*256:(head+1)*256]])
            for label,values in [('output',decoded),('normalized',normalized)]:
                interval=self.reference[case+'__'+name+'_'+label+'_bounds']
                source_checks[name+'_'+label]=bool(np.all((interval[:,0]<=values)&(values<=interval[:,1])))
            source_checks[name+'_official_mismatches']=int(np.count_nonzero(decoded!=self.reference[case+'__'+name+'_output']))
        checks=dict(conditional=conditional,source=source_checks,transport_exact=True,protocol_exact=True,
                    actual_operand_gate_separate_from_original_hidden=True)
        (self.root/'consumer-checks.json').write_text(json.dumps(checks,indent=2)+'\n')
        assert conditional['passed'] and all(v for k,v in source_checks.items() if not k.endswith('_mismatches'))
        return checks
