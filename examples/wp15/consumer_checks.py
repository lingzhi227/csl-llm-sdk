"""WP15 exact device transport/cache and separate source/conditional observations."""
import json
import numpy as np
from qwen38.wp15_layout import ARRAYS,QK_INITIAL,ATTENTION_INITIAL,QK_READS,ATTENTION_READS,PROBES,integer
from qwen38.projected_attention_protocol import FRAGMENTS,frame,ack,QK_MASK,ATTENTION_MASK
from qwen38.qk_rope_numerics import validate as validate_qk,expanded,bits
from qwen38.qk_rope_protocol import expected_state as qk_state,EVENTS as QK_EVENTS
from qwen38.attention_composed_numerics import validate_attention
from qwen38.attention_protocol import NORMAL_EVENTS
from qwen38.projected_attention_checks import ignored_qk_witness


class ConsumerChecks:
    def __init__(self,root,transfer,launch,observed,norm_weights):
        self.root,self.transfer,self.launch,self.observed=root,transfer,launch,observed
        self.reference=np.load(root/'attention-reference-observations.npz',allow_pickle=False)
        self.intervals=np.load(root/'source-intervals.npz',allow_pickle=False)
        self.weights=norm_weights;self.frequencies=self.reference['inverse_frequencies'].tolist()
        assert np.array_equal(np.array(norm_weights),self.reference['norm_weights'])
        self.initial={};self.inputs={8:np.array([0xa55a]+[0x3f00]*512+[0x5aa5],dtype=np.uint32),
                                   9:np.array([0xa55a]+[0xbe80]*1024+[0x5aa5],dtype=np.uint32)}
        self.cache=np.array([0xa55a]+[0x3e80]*4096+[0x5aa5],dtype=np.uint32)
        self.rows=[];self.transport_checks=[]

    def read(self,name,pe):
        n,b=ARRAYS[name]
        return self.transfer('d2h',name,pe,n,b,dtype=np.uint32 if integer(name) else np.float32)

    def write(self,name,values,pe):
        n,b=ARRAYS[name]
        return self.transfer('h2d',name,pe,n,b,values=np.array(values,dtype=np.uint32 if integer(name) else np.float32))

    def save(self):
        np.savez(self.root/'device-observations.npz',**self.observed)
        (self.root/'consumer-checks.json').write_text(json.dumps(self.rows,indent=2)+'\n')
        (self.root/'transport-checks.json').write_text(json.dumps(self.transport_checks,indent=2)+'\n')

    def initialize(self):
        values={'qk_input':self.inputs[8].tolist(),'attention_input':self.inputs[9].tolist(),
                'attention_cache':self.cache.tolist(),
                'qk_weights':[0xa55a,*[bits(v)>>16 for v in self.weights],0x5aa5],
                'qk_frequencies':[1234.5,*self.frequencies,-4321.25],
                'qk_probe_inputs':[1234.5,*PROBES,-4321.25]}
        for pe,names in ((8,QK_INITIAL),(9,ATTENTION_INITIAL)):
            for name in names:
                n,b=ARRAYS[name]
                defaults=[0xa55a]+[0]*(n-2)+[0x5aa5] if b==16 else [1234.5]+[0.]*(n-2)+[-4321.25]
                self.initial[name]=self.write(name,values.get(name,defaults),pe).copy()
        for pe in range(10):self.write('request_control',[0,1,0,0],pe)
        self.launch('initialize_request')
        for pe in range(10):
            st=self.read('request_state',pe);self.observed[f'initial_request_{pe}']=st
            assert st.tolist()==[0,1,0,0,0,0,1,0]
        st=self.read('attention_state',9);cache=self.read('attention_cache',9)
        self.observed.update(initial_attention_state=st,initial_attention_cache=cache);self.save()
        assert st.tolist()==[1,0,0,1,0,0,0,0,0,0,0,8,256,0,0,0]
        assert np.array_equal(cache,self.cache)

    def begin_metadata(self,call):
        self.call=call;self.position=call-1;self.identity=(call,1,call-1)
        for pe in range(10):self.write('request_control',[call,1,call-1,call],pe)
        self.write('qk_control',[1,call-1,call],8)

    def verify_begin(self):
        for pe in (8,9):
            st=self.read('handoff_state',pe);self.observed[f'{self.call}_begin_handoff_{pe}']=st
            assert st.tolist()==list(self.identity)+[0]*29
        cache=self.read('attention_cache',9);self.observed[f'{self.call}_cache_before_projection']=cache;self.save()
        assert np.array_equal(cache,self.cache),'New contraction erased or changed retained KV'

    def source(self,case,path,values):
        bounds=self.intervals[case+'__'+path];values=np.array(values,dtype=np.float64).reshape(-1)
        assert bounds.shape==(len(values),2)
        outside=(values<bounds[:,0])|(values>bounds[:,1])|~np.isfinite(values)
        return dict(passed=not bool(np.any(outside)),count=len(values),outside=int(np.count_nonzero(outside)),
                    max_width=float(np.max(bounds[:,1]-bounds[:,0])))

    @staticmethod
    def decoded(words):return [expanded(int(w)) for w in words]

    def guards(self,back,names):
        for name in names:
            if name.endswith(('_state','_events','_control','_stats')):continue
            assert (back[name][0],back[name][-1])==((0xa55a,0x5aa5) if ARRAYS[name][1]==16 else (1234.5,-4321.25)),name

    def preprocess(self,case):
        self.launch('preprocess')
        back={n:self.read(n,8) for n in (*QK_READS,'handoff_state')}
        self.observed.update({f'{self.call}_qk_{n}':a for n,a in back.items()});self.save()
        self.qk_back=back
        self.guards(back,QK_READS)
        assert np.array_equal(back['qk_input'],self.inputs[8])
        for n in ('qk_weights','qk_frequencies','qk_probe_inputs'):
            assert np.array_equal(back[n].view(np.uint32),self.initial[n].view(np.uint32))
        assert back['qk_control'].tolist()==[1,self.position,self.call]
        assert back['qk_state'].tolist()==qk_state(1,self.position,self.call,1)
        assert back['qk_events'].tolist()==QK_EVENTS
        st=back['handoff_state'];assert st[:4].tolist()==[*self.identity,4] and st[4:10].tolist()==[QK_MASK,0,4,0,0,4] and st[12]==0 and st[21]==1
        actual={n:(back['qk_'+n] if n=='stats' else back['qk_'+n][1:-1]).tolist()
                for n in ('stats','rms_stages','normalized','trig','trig_casts','rotary_stages','product_casts','output','probes')}
        token={'position':self.position,'q':self.decoded(self.inputs[8][1:257]),'k':self.decoded(self.inputs[8][257:-1])}
        conditional=validate_qk(token,self.weights[:256],self.weights[256:],self.frequencies,actual,PROBES)
        source={};official={}
        for h,name in enumerate(('q','k')):
            prefix=f'preprocessing__heads__{h}__'
            mapped={'rms__stats':actual['stats'][h*5:(h+1)*5],
                    'rms__normalized':actual['rms_stages'][h*512:h*512+256],
                    'rms__gained':actual['rms_stages'][h*512+256:(h+1)*512],
                    'rms__output':self.decoded(actual['normalized'][h*256:(h+1)*256]),
                    'product0':self.decoded(actual['product_casts'][h*128:h*128+64]),
                    'product1':self.decoded(actual['product_casts'][h*128+64:(h+1)*128]),
                    'sums':actual['rotary_stages'][h*192+128:(h+1)*192],
                    'output':self.decoded(actual['output'][h*256:(h+1)*256])}
            for label,values in mapped.items():source[name+'_'+label]=self.source(case,prefix+label,values)
            official[name]=int(np.count_nonzero(np.array(mapped['output'])!=self.reference[case+'__'+name]))
        source['sine']=self.source(case,'preprocessing__sine',self.decoded(actual['trig_casts'][:32]))
        source['cosine']=self.source(case,'preprocessing__cosine',self.decoded(actual['trig_casts'][32:]))
        checks=dict(kind='qk',call=self.call,conditional=conditional,source=source,official_BF16_mismatches=official,
                    actual_zero_products=sum(v==0 for h in range(2) for v in actual['rotary_stages'][h*192:h*192+128]),
                    actual_zero_sums=sum(v==0 for h in range(2) for v in actual['rotary_stages'][h*192+128:(h+1)*192]))
        self.rows.append(checks);self.save();assert conditional['passed'] and all(v['passed'] for v in source.values())

    def handoff_and_compute(self,case):
        rxmask={8:0,9:0};txmask={pe:0 for pe in range(9)};received={8:0,9:0};sent={pe:0 for pe in range(9)}
        for f in FRAGMENTS:
            if f.identity==8:self.preprocess(case)
            descriptor=f.descriptor(*self.identity)
            for pe in range(10):self.write('handoff_control',descriptor,pe)
            self.launch('arm_handoff')
            armed=self.read('handoff_state',f.destination);self.observed[f'{self.call}_fragment{f.identity}_armed']=armed;self.save()
            assert armed[:4].tolist()==[*self.identity,1] and armed[4]==rxmask[f.destination] and armed[12]==0 and armed[13:16].tolist()==[0,0,0]
            self.launch('handoff')
            payload=(self.observed[f'{self.call}_{f.source}_bf16'][1:-1] if f.source<8 else
                     self.qk_back['qk_output'][1+f.destination_offset:1+f.destination_offset+128]).tolist()
            expected_frame=frame(f,self.identity,payload);expected_ack=ack(f,self.identity)
            rxmask[f.destination]|=1<<f.identity;txmask[f.source]|=1<<f.identity
            received[f.destination]+=1;sent[f.source]+=1
            endpoint_orders={}
            for pe in (f.source,f.destination):
                back={n:self.read(n,pe) for n in ('handoff_state','wire_data','wire_ack')}
                self.observed.update({f'{self.call}_fragment{f.identity}_pe{pe}_{n}':a for n,a in back.items()});self.save()
                assert back['wire_data'].tolist()==expected_frame and back['wire_ack'].tolist()==expected_ack
                st=back['handoff_state'];assert st[:4].tolist()==[*self.identity,3] and st[12]==0 and st[13:16].tolist()==[1,1,1]
                assert st[25:28].tolist()==[f.identity,f.source,f.destination]
                if pe==f.source:
                    assert st[5]==txmask[pe] and st[7]==sent[pe] and st[8]==sent[pe] and st[11]==sent[pe]*140
                    assert 0<st[29]<st[17]<st[19]<st[20]
                else:
                    assert st[4]==rxmask[pe] and st[6]==received[pe] and st[9]==received[pe] and st[10]==received[pe]*140
                    assert 0<st[17]<st[18]<st[19]<st[20] and 0<st[29]<st[20]
                endpoint_orders[str(pe)]={key:int(st[i]) for key,i in [('receive_or_send_done',17),('commit',18),('ack',19),('unblock',20),('command',29)]}
            dest=self.inputs[f.destination];dest[1+f.destination_offset:1+f.destination_offset+128]=payload
            actual=self.read('qk_input' if f.destination==8 else 'attention_input',f.destination)
            self.observed[f'{self.call}_fragment{f.identity}_destination']=actual;self.save()
            assert np.array_equal(actual,dest),'Partial commit changed another tensor/fragment or guards'
            self.transport_checks.append(dict(call=self.call,fragment=f.identity,descriptor=descriptor,exact_frame_words=142,exact_ack_words=15,
                                              receiver_mask=rxmask[f.destination],sender_mask=txmask[f.source],endpoint_event_orders=endpoint_orders))
        self.launch('attend')
        back={n:self.read(n,9) for n in (*ATTENTION_READS,'handoff_state')}
        self.observed.update({f'{self.call}_attention_{n}':a for n,a in back.items()});self.save()
        self.guards(back,ATTENTION_READS)
        assert np.array_equal(back['attention_input'],self.inputs[9])
        pos=self.position
        self.cache[1+pos*256:1+(pos+1)*256]=self.inputs[9][257:513]
        self.cache[2049+pos*256:2049+(pos+1)*256]=self.inputs[9][513:769]
        assert np.array_equal(back['attention_cache'],self.cache),'Full physical cache history/unused slots/guards differ'
        assert back['attention_state'].tolist()==[1,pos,self.call,1,self.call,3,0,0,1,self.call,self.call,8,256,0,0xa55a,0x5aa5]
        assert back['attention_events'].tolist()==NORMAL_EVENTS
        st=back['handoff_state'];assert st[:4].tolist()==[*self.identity,4] and st[4:10].tolist()==[ATTENTION_MASK,0,8,0,0,8] and st[12]==0 and st[21]==1
        token={'generation':1,'position':pos,'token':self.call}
        token.update({k:self.decoded(self.inputs[9][1+j*256:1+(j+1)*256]) for j,k in enumerate(('q','k','v','gate'))})
        actual=dict(scores=back['attention_scores'][1:-1].tolist(),score_casts=back['attention_score_casts'][1:-1].tolist(),
                    stats=back['attention_stats'].tolist(),stages=back['attention_stages'][1:-1].tolist(),output_casts=back['attention_casts'][1:-1].tolist())
        conditional=validate_attention(token,self.decoded(self.cache[1:-1]),actual)
        witness=ignored_qk_witness(token['q'],[self.decoded(self.cache[1+j*256:1+(j+1)*256]) for j in range(self.call)])
        if self.call==2:assert witness['rejected'],'Actual-operand dot gate failed to reject frozen ignore-QK counterexample'
        scores=actual['scores'];casts=actual['score_casts'];stats=actual['stats'];stage=actual['stages'];out=actual['output_casts'];n=self.call
        mapped={**{key:scores[j*8:j*8+n] for j,key in enumerate(('dot','scale','shift','exp','probability'))},
                **{key:self.decoded(casts[j*8:j*8+n]) for j,key in enumerate(('dot_bf16','scaled_bf16','probability_bf16'))},
                **{key:[stats[j]] for j,key in enumerate(('maximum','denominator','inverse'))},
                'attention':stage[:256],'gate':stage[768:1024],'product':stage[1024:],
                **{key:self.decoded(out[j*256:(j+1)*256]) for j,key in enumerate(('attention_bf16','gate_bf16','output'))}}
        source={key:self.source(case,'attention__'+key,value) for key,value in mapped.items()}
        for slot in range(n):
            source[f'cache_K_{slot}']=self.source(case,f'key_intervals__{slot}',self.decoded(self.cache[1+slot*256:1+(slot+1)*256]))
            source[f'cache_V_{slot}']=self.source(case,f'value_intervals__{slot}',self.decoded(self.cache[2049+slot*256:2049+(slot+1)*256]))
        official={key:int(np.count_nonzero(np.array(mapped[key])!=self.reference[case+'__attention_stage_'+key])) for key in
                  ('dot_bf16','scaled_bf16','attention_bf16','gate_bf16','output')}
        checks=dict(kind='attention',call=self.call,conditional=conditional,source=source,official_BF16_mismatches=official,ignore_QK_conditional_witness=witness,
                    exact_physical_cache_words=4098,valid_tokens=n,source_limit='Token2 original-source final intervals cannot detect uniform attention; conditional dot/scale/softmax checks use actual operands.')
        self.rows.append(checks);self.save();assert conditional['passed'] and all(v['passed'] for v in source.values())
        for name in ('qk_input','qk_output','qk_weights','handoff_state'):
            retained=self.read(name,8);self.observed[f'{self.call}_retained_{name}']=retained;self.save()
            if name!='handoff_state':assert np.array_equal(retained,self.qk_back[name])
            else:assert retained[:4].tolist()==[*self.identity,3] and retained[4:10].tolist()==[QK_MASK,0xf00,4,4,4,4] and retained[12]==0 and retained[21]==1
        return checks

    def verify_release(self):
        for pe in range(10):
            st=self.read('request_state',pe);self.observed[f'{self.call}_released_request_{pe}']=st
            assert st.tolist()==[self.call,1,self.call,5,self.call,self.call,1,0]
        for pe in (8,9):
            st=self.read('handoff_state',pe);self.observed[f'{self.call}_released_handoff_{pe}']=st
            assert st[:4].tolist()==[*self.identity,5] and st[12]==0 and st[21]==1
        self.save()
