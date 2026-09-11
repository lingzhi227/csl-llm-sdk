"""Connected generation checks; every observation and prerequisite is epoch-bound."""
import math
import numpy as np
from qwen38.mlp_numerics import FP32_TINY, Interval, intermediate_product, silu
from qwen38.mlp_reference_checks import decode, encode, rounded
from qwen38.mlp_streamed_source import linear_bounds
from qwen38.wp16_reuse_layout import ARRAYS, FLOAT32, sequence
from qwen38.wp16_reuse_protocol import (identity, key, packet, verify_ack, descriptor,
    numeric_state, final_numeric_state, mlp_state, initial_handoff_state,
    completed_handoff_state, lifecycle_state, tile_snapshot, weight_snapshot)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def f64_of_f32(array):
    """Read exact finite FP32 bits, including subnormals, without FP32 arithmetic."""
    require(array.dtype == np.float32, 'Observed FP32 array required')
    words = array.view(np.uint32)
    exponent = ((words >> 23) & 255).astype(np.int32)
    require(not np.any(exponent == 255), 'Nonfinite device FP32 value')
    fraction = (words & 0x7fffff).astype(np.int64)
    significant = np.where(exponent == 0, fraction, fraction + (1 << 23))
    power = np.where(exponent == 0, -149, exponent - 150)
    values = np.ldexp(significant.astype(np.float64), power)
    return np.copysign(values, np.where(words & 0x80000000, -1., 1.))


def bits_equal(left, right):
    return left.dtype == right.dtype and left.shape == right.shape and np.array_equal(left.view(np.uint32), right.view(np.uint32))


def inside(values, bounds, label):
    require(bounds.shape == (values.size, 2), label + ': bound shape')
    require(np.isfinite(values).all() and np.isfinite(bounds).all(), label + ': finite values')
    rejected = (values < bounds[:, 0]) | (values > bounds[:, 1])
    require(not np.any(rejected), label + ': ' + str(int(np.count_nonzero(rejected))) + ' interval violations')


def item_key(item):
    return key(item['generation'],item['input_index'],item['generation'],
        item['phase'],item['name'],item['pe'],item['tile'])


class Checks:
    def __init__(self,oracle,weights):
        self.all_oracles=oracle;self.weights=weights
        self.operations=sequence();self.next_operation=0;self.awaiting=None
        self.g=0;self.input_index=None
        self.observed={};self.uploaded={};self.validated=set()
        self.weight_values={};self.resident_pre=set();self.resident_post=set()
        self.pending_weights={};self.pending_projections=set();self.next_down=(1,0)
        self.reset_seen={g:set() for g in (1,2,3)}
        self.retention_seen={g:set() for g in (1,2,3)}
        self.release_seen={g:set() for g in (1,2,3)}
        self.reports=dict(handoffs=[],nonlinear=[],down=[],casts=[],retention=[],timing=[],
            tile_groups=[],reset=[],released=[],resident_weights=[])

    @property
    def oracle(self):
        require(self.g in (1,2,3),'Current source generation')
        return {name:a[self.g-1] for name,a in self.all_oracles.items()}

    def info(self,**fields):
        return dict(generation=self.g,input_index=self.input_index,contraction_id=self.g,**fields)

    def tag(self,phase,name,pe,tile=None,generation=None):
        g=self.g if generation is None else generation
        index=None if g==0 else identity(g)[1]
        return key(g,index,g,phase,name,pe,tile)

    def get(self,phase,name,pe,tile=None,generation=None):
        return self.observed[self.tag(phase,name,pe,tile,generation)]

    def before_operation(self,item):
        require(self.awaiting is None and self.next_operation<len(self.operations) and
            item==self.operations[self.next_operation],'Exact next physical operation without overlap/skip')
        g,index=item['generation'],item['input_index']
        require((g,index,g)==((0,None,0) if g==0 else identity(g)),'Exact operation epoch/input/contraction')
        phase=item['phase']
        if phase=='begin_generation':
            require(self.resident_pre=={0,1},'Both resident weight payloads read before generation1')
            if g>1:
                require(len(self.release_seen[g-1])==16,'All four prior release joins before next epoch')
                require(not self.pending_weights and not self.pending_projections,'Prior overwrite groups complete before reset')
        if g and phase not in ('begin_generation','reset_observation'):
            require(len(self.reset_seen[g])==56,'All current-generation reset observations before computation')
        if phase=='release_after_retention':
            require(len(self.retention_seen[g])==31,'All current-generation retention observations before release')
        if phase=='resident_weights_after_generation3':
            require(len(self.release_seen[3])==16,'Final resident retention only after all generation3 releases')
        if item['kind']=='launch' and item['name']=='accumulate_projections':
            require(not self.pending_projections,'Previous projection group incomplete')
            self.pending_projections={(g,0),(g,1)}
        self.g=g;self.input_index=index;self.awaiting=item

    def after_operation(self,item):
        require(self.awaiting==item,'Matching completed physical operation')
        if item.get('direction')=='d2h':require(item_key(item) in self.validated,'D2H must validate before progression')
        self.next_operation+=1;self.awaiting=None

    def upload(self,item,values):
        require(self.awaiting==item and item['direction']=='h2d','Upload belongs to current entered operation')
        name,pe,g=item['name'],item['pe'],item['generation'];tag=item_key(item)
        require(tag not in self.uploaded,'No duplicate epoch-bound upload')
        if name=='weights_storage':
            if pe<2:
                require(g==0 and pe not in self.weight_values and item['tile']==0,'Resident projection upload exactly once')
            else:
                require(pe==3 and pe not in self.pending_weights and (g,item['tile'])==self.next_down,
                    'Down overwrite only after complete prior observation group')
                self.pending_weights[pe]=(g,item['tile'])
            self.weight_values[pe]=values.copy()
        self.uploaded[tag]=values.copy()

    def reset_expected(self,name,pe):
        if name=='state':return np.array(numeric_state(self.g,pe),dtype=np.uint32)
        if name=='mlp_state':return np.array(mlp_state(self.g,pe),dtype=np.uint32)
        if name=='handoff_state':return np.array(initial_handoff_state(self.g),dtype=np.uint32)
        if name=='lifecycle_state':return np.array(lifecycle_state(self.g,pe,stage='reset'),dtype=np.uint32)
        if name=='weight_guard_snapshot':return np.array(weight_snapshot(self.g,pe,reset=True),dtype=np.uint32)
        bits,counts=ARRAYS[name];result=np.zeros(counts[pe],dtype=np.float32 if name in FLOAT32 else np.uint32)
        if name=='wire_data' or name=='wire_ack':result[0],result[-1]=0xa55aa55a,0x5aa55aa5
        elif name in ('input_storage','output_storage','mlp_exp','mlp_sigmoid','mlp_silu_fp32','mlp_product_fp32'):
            result[0],result[-1]=(77.5,-88.25) if name=='input_storage' else (1234.5,-4321.25)
        elif name in ('bf16_storage','mlp_gate','mlp_up','mlp_silu','mlp_product'):
            result[0],result[-1]=0xa55a,0x5aa5
        else:require(name in ('handoff_latched','tile_guard_snapshot','timing'),'Explicit reset observation array')
        return result

    def handoff(self,edge):
        sender,receiver=edge,2 if edge<2 else 3;phase='handoff_'+str(edge)
        tensor=('mlp_gate','mlp_up','mlp_product')[edge]
        body=self.bf16_body('finalize_projections','bf16_storage',sender) if edge<2 else self.bf16_body('whole_silu_and_product','mlp_product',2)
        frame=np.array(packet(self.g,edge,body.astype(np.uint32).tolist()),dtype=np.uint32)
        orders=[]
        for pe in (sender,receiver):
            require(bits_equal(self.get(phase,'wire_data',pe),frame),'Exact current sender/receiver data frame')
            verify_ack(self.g,edge,self.get(phase,'wire_ack',pe).tolist())
            require(self.get(phase,'handoff_latched',pe).tolist()==list(descriptor(self.g,edge)),'Exact latched current descriptor')
            value=self.get(phase,'handoff_state',pe)
            variants=[completed_handoff_state(self.g,edge,pe)]
            if pe==receiver:
                variants += [completed_handoff_state(self.g,edge,pe,receipt_first=True,command_before_ack=before) for before in (False,True)]
            require(any(value.tolist()==expected for expected in variants),'Exact event/word/mask/count handoff state')
            if pe==receiver:orders.append('receipt_before_command' if value[30] else 'command_before_receipt')
        require(np.array_equal(self.bf16_body(phase,tensor,receiver),body),'Exact device-owned current consumer operand')
        require(self.get(phase,'mlp_state',receiver).tolist()==mlp_state(self.g,receiver,incoming=(1,3,4)[edge]),'Exact receiver compute state after current commit')
        self.reports['handoffs'].append(self.info(edge=edge,payload_count=128,exact_frame_words=144,
            observed_receiver_orders=orders,protocol_passed=True))

    def receive(self,item,values):
        require(self.awaiting==item and item['direction']=='d2h','Readback belongs to current entered operation')
        phase,name,pe,tile=(item[k] for k in ('phase','name','pe','tile'));tag=item_key(item)
        require(tag not in self.observed,'Unique complete epoch/input/contraction observation key')
        require(values.dtype==(np.float32 if name in FLOAT32 else np.uint32) and values.shape==(item['count'],),'Exact readback dtype/shape')
        self.observed[tag]=values.copy()
        if phase=='initialize_epoch0':
            expected={'state':numeric_state(0,pe),'mlp_state':mlp_state(0,pe),'handoff_state':initial_handoff_state(0),
                'lifecycle_state':lifecycle_state(0,pe,stage='initialize')}[name]
            require(values.tolist()==expected,'Exact initialized epoch0 state')
        elif phase=='reset_observation':
            require(bits_equal(values,self.reset_expected(name,pe)),'Exact current reset zeros/guards/identity/counters')
            self.reset_seen[self.g].add(tag)
            self.reports['reset'].append(self.info(name=name,pe=pe))
        else:
            self.receive_active(item,values)
        self.validated.add(tag)

    def receive_active(self,item,values):
        phase,name,pe,tile=(item[k] for k in ('phase','name','pe','tile'));tag=item_key(item)
        if name=='weights_storage':
            require(pe in self.weight_values and bits_equal(values,self.weight_values[pe]),'Exact original uploaded weight readback')
            if phase=='resident_weights_before_generation1':
                self.resident_pre.add(pe);self.reports['resident_weights'].append(self.info(pe=pe,phase=phase))
            elif phase=='resident_weights_after_generation3':
                require(bits_equal(values,self.get('resident_weights_before_generation1',name,pe,0,generation=0)),'Exact resident weight retention across all generations')
                self.resident_post.add(pe);self.reports['resident_weights'].append(self.info(pe=pe,phase=phase))
        if name=='input_storage':
            if pe<2:
                upload=self.uploaded[self.tag('projection_tile',name,pe,0)]
                require(bits_equal(values,upload),'Exact current original projection input/guards')
            else:
                words=self.bf16_body('handoff_2','mlp_product',3)
                offset=64*(tile if tile is not None else 1)
                expected=np.zeros(66,dtype=np.float32);expected[0],expected[-1]=77.5,-88.25
                expected.view(np.uint32)[1:-1]=words[offset:offset+64].astype(np.uint32)<<16
                require(bits_equal(values,expected),'Exact current device BF16 down-input expansion')
        if name=='state':
            if phase in ('projection_tile_readback','down_tile_readback'):
                expected=numeric_state(self.g,pe,phase=1,tiles=tile+1,consumed=112 if pe<2 else (tile+1)*64)
            else:expected=final_numeric_state(self.g,pe,released=phase=='released_join')
            require(values.tolist()==expected,'Exact current numeric commit/phase/counters')
        if name=='weight_guard_snapshot':require(values.tolist()==weight_snapshot(self.g,pe),'Exact current final guard snapshot and boundary serial')
        if name=='tile_guard_snapshot':
            require(values.tolist()==tile_snapshot(self.g,pe,tile),'Exact current pre/post guard and tile identity snapshot')
            names=['input_storage','output_storage','state']
            if pe==3:
                require(self.pending_weights.get(pe)==(self.g,tile),'Matching current pending down overwrite')
                names.append('weights_storage')
            else:require((self.g,pe) in self.pending_projections,'Matching current pending projection observation')
            require(all(self.tag(phase,n,pe,tile) in self.validated for n in names),'Complete current tile observations before overwrite')
            if pe==3:
                del self.pending_weights[pe];self.next_down=(self.g,1) if tile==0 else (self.g+1,0)
            else:self.pending_projections.remove((self.g,pe))
            self.reports['tile_groups'].append(self.info(pe=pe,tile=tile,consumed_columns=int(values[9]),full_weight_payload_bitwise_D2H=pe==3))
        if name=='timing':
            self.check_timing(values,pe,phase)
            if phase=='whole_silu_and_product':self.nonlinear()
        if name=='output_storage' and phase in ('projection_tile_readback','finalize_projections'):
            stage='gate' if pe==0 else 'up'
            inside(self.float_body(phase,name,pe,tile),self.oracle[stage+'_fp32'],'Current original112 projection FP32 source')
        if name=='bf16_storage' and phase=='finalize_projections':
            stage='gate' if pe==0 else 'up';self.check_cast(phase,name,'output_storage',pe)
            inside(decode(self.bf16_body(phase,name,pe)),self.oracle[stage+'_bf16'],'Current projection BF16 source')
        if phase.startswith('handoff_') and name=='mlp_state':self.handoff(int(phase[-1]))
        if pe==3 and name=='output_storage' and phase=='down_tile_readback':self.down_output(phase,tile)
        if pe==3 and name=='bf16_storage' and phase=='finalize_down':
            self.check_cast(phase,name,'output_storage',pe);value=decode(self.bf16_body(phase,name,pe))
            inside(value,self.oracle['down_bf16'],'Current partial source BF16')
            inside(value,self.down_output(phase),'Current conditional device-product BF16')
        if phase=='retention_after_down':self.retention(name,pe,values,tag)
        if phase=='released_join':
            if name=='handoff_state':
                expected=self.get('retention_after_down',name,pe).copy();expected[0]=5;expected[29]+=1
                require(bits_equal(values,expected),'Release preserves completed current handoff/ACK state')
            elif name=='mlp_state':
                expected=mlp_state(self.g,pe,incoming=3 if pe==2 else 4 if pe==3 else 0,
                    computed=pe>=2,tiles=2 if pe==3 else 0,released=True)
                require(values.tolist()==expected,'Exact current released MLP state')
            elif name=='lifecycle_state':require(values.tolist()==lifecycle_state(self.g,pe,stage='released'),'Exact cumulative current release join')
            self.release_seen[self.g].add(tag);self.reports['released'].append(self.info(name=name,pe=pe))

    def retention(self,name,pe,values,tag):
        if name=='lifecycle_state':require(values.tolist()==lifecycle_state(self.g,pe,stage='retention'),'Exact completed lifecycle before release')
        else:
            if name=='handoff_state':original=self.get('handoff_'+str(pe if pe<2 else 2),name,pe)
            elif pe==2:original=self.get('whole_silu_and_product',name,pe)
            elif name in ('weights_storage','input_storage'):original=self.get('projection_tile_readback' if pe<2 else 'down_tile_readback',name,pe,0 if pe<2 else 1)
            elif name=='mlp_product':original=self.get('handoff_2',name,3)
            else:original=self.get('finalize_projections' if pe<2 else 'finalize_down',name,pe)
            require(bits_equal(values,original),'Exact current-generation retained numerical/protocol operand')
        self.retention_seen[self.g].add(tag);self.reports['retention'].append(self.info(name=name,pe=pe,words=values.size))

    def check_timing(self,array,pe,phase):
        require(array.dtype==np.uint32 and array.size%6==0 and not np.any(array>>16),'Raw native16 timestamp payload')
        active=2 if pe==3 else 1
        require(not np.any(array[active*6:]),'Unused current-generation timestamp words must remain zero')
        cycles=[]
        for row in array[:active*6].reshape(-1,6).tolist():
            start=row[0]+(row[1]<<16)+(row[2]<<32);end=row[3]+(row[4]<<16)+(row[5]<<32)
            delta=(end-start)&((1<<48)-1);require(0<delta<1<<40,'Positive bounded current timestamp delta');cycles.append(delta)
        self.reports['timing'].append(self.info(pe=pe,phase=phase,cycles=cycles))

    def finish(self):
        require(self.next_operation==696 and self.awaiting is None,'All exact entered operations completed')
        require(not self.pending_weights and not self.pending_projections and self.next_down==(4,0),'All current and inter-generation overwrite groups completed')
        require(self.resident_pre==self.resident_post=={0,1},'Both resident weight endpoint reads completed')
        require(all(len(self.reset_seen[g])==56 and len(self.retention_seen[g])==31 and len(self.release_seen[g])==16 for g in (1,2,3)),'All three reset/retention/release groups')
        require(len(self.reports['tile_groups'])==12 and len(self.reports['handoffs'])==9 and len(self.reports['casts'])==15 and len(self.reports['down'])==9 and len(self.reports['nonlinear'])==3,'Complete connected numerical and transport coverage')
        return self.reports



    def bf16_body(self, phase, name, pe, tile=None):
        array = self.get(phase, name, pe, tile)
        require(array.dtype == np.uint32 and array.shape == (130,), 'Guarded BF16 shape/type')
        require(array[0] == 0xa55a and array[-1] == 0x5aa5 and not np.any(array >> 16), 'Guarded BF16 integrity')
        words = array[1:-1].astype('<u2')
        decode(words)
        return words


    def float_body(self, phase, name, pe, tile=None):
        array = self.get(phase, name, pe, tile)
        require(array.dtype == np.float32 and array.shape == (130,), 'Guarded FP32 shape/type')
        require(array[0] == 1234.5 and array[-1] == -4321.25, 'Guarded FP32 integrity')
        return f64_of_f32(array[1:-1])


    def check_cast(self, phase, bf16_name, fp32_name, pe):
        actual = self.bf16_body(phase, bf16_name, pe)
        expected = encode(rounded(self.float_body(phase, fp32_name, pe)))
        require(np.array_equal(actual, expected), phase + ': exact observed FP32 to BF16 RNE')
        self.reports['casts'].append(self.info(phase=phase, pe=pe, source=fp32_name,
            destination=bf16_name, count=128, bit_mismatches=0))


    def nonlinear(self):
        phase = 'whole_silu_and_product'
        for edge, name in ((0, 'mlp_gate'), (1, 'mlp_up')):
            require(bits_equal(self.get(phase, name, 2), self.get('handoff_' + str(edge), name, 2)),
                    'Exact received operand retained through nonlinear compute: ' + name)
        gate = decode(self.bf16_body(phase, 'mlp_gate', 2))
        upper = decode(self.bf16_body(phase, 'mlp_up', 2))
        act = decode(self.bf16_body(phase, 'mlp_silu', 2))
        prod = decode(self.bf16_body(phase, 'mlp_product', 2))
        observed = {name: self.float_body(phase, name, 2) for name in
                    ('mlp_exp', 'mlp_sigmoid', 'mlp_silu_fp32', 'mlp_product_fp32')}
        require(np.max(np.abs(gate)) <= 24 and np.max(np.abs(upper)) <= 24, 'Device nonlinear domain')
        inside(gate, self.oracle['gate_bf16'], 'Original gate source')
        inside(upper, self.oracle['up_bf16'], 'Original up source')
        inside(act, self.oracle['silu_bf16'], 'Original whole SiLU source')
        inside(prod, self.oracle['product_bf16'], 'Original product source')
        inside(observed['mlp_silu_fp32'], self.oracle['silu_fp32'], 'Source FP32 whole SiLU')
        inside(observed['mlp_product_fp32'], self.oracle['product_fp32'], 'Source FP32 product')
        for i, g in enumerate(gate):
            exponential = math.exp(-abs(float(g)))
            e_allowance = exponential * (2.**-20 + 2.**-44) + FP32_TINY
            require(abs(observed['mlp_exp'][i] - exponential) <= math.nextafter(e_allowance, math.inf), 'Actual-gate bounded exp allowance')
            ideal_sigmoid = exponential / (1. + exponential) if g < 0 else 1. / (1. + exponential)
            s_allowance = ideal_sigmoid * (2.**-18 + 2.**-44) + FP32_TINY
            require(abs(observed['mlp_sigmoid'][i] - ideal_sigmoid) <= math.nextafter(s_allowance, math.inf), 'Actual-gate stable sigmoid allowance')
            activation = silu(Interval.point(float(g)))
            multiplication = intermediate_product(Interval.point(float(act[i])), Interval.point(float(upper[i])))
            for stage, value in ((activation, observed['mlp_silu_fp32'][i]), (multiplication, observed['mlp_product_fp32'][i])):
                require(stage['fp32'].lo <= value <= stage['fp32'].hi, 'Conditional actual-operand FP32 arithmetic')
        self.check_cast(phase, 'mlp_silu', 'mlp_silu_fp32', 2)
        self.check_cast(phase, 'mlp_product', 'mlp_product_fp32', 2)
        mlp = self.get(phase, 'mlp_state', 2)
        require(mlp.tolist() == mlp_state(self.g,2,incoming=3,computed=True), 'Exact nonlinear compute commit/counts')
        self.reports['nonlinear'].append(self.info(count=128,source_passed=True,conditional_passed=True,
            observed_FP32_stages=4,extra_sigmoid_BF16_boundary=False))


    def down_output(self, phase, tile=None):
        columns = 64 * (tile + 1) if tile is not None else 128
        values = self.float_body(phase, 'output_storage', 3, tile)
        name = 'down_prefix64_fp32' if columns == 64 else 'down_fp32'
        inside(values, self.oracle[name], 'Original128-channel partial down source')
        product = decode(self.bf16_body('handoff_2', 'mlp_product', 3))[:columns]
        points = np.column_stack((product, product))
        conditional = np.empty((128, 2), dtype=np.float64)
        conditional_bf16 = np.empty((128, 2), dtype=np.float64)
        for first in (0, 64):
            result = linear_bounds(decode(self.weights['down_proj'][first:first + 64, :columns]), points)
            conditional[first:first + 64] = result['fp32']
            conditional_bf16[first:first + 64] = result['bf16']
        inside(values, conditional, 'Conditional device-product partial down')
        self.reports['down'].append(self.info(phase=phase, consumed_columns=columns,
            source_and_conditional_passed=True, outputs=128, BF16_prefix_boundary=False if columns == 64 else None))
        return conditional_bf16
