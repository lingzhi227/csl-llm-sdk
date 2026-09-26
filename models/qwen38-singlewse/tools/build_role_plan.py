"""Allocate adjacent operator groups without moving any140-row-atlas weight.

This supplies complete control/state coordinates for later CSL integration;
accepted standalone group code has not yet been composed with the global bus.
"""
import hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build():
    source=ROOT/'configs/placement-resident-96-bf16-140.json'
    atlas=json.loads(source.read_text());width,height=atlas['application']
    occupied=bytearray(width*height)
    def claim(x,y,n,role):
        assert 0<=y<height and 0<=x<x+n<=width
        offset=y*width+x;assert not any(occupied[offset:offset+n])
        occupied[offset:offset+n]=bytes([role])*n
    for name,a in atlas['assignments'].items():
        if 'strips' in a:
            n=a['shape'][1]//128
            for _,x,y in a['strips']:claim(x,y,n,1)
        elif a.get('role')=='other':
            for x,y in a['coordinates']:claim(x,y,1,2)
    assert occupied.count(1)+occupied.count(2)==849313
    for x,y in atlas['bus']['coordinates']:claim(x,y,1,3)
    def groups(count,size,role):
        result=[];cursor=0
        for _ in range(count):
            while cursor<len(occupied):
                x=cursor%width;y=cursor//width
                if x+size<=width and not any(occupied[cursor:cursor+size]):
                    claim(x,y,size,role);result.append([[x+i,y] for i in range(size)]);cursor+=size;break
                cursor+=1
            else:raise ValueError('Adjacent operator group does not fit')
        return result
    # Allocate scarce five-consecutive-cell runs before the three-cell groups.
    attention_coordinates=iter(groups(64,5,4))
    gdn_coordinates=iter(groups(2304,3,5))
    attention=[];gdn=[]
    for layer in range(64):
        if layer%4==3:
            for kv_head in range(4):
                coords=next(attention_coordinates)
                attention.append(dict(layer=layer,kv_head=kv_head,query_heads=list(range(kv_head*6,(kv_head+1)*6)),
                    controller=coords[0],shards=coords[1:],channels_per_shard=64,context=96,
                    cache_bytes_per_shard=24576,replicated_controller_parameter_bytes=1152))
        else:
            for value_head in range(48):
                coords=next(gdn_coordinates);key_head=value_head//3
                gdn.append(dict(layer=layer,value_head=value_head,key_head=key_head,
                    controller=coords[0],state_shards=coords[1:],state_bytes_per_shard=32768,
                    conv_channel_starts=[key_head*128,2048+key_head*128,4096+value_head*128],
                    channels_per_segment=128,history_bytes=3072,replicated_controller_parameter_bytes=3332))
    assert len(gdn)==2304 and len(attention)==64
    banks=[]
    names=['hidden','residual','normalized','qkv_or_query_gate','z_or_gate','a_or_key','b_or_value','attention_context',
           'attention_output','mlp_gate','mlp_up','mlp_product','mlp_down','scratch0','scratch1','scratch2']
    for name,coords in zip(names,groups(16,1,6)):
        banks.append(dict(name=name,coordinate=coords[0],capacity_bf16_elements=17536,
                          scope='Reusable activation storage; full bank/communication SRAM not compiled'))
    norms={name:a['coordinates'][0] for name,a in atlas['assignments'].items() if
        name.endswith(('input_layernorm.weight','post_attention_layernorm.weight')) or name=='model.language_model.norm.weight'}
    assert len(norms)==129
    assert occupied.count(0)==11530
    trace_sizes=dict(layer_outputs=math.ceil(96*64*5120/17536),
                     full_logits=math.ceil(96*248320/17536),final_norm=math.ceil(96*5120/17536))
    assert trace_sizes==dict(layer_outputs=1794,full_logits=1360,final_norm=29)
    trace_coordinates=[g[0] for g in groups(sum(trace_sizes.values()),1,7)]
    for i,xy in enumerate(trace_coordinates):
        if i==0:expected=[748,161]
        elif i<1483:n=i-1;expected=[747+n%2,162+n//2]
        elif i<1503:expected=[729+i-1483,903]
        else:n=i-1503;expected=[720+n%29,904+n//29]
        assert xy==expected
    assert occupied.count(0)==8347
    return dict(status='Complete candidate role coordinates, not an integrated CSL program',full_model_complete=False,
        model=atlas['model'],revision=atlas['revision'],context=96,application=atlas['application'],
        weight_atlas_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        controller=[749,1159],controller_in_reserved_bus=True,original_weight_pes=849313,
        bus_pes=1909,gdn_group_pes=6912,attention_group_pes=320,activation_bank_pes=16,
        validation_bank_pes=len(trace_coordinates),unallocated_pes=8347,occupancy_sha256=hashlib.sha256(occupied).hexdigest(),
        replicated_controller_parameter_bytes=2304*3332+64*1152,
        convolution_history_bytes=2304*3072,
        original_weights_unchanged=True,old_abstract_state_coordinates_replaced=True,
        gdn=gdn,attention=attention,normalization=norms,activation_banks=banks,
        validation=dict(enabled_by_request=False,capacity_bf16_elements_per_bank=17536,
            banks_per_capture=trace_sizes,coordinates=trace_coordinates,
            scope='Candidate on-wafer capture of every layer output, complete vocabulary logits and final norm at up to96 positions; no reference values enter neural computation',
            coordinate_formula='i=0:(748,161); i<1483:n=i-1,(747+n%2,162+n//2); i<1503:(729+i-1483,903); else n=i-1503,(720+n%29,904+n//29)',
            timing='Capture-enabled validation must be distinguished from capture-disabled measured requests'),
        unresolved=['Global bus and local group descriptor/task/queue/color leases',
                    'Integrated SRAM for every role, including dynamic coordinates and activation banks',
                    'All-layer command executor, initialization and token feedback',
                    'Whole-wafer compiler/artifact/runtime admission and complete physical numerical acceptance'])

if __name__=='__main__':
    result=build();(ROOT/'configs/role-plan-96.json').write_text(json.dumps(result,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['gdn','attention','normalization','activation_banks','validation']},indent=2))
