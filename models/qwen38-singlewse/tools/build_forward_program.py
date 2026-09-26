"""Lower every text layer into a bounded four-u16 device instruction stream.

All operands refer to resident PEs. The future device interpreter must execute
the entire stream and token-feedback loop; this file does not execute inference.
"""
import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def build():
    role_path=ROOT/'configs/role-plan-96.json';roles=json.loads(role_path.read_text())
    matrix_path=ROOT/'configs/device-matrices-bf16-140.json';matrices=json.loads(matrix_path.read_text())
    tensors=json.loads((ROOT/'configs/tensors.json').read_text())['tensors']
    required=set(json.loads((ROOT/'configs/capacity-bf16-140.json').read_text())['assignments'])
    instructions=[];annotations=[];bindings=set();sizes={}
    io=[dict(kind='activation',name=b['name'],coordinate=b['coordinate'],capacity=b['capacity_bf16_elements']) for b in roles['activation_banks']]
    def emit(op,a=0,b=0,c=0,description=''):
        row=[op,a,b,c];assert all(type(x) is int and 0<=x<65536 for x in row)
        instructions.append(row);annotations.append(description)
    def norm(layer,post=False,final=False):
        name='model.language_model.norm.weight' if final else f'model.language_model.layers.{layer}.'+('post_attention_layernorm.weight' if post else 'input_layernorm.weight')
        target=len(io);assert tensors[name]['shape']==[5120] and sizes[0]==5120
        io.append(dict(kind='normalization',name=name,coordinate=roles['normalization'][name],capacity=5120))
        bindings.add(name);sizes[target]=5120
        emit(2,0,target,5120,description='Copy hidden to the original norm owner; in-place zero-centered RMSNorm')
        return target
    def matrix(name,source,target):
        spec=tensors[name];m,k=spec['shape'];assert sizes[source]==k
        assert io[target]['capacity']>=m
        index=matrices['matrix_ids'][name];bindings.add(name)
        if spec['dtype']=='F8_E4M3':bindings.add(name+'_scale_inv')
        emit(3,index,source,target,description=name);sizes[target]=m
    embed='model.language_model.embed_tokens.weight';bindings.add(embed);sizes[0]=5120
    emit(1,matrices['matrix_ids'][embed],0,5120,description='Original complete vocabulary embedding of device current_token')
    for layer in range(64):
        prefix=f'model.language_model.layers.{layer}';normalized=norm(layer)
        if layer%4!=3:
            p=prefix+'.linear_attn'
            for suffix,target in [('in_proj_qkv',3),('in_proj_z',4),('in_proj_a',5),('in_proj_b',6)]:matrix(p+'.'+suffix+'.weight',normalized,target)
            assert [sizes[i] for i in [3,4,5,6]]==[10240,6144,48,48]
            bindings.update(p+'.'+n for n in ['conv1d.weight','A_log','dt_bias','norm.weight'])
            emit(4,layer,0,7,description='All48 original GDN heads, persistent convolution/recurrent state and gated norm')
            sizes[7]=6144;matrix(p+'.out_proj.weight',7,8)
        else:
            p=prefix+'.self_attn'
            for suffix,target in [('q_proj',3),('k_proj',5),('v_proj',6)]:matrix(p+'.'+suffix+'.weight',normalized,target)
            assert [sizes[i] for i in [3,5,6]]==[12288,1024,1024]
            bindings.update([p+'.q_norm.weight',p+'.k_norm.weight'])
            emit(5,layer,0,7,description='All4 KV groups/24 query heads; per-head query/gate split, norm,RoPE,cache,softmax,gate')
            sizes[7]=6144;matrix(p+'.o_proj.weight',7,8)
        assert sizes[8]==sizes[0]==5120
        emit(6,0,8,5120,description='BF16 attention residual in-place in hidden')
        normalized=norm(layer,post=True)
        matrix(prefix+'.mlp.gate_proj.weight',normalized,9);matrix(prefix+'.mlp.up_proj.weight',normalized,10)
        assert sizes[9]==sizes[10]==17408
        emit(7,9,10,11,description='BF16 SiLU(gate), then BF16 product with up, all17408 entries');sizes[11]=17408
        matrix(prefix+'.mlp.down_proj.weight',11,12)
        assert sizes[12]==5120
        emit(6,0,12,5120,description='BF16 MLP residual in-place in hidden')
    normalized=norm(None,final=True);head='lm_head.weight';bindings.add(head)
    emit(8,matrices['matrix_ids'][head],normalized,0,description='Full248320-entry original BF16 head; masked lowest-ID argmax retained on device')
    assert bindings==required,(sorted(required-bindings),sorted(bindings-required))
    assert len(io)==145 and len(instructions)==883
    assert sum(row[0]==3 for row in instructions)==496
    # Operator groups have compact exact coordinate formulas; no per-head
    # 2304-row controller address table is needed in the southeast controller.
    for index,g in enumerate(roles['gdn']):
        if index<555:xy=[744,154+index]
        else:n=index-555;xy=[720+(n%9)*3,709+n//9]
        assert g['controller']==xy and g['state_shards']==[[xy[0]+1,xy[1]],[xy[0]+2,xy[1]]]
    for index,g in enumerate(roles['attention']):
        assert g['controller']==[744,90+index]
        assert g['shards']==[[745+i,90+index] for i in range(4)]
    return dict(status='Complete lowered device program data; interpreter/global role integration not yet implemented',
        full_model_complete=False,model=roles['model'],revision=roles['revision'],context=96,
        text_tensors_bound=len(bindings),instruction_count=883,program_storage_bytes=883*8,
        matrix_descriptor_bytes=matrices['storage_bytes'],io_coordinate_bytes=len(io)*4,
        total_controller_table_bytes=883*8+matrices['storage_bytes']+len(io)*4,
        instruction_format=['opcode','a','b','c'],opcodes=dict(embedding=1,norm=2,matrix=3,gdn=4,attention=5,residual=6,silu_multiply=7,head_argmax=8),
        instructions=instructions,annotations=annotations,io=io,
        token_loop='Device current_token selects prompt tokens, then full-head argmax feedback; host supplies request and reads completed tokens/timing only',
        gdn_coordinate_formula='index=GDN_layer_ordinal*48+value_head; index<555 ->(744,154+index); else n=index-555 ->(720+3*(n%9),709+n//9)',
        attention_coordinate_formula='index=full_attention_layer_ordinal*4+kv_head ->(744,90+index)',
        role_plan_sha256=hashlib.sha256(role_path.read_bytes()).hexdigest(),matrix_table_sha256=hashlib.sha256(matrix_path.read_bytes()).hexdigest())

if __name__=='__main__':
    result=build();(ROOT/'configs/forward-program.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['instructions','annotations','io']},indent=2))
