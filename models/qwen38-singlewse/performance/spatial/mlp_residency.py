"""Full checkpoint residency candidate anchored to the real MLP coordinates.

MLP matrices keep their spatial ownership across all64 layers. Embedding-only
cohosts make collectors/turn actors useful storage without adding GEMV code to
those roles. Other matrices and auxiliary pages receive exact compact addresses;
their compute routes, full role programs and dynamic live buffers are unqualified.
"""
from array import array
from bisect import bisect_right
from collections import Counter
import math
import re


class MlpResidency:
    width=750
    height=1160
    payload_limit=35256

    def __init__(self, flow, graph, tensors):
        if flow.spec.columns!=9:
            raise ValueError('Current storage profiles are qualified only for nine columns')
        self.flow=flow;self.graph=graph;self.tensors=tensors
        self.names=list(dict.fromkeys(w for n in graph['nodes'] for w in n['weights']))
        if len(self.names)!=1251 or graph['revision']!=flow.semantic['revision'] or graph['context']!=96:
            raise ValueError('Full pinned text graph required')
        self.embedding='model.language_model.embed_tokens.weight'
        self.helper_count=(flow.groups+flow.spec.columns)*flow.k
        self.embedding_reserved=68*self.helper_count
        self.spectator_count=870000-9104-696320-self.helper_count
        self.lower_bf_up_count=3*348160+17*348160+18*self.spectator_count+68*self.helper_count-10024960
        self.matrices={};self.other=[];fp8_total=bf16_total=0;used=set()
        scale_names={n+'_scale_inv' for n in self.names if tensors[n]['dtype']=='F8_E4M3'}
        for name in self.names:
            if name in scale_names:continue
            s=tensors[name];used.add(name)
            if len(s['shape'])!=2:
                self.other.append(name);continue
            m,k=s['shape'];assert m%2==k%128==0
            count=m//2*(k//128);kind='fp8' if s['dtype']=='F8_E4M3' else 'bf16'
            assert s['dtype'] in ['F8_E4M3','BF16']
            record=dict(tensor=name,shape=[m,k],tiles=count,kind=kind)
            match=re.fullmatch(r'model.language_model.layers.(\d+).mlp.(gate|up|down)_proj.weight',name)
            if kind=='fp8':
                scale=name+'_scale_inv';used.add(scale)
                assert tensors[scale]['dtype']=='BF16' and tensors[scale]['shape']==[math.ceil(m/128),k//128]
            if match:
                layer=int(match[1]);branch=match[2]
                assert 0<=layer<64 and s['shape']==([5120,17408] if branch=='down' else [17408,5120])
                record.update(mapping='mlp-fixed',layer=layer,branch=branch)
            elif kind=='fp8':
                record.update(mapping='other-fp8',stream_start=fp8_total);fp8_total+=count
            else:
                reserved=self.embedding_reserved if name==self.embedding else 0
                record.update(mapping='bf16',reserved_prefix=reserved,stream_start=bf16_total)
                bf16_total+=count-reserved
            self.matrices[name]=record
        assert used==set(self.names) and len(self.matrices)==498
        assert len([m for m in self.matrices.values() if m['mapping']=='mlp-fixed'])==192
        assert (fp8_total,bf16_total)==(28180480,10024960-self.embedding_reserved)
        self.other_fp8_total=fp8_total;self.normal_bf16_total=bf16_total
        self.gdn_layers=[int(n['attributes']['state'].split('.')[1]) for n in graph['nodes'] if n['op']=='gated_delta_recurrence']
        self.attention_layers=[int(n['attributes']['state'].split('.')[1]) for n in graph['nodes'] if n['op']=='kv_append']
        assert len(self.gdn_layers)==48 and len(self.attention_layers)==16
        n=self.width*self.height
        self.roles=bytearray(n);self.fp8=array('H',[0])*n;self.bf16=array('H',[0])*n
        self.base=array('H',[0])*n;self.state=array('H',[0])*n
        self.fp8_prefix=array('I',[0]);self.bf16_prefix=array('I',[0]);self.aux_prefix=array('I',[0])
        self.helpers=array('I');normal_up=0;fp=bf=aux=0;self.histogram=Counter()
        for index in range(n):
            x,y=index%self.width,index//self.width;role=self.classify(x,y);self.roles[index]=role
            base=128 if role==1 else 64 if role in (2,3) else 0
            state=4096 if role==3 else 0
            if role==0:bfslots=0
            elif role==1:bfslots=3
            elif role in (2,3):
                bfslots=16 if role==2 and normal_up<self.lower_bf_up_count else 17
                if role==2:normal_up+=1
            elif role==4:bfslots=68;self.helpers.append(index)
            else:bfslots=18
            capacity=0 if role in (0,4) else (self.payload_limit-base*260-bfslots*512-state)//260
            extra=min(capacity,max(0,fp8_total-fp));fp+=extra
            self.fp8[index]=base+extra;self.bf16[index]=bfslots;self.base[index]=base;self.state[index]=state
            if role!=4:bf+=bfslots
            self.fp8_prefix.append(fp);self.bf16_prefix.append(bf)
            used_bytes=(base+extra)*260+bfslots*512+state
            if role not in (0,4):aux+=(self.payload_limit-used_bytes)//128
            self.aux_prefix.append(aux)
            self.histogram[role,base+extra,bfslots,state]+=1
            assert used_bytes<=self.payload_limit
        assert len(self.helpers)==self.helper_count and normal_up==311296 and fp==fp8_total and bf==bf16_total
        assert Counter(self.roles)=={0:9104,1:348160,2:311296,3:36864,4:self.helper_count,5:self.spectator_count}
        self.auxiliary=[];self.norms=[];cursor=0
        for name in self.other:
            if re.fullmatch(r'model.language_model.layers.\d+.(input_layernorm|post_attention_layernorm).weight',name) or name=='model.language_model.norm.weight':
                assert tensors[name]['shape']==[5120] and tensors[name]['dtype']=='BF16'
                self.norms.append(dict(tensor=name,slot=len(self.norms),bytes_per_owner=256,owners=40));continue
            size=tensors[name]['bytes'];pages=(size+127)//128
            self.auxiliary.append(dict(id=name,kind='original-weight',bytes=size,pages=pages,page_start=cursor));cursor+=pages
        for layer in self.attention_layers:
            size=2*4*96*256*2
            self.auxiliary.append(dict(id=f'kv:{layer}',kind='persistent-state',shape=[2,4,96,256],dtype='BF16',bytes=size,pages=size//128,page_start=cursor));cursor+=size//128
        for layer in self.gdn_layers:
            size=10240*3*2
            self.auxiliary.append(dict(id=f'conv:{layer}',kind='persistent-state',shape=[3,10240],dtype='BF16',bytes=size,pages=size//128,page_start=cursor));cursor+=size//128
        assert cursor<=aux;self.used_aux_pages=cursor
        assert len(self.norms)==129 and len(self.norms)*256<=self.payload_limit

    def classify(self,x,y):
        """0 dedicated,1 gate,2 up,3 up+GDN,4 embedding helper,5 other bank."""
        if not (0<=x<750 and 0<=y<1160):raise ValueError('Wafer coordinate')
        if y<40:
            if x==0:return 0
            if x<747:
                local=x%83
                if local==40-y:return 4
                if local==42+y:return 0
            return 5
        bx,lx=divmod(x,83);by,ly=divmod(y-40,65);group=by*9+bx
        if bx>=9 or group>=136:return 5
        if ly==0:return 4 if 1<=lx<=40 else 5
        if lx==41:return 0
        if 1<=lx<=40:return 1
        if 42<=lx<=81:
            pair=ly-1;k=lx-42;head=group*160+(pair//4)*10+k//4
            return 3 if head<2304 else 2
        return 5

    def bank(self,index):
        if not 0<=index<870000:raise ValueError('PE')
        role=self.roles[index];fp=int(self.fp8[index]);bf=int(self.bf16[index]);state=int(self.state[index])
        return dict(xy=[index%750,index//750],role=role,fp8_slots=fp,bf16_slots=bf,
                    fp8_weight_base=0,fp8_scale_base=fp*256,bf16_base=fp*260,
                    matrix_bytes=fp*260+bf*512,gdn_state_offset=fp*260+bf*512,gdn_state_bytes=state,
                    aux_base=fp*260+bf*512+state,aux_capacity_pages=int(self.aux_prefix[index+1]-self.aux_prefix[index]),
                    original_embedding_only=role==4)

    def address(self,name,tile):
        m=self.matrices[name]
        if not 0<=tile<m['tiles']:raise ValueError('Tile')
        row,k=divmod(tile,m['shape'][1]//128);format='column-major-2x128'
        if m['mapping']=='mlp-fixed':
            layer,branch=m['layer'],m['branch']
            if branch=='down':xy=self.flow.worker(k,'gate',row%64,row//64);slot=2*layer+1
            else:xy=self.flow.worker(row//64,branch,row%64,k);slot=2*layer if branch=='gate' else layer
            index=xy[1]*750+xy[0]
        elif m['mapping']=='other-fp8':
            absolute=m['stream_start']+tile;index=bisect_right(self.fp8_prefix,absolute)-1
            slot=int(self.base[index])+absolute-self.fp8_prefix[index]
        elif tile<m['reserved_prefix']:
            index=self.helpers[tile//68];slot=tile%68;format='row-major-2x128'
        else:
            absolute=m['stream_start']+tile-m['reserved_prefix'];index=bisect_right(self.bf16_prefix,absolute)-1
            slot=absolute-self.bf16_prefix[index]
        bank=self.bank(index);assert slot<(bank['fp8_slots'] if m['kind']=='fp8' else bank['bf16_slots'])
        offset=slot*256 if m['kind']=='fp8' else bank['bf16_base']+slot*512
        return dict(xy=bank['xy'],slot=slot,byte_offset=offset,bytes=256 if m['kind']=='fp8' else 512,format=format,
                    rows=[row*2,row*2+2],columns=[k*128,k*128+128],
                    scale_byte_offset=bank['fp8_scale_base']+slot*4 if m['kind']=='fp8' else None,
                    original_scale_index=[row//64,k] if m['kind']=='fp8' else None)

    def auxiliary_address(self,page):
        if not 0<=page<self.used_aux_pages:raise ValueError('Auxiliary page')
        index=bisect_right(self.aux_prefix,page)-1;bank=self.bank(index)
        return dict(xy=bank['xy'],byte_offset=bank['aux_base']+128*(page-self.aux_prefix[index]),bytes=128)

    def norm_address(self,name,group):
        if not 0<=group<40:raise ValueError('Norm group')
        records=[n for n in self.norms if n['tensor']==name]
        if len(records)!=1:raise ValueError('Original norm')
        return dict(xy=[0,group],byte_offset=records[0]['slot']*256,bytes=256,rows=[128*group,128*group+128])

    def gdn_address(self,layer,head,key_shard,value_shard):
        if layer not in self.gdn_layers or not 0<=head<48 or not 0<=key_shard<4 or not 0<=value_shard<4:
            raise ValueError('GDN state shard')
        ordinal=self.gdn_layers.index(layer)*48+head;group,cell=divmod(ordinal,160)
        xy=self.flow.worker(group,'up',(cell//10)*4+key_shard,(cell%10)*4+value_shard)
        bank=self.bank(xy[1]*750+xy[0]);assert bank['gdn_state_bytes']==4096
        return dict(xy=list(xy),byte_offset=bank['gdn_state_offset'],bytes=4096,
                    rows=[32*key_shard,32*key_shard+32],columns=[32*value_shard,32*value_shard+32])

    def document(self):
        hist=[dict(role=role,fp8_slots=fp,bf16_slots=bf,gdn_state_bytes=s,pes=count,
                   matrix_and_gdn_bytes=fp*260+bf*512+s) for (role,fp,bf,s),count in sorted(self.histogram.items())]
        return dict(schema='wse-mlp-anchored-residency-v1',revision=self.graph['revision'],application=[750,1160],
                    original_tensors=len(self.names),original_bytes=sum(self.tensors[n]['bytes'] for n in self.names),
                    matrix_tiles=sum(m['tiles'] for m in self.matrices.values()),matrices=list(self.matrices.values()),
                    role_names={0:'dedicated activation/normalization/up-distributor',1:'MLP gate/down bank',2:'MLP up bank',
                                3:'MLP up plus GDN state bank',4:'embedding-only collector/header',5:'other bank'},
                    profiles=hist,bank_pes=860896,dedicated_pes=9104,embedding_helpers=self.helper_count,
                    embedding_helper_tiles=self.embedding_reserved,embedding_helper_payload_bytes=34816,
                    payload_ceiling=self.payload_limit,other_fp8_tiles=self.other_fp8_total,normal_bf16_tiles=self.normal_bf16_total,
                    gdn_state=dict(layers=self.gdn_layers,heads_per_layer=48,shards_per_head=16,shard_shape=[32,32],dtype='FP32',
                                   bytes=36864*4096,placement='First2304 complete4x4 head blocks within up-worker regions, ordered by region then4-row block then4-column block'),
                    norm_weights=self.norms,norm_gain_bytes_per_input_owner=129*256,
                    auxiliary=self.auxiliary,auxiliary_used_pages=self.used_aux_pages,auxiliary_capacity_pages=int(self.aux_prefix[-1]),
                    original_weight_addresses_complete=True,persistent_state_addresses_complete=True,
                    dynamic_intermediates_placed=False,non_mlp_compute_routes_admitted=False,compiled_all_role_sram_admitted=False,
                    full_model_executable=False,physical=False,
                    scope='Exact metadata residency for all original text tensors and short-context persistent state, anchored to all64 MLPs. This is not all-role code/stack or runtime admission. Embedding helper program qualification is separate; remaining operator communication and dynamic intermediates remain required.')
