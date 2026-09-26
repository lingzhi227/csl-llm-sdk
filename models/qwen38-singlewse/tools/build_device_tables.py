"""Compact all matrix coordinates into a bounded controller descriptor table."""
import argparse,collections,hashlib,json,math
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def locate(width,index,bf16_rows=144):
    count40=15530 if bf16_rows==144 else 15628
    last_row=1149 if bf16_rows==144 else 1155
    tail_start=15515 if bf16_rows==144 else 15623
    if width==136:
        assert 0<=index<1216;return index%4*136,index//4
    if width==48:
        assert 0<=index<1216
        return (0,last_row) if index==1215 else (index%3*48,304+index//3)
    assert width==40 and 0<=index<count40
    if index<1520:return 544+index%5*40,index//5
    if index<7595:
        n=index-1520;return 144+n%15*40,304+n//15
    if index<tail_start:
        n=index-7595;return n%18*40,709+n//18
    return 48+(index-tail_start)*40,last_row

def build(bf16_rows=144):
    suffix='' if bf16_rows==144 else f'-bf16-{bf16_rows}'
    placement_path=ROOT/'configs'/f'placement-resident-96{suffix}.json'
    placement=json.loads(placement_path.read_text())
    graph=json.loads((ROOT/'configs/model-graph.json').read_text())
    assignments=placement['assignments']
    counts=[(40,15530 if bf16_rows==144 else 15628),(48,1216),(136,1216)]
    reverse={w:{locate(w,i,bf16_rows):i for i in range(n)} for w,n in counts}
    order=[n['weights'][0] for n in graph['nodes'] if n['op'] in ['embedding_lookup','fp8_projection','bf16_projection']]
    assert len(order)==len(set(order))==498
    assert set(order)=={n for n,a in assignments.items() if 'strips' in a}
    table=[];matrix_ids={};strips=0;seen=collections.defaultdict(set)
    for matrix_id,name in enumerate(order):
        a=assignments[name];rows,cols=a['shape'];width=cols//128;tile_rows=a['tile'][0]
        assert cols%128==0
        positions=[reverse[width][(x,y)] for _,x,y in a['strips']]
        base=positions[0]
        assert positions==list(range(base,base+len(positions)))
        assert len(positions)==math.ceil(rows/tile_rows)
        for ordinal,x,y in a['strips']:
            assert locate(width,base+ordinal,bf16_rows)==(x,y)
            assert base+ordinal not in seen[width];seen[width].add(base+ordinal)
        kind=1 if a['role']=='fp8_matrix' else 2
        if name=='model.language_model.embed_tokens.weight':kind=3
        if name=='lm_head.weight':kind=4
        # Eight u16 words; rows are split because the vocabulary exceeds65535.
        record=[kind,rows&65535,rows>>16,width,base,tile_rows,len(positions),0]
        assert all(0<=x<=65535 for x in record)
        table.append(record);matrix_ids[name]=matrix_id;strips+=len(positions)
    assert strips==sum(n for w,n in counts) and sum(len(v) for v in seen.values())==strips
    assert matrix_ids['model.language_model.embed_tokens.weight']==0 and matrix_ids['lm_head.weight']==497
    # Every four layers contain3 GDN layers with8 matrices and1 full-attention
    # layer with7 matrices. Controller sequencing therefore needs no1172-row ROM.
    for layer in range(64):
        base=1+(layer//4)*31+(layer%4)*8
        tensor_suffix='self_attn.q_proj.weight' if layer%4==3 else 'linear_attn.in_proj_qkv.weight'
        assert matrix_ids[f'model.language_model.layers.{layer}.{tensor_suffix}']==base
    helper_path=ROOT/'csl'/f'strip_coordinates{suffix}.csl'
    if bf16_rows==140:
        helper=(ROOT/'csl/strip_coordinates.csl').read_text().replace('15530','15628').replace('15515','15623').replace('1149','1155')
        helper_path.write_text(helper)
    return dict(status=f'All{strips} strips matched; compact control data only, not full CSL execution',
        model=placement['model'],revision=placement['revision'],matrix_count=498,strip_count=strips,
        words_per_matrix=8,storage_bytes=len(table)*16,format=['kind','rows_low16','rows_high16','k_blocks','strip_base','tile_rows','row_tiles','reserved'],
        kind=dict(fp8=1,bf16=2,embedding=3,head=4),table=table,matrix_ids=matrix_ids,
        bf16_tile_rows=bf16_rows,placement_sha256=hashlib.sha256(placement_path.read_bytes()).hexdigest(),
        coordinate_helper_sha256=hashlib.sha256(helper_path.read_bytes()).hexdigest())

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bf16-rows',type=int,choices=[144,140],default=144);args=p.parse_args()
    result=build(args.bf16_rows)
    filename='device-matrices.json' if args.bf16_rows==144 else f'device-matrices-bf16-{args.bf16_rows}.json'
    (ROOT/'configs'/filename).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k not in ['table','matrix_ids']},indent=2))
