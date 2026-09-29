"""Lossless PE-local FP8 scale sharing for complete original mixer matrices.

A2x128 FP8 tile has256 original code bytes. Its FP32 weight scale belongs to
an original128x128 block, so consecutive local row pairs may share exactly the
same scale word. Every alias is checked against source words during migration.
No weight or activation is requantized. BF16 A/B rows remain unchanged.
"""
from collections import defaultdict
from spatial.layer_schedule import local_rows, rank_xy


class CompactMixerPlacement:
    def __init__(self, region):
        if region['rect']!=[63,67,45,79]:raise ValueError('Pinned complete mixer rectangle')
        self.region=region;self.records={};self.groups=[]
        for m in region['matrices']:
            if m['dtype']=='F8_E4M3' and 'group_partitions' not in m:
                raise ValueError('Local shared scales require contiguous output ownership')
            if 'group_partitions' in m:
                index=[];cursor=0
                for i,g in enumerate(m['group_partitions']):
                    if (g['output_start'],g['rank_start'],g['workers'])!=(cursor,i*m['k_blocks'],m['k_blocks']):
                        raise ValueError('Complete original K group or row interval changed')
                    index.extend([i]*g['output_count']);cursor+=g['output_count']
                if cursor!=m['output_tiles']:raise ValueError('Missing original rows')
                self.groups.append(index)
            else:self.groups.append(None)
        for rank in range(3555):
            counts=[local_rows(m,rank) for m in region['matrices']]
            fp=sum(n for m,n in zip(region['matrices'],counts) if m['dtype']=='F8_E4M3')
            data_words=sum(counts)*64;cursor=data_words;setup=[];scales=[]
            for mi,(m,count) in enumerate(zip(region['matrices'],counts)):
                group,key=divmod(rank,m['k_blocks']);rows=m['tile_shape'][0]
                if 'group_partitions' in m:
                    first=m['group_partitions'][group]['output_start'] if group<m['groups'] else 0;step=1
                else:first=(group-m['group_rotation'])%m['groups'];step=m['groups']
                bf=m['dtype']=='BF16'
                preceding=sum(counts[j] for j in range(mi) if (region['matrices'][j]['dtype']=='BF16')==bf)
                base=(preceding+(fp if bf else 0))*64 if count else 0
                setup.extend([base,count,first*rows,step*rows,key,m['k_blocks'],int(bf),rows])
                scale_count=((first+count-1)//64-first//64+1) if count and not bf else 0
                scales.append(dict(base_words=cursor if scale_count else 0,first_block=first//64,
                                   count=scale_count,matrix=mi))
                cursor+=scale_count
            setup.extend(s['base_words'] for s in scales);setup.append(0)
            if len(setup)!=46 or any(not 0<=x<65536 for x in setup):raise ValueError('Compact descriptor extent')
            if (cursor-data_words)*4>64:raise ValueError('Shared scales exceed the solver reservation')
            self.records[rank]=dict(pe=rank_xy(region,rank),bytes=cursor*4,data_words=data_words,
                                    scales=scales,setup=setup)

    def tile(self, matrix, tile):
        m=self.region['matrices'][matrix]
        if not 0<=tile<m['tiles']:raise ValueError('Original tile extent')
        row,key=divmod(tile,m['k_blocks'])
        if self.groups[matrix] is not None:
            group=self.groups[matrix][row];iteration=row-m['group_partitions'][group]['output_start']
        else:group=(row+m['group_rotation'])%m['groups'];iteration=row//m['groups']
        rank=group*m['k_blocks']+key;record=self.records[rank];base=record['setup'][matrix*8]
        result=dict(pe=record['pe'],byte_offset=4*(base+iteration*64),bytes=256,rank=rank)
        if m['dtype']=='F8_E4M3':
            s=record['scales'][matrix];index=row//64-s['first_block']
            if not 0<=index<s['count']:raise ValueError('Scale block outside original local rows')
            result['scale']=dict(pe=record['pe'],byte_offset=4*(s['base_words']+index),bytes=4)
        return result

    def metadata(self):
        return dict(schema='compact-original-mixer-scales-v1',rect=self.region['rect'],
                    records=list(self.records.values()),setup_words=46,fp8_data_words=64,
                    scale_semantics='Original FP32 scale word per128x128 block, shared only within one PE and one matrix',
                    bytes=sum(r['bytes'] for r in self.records.values()),source_alias_values_verified=False,
                    compiled=False,physical=False,neural_execution=False)


def compact_csl(files):
    def one(raw,before,after):
        if raw.count(before)!=1:raise ValueError('Original mixed projection anchor changed')
        return raw.replace(before,after)
    native=files['mixer_native.csl']
    native=one(native,b'offset:u16,projection:u16,result:[*]f32',b'offset:u16,projection:u16,scale_word:u16,result:[*]f32')
    native=one(native,b'bank[offset+64]',b'bank[scale_word]')
    native=one(native,b'  unpacker.decode(',b'  @assert(scale_word<bank_words);unpacker.decode(')
    files['mixer_native.csl']=native
    source=files['mixer_projection.csl']
    source=one(source,b'var setup=@zeros([40]u16)',b'var setup=@zeros([46]u16)')
    source=one(source,b'const words:u16=if(use_bf16) 64 else 65;',b'const words:u16=64;')
    source=one(source,b'native.compute(bank,offset,matrix,&partial);',
               b'const scale_word:u16=setup[40+matrix]+((first+computed*stride)>>7)-(first>>7);native.compute(bank,offset,matrix,scale_word,&partial);')
    files['mixer_projection.csl']=source
    for name in ('device_mixer_layer_mlp_standby.csl','device_mixer_layer_norm_sender.csl','device_mixer_layer_norm_bridge.csl'):
        files[name]=one(files[name],b'half=msetup;extent=40;',b'half=msetup;extent=46;')
