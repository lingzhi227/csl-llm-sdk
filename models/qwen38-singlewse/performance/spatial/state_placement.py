"""Resolve original logical GDN state through admitted local page relocation.

The numerical tensor coordinates remain request/head/key/value. This lowering
layer owns the separate physical PE/bank location; consumers must not reconstruct
the pre-fusion address after SRAM placement has relocated a page. This module is
address admission, not an executable or numerically qualified mixer.
"""
from spatial.layer_schedule import auxiliary_owner,gdn_state_owner,rank_xy,xy_rank


class StatePlacement:
    def __init__(self,region,profiles,remaps):
        self.region=region;self.moves={};self.original_ends={};self.ends={}
        by_pe={tuple(p['pe']):p for p in profiles}
        if len(by_pe)!=len(profiles):raise ValueError('Duplicate physical bank profile')
        for c in region['banks']['classes']:
            for rank in range(c['rank_start'],c['rank_end']):
                pe=tuple(rank_xy(region,rank))
                prefix=c['page_prefix']+(rank-c['rank_start'])*c['auxiliary_pages']
                live=max(0,min(c['auxiliary_pages'],region['auxiliary_pages']-prefix))
                self.original_ends[pe]=c['fp8_slots']*260+c['bf16_slots']*256+128*live
                self.ends[pe]=by_pe[pe]['parameters'].get('bank_words',0)*4
        source_pages={};destination_pages={};destination_keys=set()
        tensors={a['id']:a for a in region['auxiliary']}
        for m in remaps:
            tensor=tensors.get(m['tensor'])
            if tensor is None or tensor['kind']!='request_state' or m['bytes']!=128:
                raise ValueError('Only original128-byte request state pages may move')
            relative=m['tensor_byte_offset']
            if relative<0 or relative%128 or relative>=tensor['pages']*128:
                raise ValueError('Logical state page outside original tensor')
            page=tensor['page_start']+relative//128
            owner=auxiliary_owner(region,page)
            if m['page']!=page or (m['source'],m['source_byte_offset'])!=(owner['pe'],owner['byte_offset']):
                raise ValueError('State page does not match its original tensor owner')
            key=(m['tensor'],relative);source=tuple(m['source']);destination=tuple(m['destination'])
            offset=m['destination_byte_offset'];destination_key=(*destination,offset)
            if key in self.moves or destination_key in destination_keys:
                raise ValueError('Duplicated logical or physical state page')
            if destination not in self.ends or not self.original_ends[destination]<=offset<offset+128<=self.ends[destination]:
                raise ValueError('Relocated state overlaps original bank or exceeds new bank')
            if not self.ends[source]<=owner['byte_offset']<owner['byte_offset']+128<=self.original_ends[source]:
                raise ValueError('Relocated source is not in the removed bank tail')
            source_pages.setdefault(source,[]).append(owner['byte_offset'])
            destination_pages.setdefault(destination,[]).append(offset)
            destination_keys.add(destination_key)
            self.moves[key]=dict(pe=list(destination),rank=xy_rank(region,destination),byte_offset=offset)
        # Prove every shortened/extended bank tail is exactly accounted for.
        # A missing page cannot be hidden by omitting it from the relocation map.
        for pe,old in self.original_ends.items():
            new=self.ends[pe]
            if (new-old)%128:raise ValueError('State relocation changes a non-page bank extent')
            expected_source=list(range(new,old,128)) if new<old else []
            expected_destination=list(range(old,new,128)) if new>old else []
            if sorted(source_pages.get(pe,[]))!=expected_source or sorted(destination_pages.get(pe,[]))!=expected_destination:
                raise ValueError('Missing or overlapping state pages in a changed bank tail')

    def gdn_owner(self,request,head,key_block,value_block):
        owner=gdn_state_owner(self.region,request,head,key_block,value_block)
        relative=((request*48+head)*32*16+key_block*16+value_block)*128
        move=self.moves.get((owner['state'],relative))
        if move is not None:owner.update(move)
        if owner['byte_offset']+owner['bytes']>self.ends[tuple(owner['pe'])]:
            raise ValueError('Resolved original state exceeds admitted bank')
        return owner
