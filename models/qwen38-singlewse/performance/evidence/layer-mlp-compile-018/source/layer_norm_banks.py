"""Local recurrent-state page relocation for the complete residual/RMS graph.

Matrix tiles stay at their original PEs. Only explicitly identified128-byte
request-state pages leave the tail of the80 norm/quantizer banks, moving two
rows north into retained mixer banks. The map is required by future mixer state
lowering; the original schedule alone is not a valid state-address source for
this candidate. This does not change state capacity or discard any model data.
"""
from spatial.layer_schedule import auxiliary_owner


def lower_norm_banks(stage, profiles, enabled):
    if not enabled:return []
    region=next(r for r in stage['regions'] if r['role']=='mix')
    owners={}
    for item in region['auxiliary']:
        for page in range(item['page_start'],item['page_start']+item['pages']):
            owner=auxiliary_owner(region,page)
            owners[(*owner['pe'],owner['byte_offset'])]=(item,page)
    by_pe={tuple(p['pe']):p for p in profiles};remaps=[]
    for source in profiles:
        pages={'layer_norm_bridge.csl':12,'layer_norm_sender.csl':8}.get(source['source'],0)
        if not pages:continue
        x,y=source['pe'];destination=by_pe[(x,y-2)]
        if destination['source']!='layer_mlp_standby.csl':raise ValueError('State destination must have an independent resident bank')
        end=source['parameters']['bank_words']*4;first=end-pages*128
        target=destination['parameters']['bank_words']*4
        if first<0 or target+pages*128>40000:raise ValueError('Candidate bank payload budget')
        for i in range(pages):
            item,page=owners.get((x,y,first+i*128),(None,None))
            if not item or item['kind']!='request_state':raise ValueError('Norm relocation may not move a matrix, gain or activation')
            remaps.append(dict(tensor=item['id'],page=page,tensor_byte_offset=(page-item['page_start'])*128,
                               source=source['pe'],source_byte_offset=first+i*128,
                               destination=destination['pe'],destination_byte_offset=target+i*128,bytes=128))
        source['parameters']['bank_words']=first//4
        destination['parameters']['bank_words']=(target+pages*128)//4
    if len(remaps)!=800 or len({(r['tensor'],r['tensor_byte_offset']) for r in remaps})!=800:
        raise ValueError('Expected800 unique original request-state pages')
    return remaps
