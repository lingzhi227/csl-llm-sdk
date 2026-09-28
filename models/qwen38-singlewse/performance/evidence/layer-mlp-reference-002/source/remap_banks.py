"""Independently prove complete byte conservation when moving state-bank tails."""


def bank_segments(original, profiles, remaps):
    old={tuple(r['pe']):r for r in original};new={};cursor=0;segments=[]
    payload={tuple(pe):p['parameters'].get('bank_words',0) for p in profiles for pe in p['pes']}
    if set(old)!={pe for pe,n in payload.items() if n}:raise ValueError('Original resident coordinates changed')
    for pe,words in sorted(payload.items(),key=lambda item:(item[0][1],item[0][0])):
        if not words:continue
        new[pe]=dict(pe=list(pe),words=words,offset=cursor);cursor+=words
    def add(src,dst,length):
        if length<=0:raise ValueError('Empty transfer')
        segments.append(dict(source_word=src,destination_word=dst,words=length))
    for pe,dest in new.items():
        src=old[pe];add(src['offset'],dest['offset'],min(src['words'],dest['words']))
    for page in remaps:
        a=tuple(page['source']);b=tuple(page['destination']);size=page['bytes']//4
        first=page['source_byte_offset']//4;target=page['destination_byte_offset']//4
        if page['bytes']!=128 or page['source_byte_offset']%4 or page['destination_byte_offset']%4:
            raise ValueError('State page alignment')
        if not new[a]['words']<=first<first+size<=old[a]['words']:raise ValueError('Only a removed source tail may move')
        if not old[b]['words']<=target<target+size<=new[b]['words']:raise ValueError('Only a new destination tail may receive')
        add(old[a]['offset']+first,new[b]['offset']+target,size)
    total=sum(r['words'] for r in original)
    if cursor!=total:raise ValueError('Resident capacity changed')
    for key in ['source_word','destination_word']:
        end=0
        for item in sorted(segments,key=lambda s:s[key]):
            if item[key]!=end:raise ValueError('Missing or duplicate original resident word')
            end+=item['words']
        if end!=total:raise ValueError('Incomplete original resident storage')
    return dict(records=list(new.values()),total_words=total,segments=segments)
