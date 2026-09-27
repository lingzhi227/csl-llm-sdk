"""Cerebras ELF rectangle decoder adapted from preserved runtime/elf_inventory.py.

Only the fabric-note expectation is parameterized for bounded simulator fabrics.
Rectangle encoding and low-memory coverage checks are unchanged.
"""
import struct
import json


def placement(data, expected_fabric):
    """Cerebras ELF64 rectangle extension, checked against SDK ELFObject.

    p_paddr packs linear origin at bit40,width at bit20,height at bit0.
    Only complete low-memory PT_LOAD regions at virtual address0 establish
    application coverage. The SDK cs-readelf manual defines the extension;
    the exact packing must pass the separate installed-SDK qualification.
    """
    if data[:6]!=b'\x7fELF\x02\x01':raise ValueError('Expected SDK little-endian ELF64')
    shoff=struct.unpack_from('<Q',data,40)[0];size,count,ni=struct.unpack_from('<HHH',data,58)
    if size<64 or not 0<=ni<count or shoff+size*count>len(data):raise ValueError('Section table extent')
    entries=[struct.unpack_from('<IIQQQQIIQQ',data,shoff+i*size) for i in range(count)]
    strings=entries[ni];names=data[strings[4]:strings[4]+strings[5]];fabric=None;source=None
    for entry in entries:
        name=names[entry[0]:].split(b'\0',1)[0].decode('ascii')
        if name not in ['.note','.pe_debug_info']:continue
        raw=data[entry[4]:entry[4]+entry[5]]
        if name=='.note':
            ns,ds,kind=struct.unpack_from('<III',raw)
            if (ns,ds,kind)!=(9,8,1) or raw[12:21]!=b'Cerebras\0':raise ValueError('Cerebras fabric note profile')
            fabric=struct.unpack_from('<II',raw,24)
        else:source=json.loads(raw)['filename']
    if fabric!=tuple(expected_fabric):raise ValueError('Declared fabric mismatch')
    phoff=struct.unpack_from('<Q',data,32)[0];ps,pn=struct.unpack_from('<HH',data,54)
    if ps<56 or phoff+ps*pn>len(data):raise ValueError('Program table extent')
    regions=set();all_regions=set()
    for i in range(pn):
        kind,flags,offset,address,encoded,filesz,memsz,align=struct.unpack_from('<IIQQQQQQ',data,phoff+i*ps)
        if kind!=1:continue
        if offset+filesz>len(data):raise ValueError('Load segment file extent')
        linear=encoded>>40;width=(encoded>>20)&1048575;height=encoded&1048575
        x=linear%fabric[0];y=linear//fabric[0]
        if width==0 or height==0 or x+width>fabric[0] or y+height>fabric[1]:raise ValueError('Load rectangle bounds')
        rectangle=(x,y,width,height);all_regions.add(rectangle)
        if address==0:
            if not 0<memsz<=49152:raise ValueError('Complete low memory segment extent')
            regions.add(rectangle)
    if not regions:raise ValueError('No complete application memory region')
    return dict(fabric=list(fabric),source=source,rectangles=sorted(regions),all_load_rectangles=sorted(all_regions))

