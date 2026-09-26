"""Small ELF section inventory. Allocated sections are not a runtime stack peak."""
import struct
import json


def placement(data):
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
    if fabric!=(762,1172):raise ValueError('Physical fabric mismatch')
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


def admit_wse3_sram(footprint, stack_allowance=4096, ceiling=49152):
    """48 KiB application SRAM; high device configuration addresses add no capacity."""
    if type(stack_allowance) is not int or stack_allowance < 4096:
        raise ValueError('At least the declared 4 KiB stack allowance is required')
    if type(ceiling) is not int or not 0 < ceiling <= 49152:
        raise ValueError('Application SRAM ceiling cannot exceed 48 KiB')
    config_names={'.entry_ival','.fpcw','.fscale','.blocked_ival','.active_ival','.blocked_ut_ival',
                  '.fabric_routes','.fabric_switches','.ce_in_q','.ce_out_q','.prng_state','.filters',
                  '.user_cfg_0','.user_cfg_1','.user_cfg_2','.user_cfg_3'}
    # FIFO8 / XDSR2 emitted by SDK 2.10.1. Its cs-readelf --msize report
    # independently excludes exactly these initializer sections from main SRAM.
    # Keep this narrow: unknown names, addresses, sizes or section types count.
    fifo_initializers={'.s1ds':(64832,8),'.dds':(64960,8),'.xds':(64128,24)}
    def is_configuration(section):
        if section['name'] in config_names and section['address']>=63104:
            return True
        return (section.get('type')==1 and
                fifo_initializers.get(section['name'])==(section['address'],section['bytes']))
    low=[s for s in footprint['allocated_sections'] if not is_configuration(s)]
    if not low: raise ValueError('No application SRAM sections')
    end=max(s['address']+s['bytes'] for s in low)
    return {'low_section_end':end,'low_section_bytes':sum(s['bytes'] for s in low),
            'static_allocated_section_bytes':footprint['static_allocated_section_bytes'],
            'stack_allowance_bytes':stack_allowance,'ordinary_address_ceiling':ceiling,
            'passed':end+stack_allowance<=ceiling,
            'scope':'48 KiB application SRAM plus declared stack allowance gate; not measured dynamic stack peak'}


def inventory(data):
    if data[:4]!=b'\x7fELF' or data[5] not in (1,2):
        raise ValueError('Unsupported ELF header')
    endian='<' if data[5]==1 else '>'
    if data[4]==1:
        shoff=struct.unpack_from(endian+'I',data,32)[0]
        size,count,names_index=struct.unpack_from(endian+'HHH',data,46)
        fmt=endian+'IIIIIIIIII'
    elif data[4]==2:
        shoff=struct.unpack_from(endian+'Q',data,40)[0]
        size,count,names_index=struct.unpack_from(endian+'HHH',data,58)
        fmt=endian+'IIQQQQIIQQ'
    else:
        raise ValueError('Unsupported ELF class')
    if count==0 or names_index>=count or size<struct.calcsize(fmt) or shoff+size*count>len(data):
        raise ValueError('Invalid ELF section table')
    entries=[struct.unpack_from(fmt,data,shoff+i*size) for i in range(count)]
    names=entries[names_index];strings=data[names[4]:names[4]+names[5]]
    allocated=[]
    for section in entries:
        name_at,kind,flags,address,offset,length=section[:6]
        if flags & 2 and length:
            name=strings[name_at:].split(b'\0',1)[0].decode('ascii')
            allocated.append({'name':name,'type':kind,'address':address,'bytes':length})
    return {'elf_class_bits':32 if data[4]==1 else 64,'allocated_sections':allocated,
            'static_allocated_section_bytes':sum(s['bytes'] for s in allocated),
            'scope':'ELF SHF_ALLOC sections; excludes unrepresented runtime stack/scratch demand'}
