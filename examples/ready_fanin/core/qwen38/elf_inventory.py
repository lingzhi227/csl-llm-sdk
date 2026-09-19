"""Small ELF section inventory. Allocated sections are not a runtime stack peak."""
import struct


def admit_wse3_sram(footprint, stack_allowance=4096, ceiling=49152):
    """48 KiB application SRAM; high device configuration addresses add no capacity."""
    if type(stack_allowance) is not int or stack_allowance < 4096:
        raise ValueError('At least the declared 4 KiB stack allowance is required')
    if type(ceiling) is not int or not 0 < ceiling <= 49152:
        raise ValueError('Application SRAM ceiling cannot exceed 48 KiB')
    config_names={'.entry_ival','.fpcw','.fscale','.blocked_ival','.active_ival','.blocked_ut_ival',
                  '.fabric_routes','.fabric_switches','.ce_in_q','.ce_out_q','.prng_state','.filters',
                  '.user_cfg_0','.user_cfg_1','.user_cfg_2','.user_cfg_3'}
    low=[s for s in footprint['allocated_sections'] if not (s['name'] in config_names and s['address']>=63104)]
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
