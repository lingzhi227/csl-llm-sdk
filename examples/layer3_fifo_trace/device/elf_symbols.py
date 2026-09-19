"""Read standard ELF symbol addresses/sizes without vendor decoder assumptions."""
import struct


def symbols(data):
    if data[:4]!=b'\x7fELF' or data[5] not in (1,2):raise ValueError('ELF identification')
    end='<' if data[5]==1 else '>'
    if data[4]==1:
        offset=struct.unpack_from(end+'I',data,32)[0]
        stride,count=struct.unpack_from(end+'HH',data,46);section_fmt=end+'IIIIIIIIII';symbol_fmt=end+'IIIBBH'
    elif data[4]==2:
        offset=struct.unpack_from(end+'Q',data,40)[0]
        stride,count=struct.unpack_from(end+'HH',data,58);section_fmt=end+'IIQQQQIIQQ';symbol_fmt=end+'IBBHQQ'
    else:raise ValueError('ELF class')
    if not 0<count<=256 or stride<struct.calcsize(section_fmt) or offset+count*stride>len(data):raise ValueError('ELF section extent')
    sections=[struct.unpack_from(section_fmt,data,offset+i*stride) for i in range(count)];answer=[]
    for section in sections:
        if section[1]!=2:continue
        if section[6]>=count or section[9]<struct.calcsize(symbol_fmt) or section[5]%section[9]:raise ValueError('Symbol table shape')
        strings=sections[section[6]]
        if strings[4]+strings[5]>len(data) or section[4]+section[5]>len(data):raise ValueError('Symbol table extent')
        names=data[strings[4]:strings[4]+strings[5]]
        for at in range(section[4],section[4]+section[5],section[9]):
            fields=struct.unpack_from(symbol_fmt,data,at)
            if data[4]==1:name,address,size,info,other,index=fields
            else:name,info,other,index,address,size=fields
            if name>=len(names):raise ValueError('Symbol string offset')
            value=names[name:].split(b'\0',1)[0].decode('ascii')
            answer.append(dict(name=value,address=address,bytes=size,section=index,type=info&15))
    return answer
