"""Original-weight initialization stream for resident layer-native addresses.

Only checkpoint packing happens on the host. No neural computation, generated
operands or modified weight precision. The reader is runtime.weights.OriginalWeights.
One original output-row block and one scale row are resident at a time.
"""
import math
import struct


def matrix_tiles(weights,matrix,row_blocks=None):
    name=matrix['tensor'];m,k=matrix['shape'];rows,columns=matrix['tile_shape']
    fp8=matrix['dtype']=='F8_E4M3';scale_index=None;scale=None
    blocks=range(m//rows) if row_blocks is None else tuple(row_blocks)
    if row_blocks is not None and (len(blocks)>16 or tuple(sorted(set(blocks)))!=blocks or any(not 0<=b<m//rows for b in blocks)):
        raise ValueError('At most16 ordered distinct sample row blocks')
    for block in blocks:
        first=block*rows
        data=weights.rows(name,first,rows)
        if fp8 and first//128!=scale_index:
            scale_index=first//128;scale=weights.rows(matrix['scale_tensor'],scale_index,1)
        for column in range(0,k,columns):
            tile=(first//rows)*(k//columns)+column//columns
            # Explicit column-then-row native FMA order, independent of NumPy
            # reshape/transpose-based original fixture implementations.
            if fp8:
                packed=bytes(int(data[r,c]) for c in range(column,column+columns) for r in range(rows))
                if any((v&127)==127 for v in packed):raise ValueError('FP8 NaN in original tile')
                bits=int(scale[0,column//128])<<16
                value=struct.unpack('<f',struct.pack('<I',bits))[0]
                if not math.isfinite(value) or value<=0:raise ValueError('Invalid original scale')
                packed+=struct.pack('<I',bits)
            else:
                if (rows,columns)!=(1,128):raise ValueError('Current BF16 row ABI')
                packed=b''.join(struct.pack('<H',int(data[0,c])) for c in range(column,column+columns))
            if len(packed)!=(260 if fp8 else 256):raise ValueError('Native tile payload extent')
            yield tile,packed


def matrix_writes(weights,region,index,owner):
    for tile,payload in matrix_tiles(weights,region['matrices'][index]):
        destination=owner(region,index,tile)
        if destination['bytes']!=len(payload) or destination['byte_offset']%4:
            raise ValueError('Unaligned initialization record')
        yield destination,payload
