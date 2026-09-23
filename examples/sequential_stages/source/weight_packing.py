"""Unchanged original-BF16 packing and independent bit validation; remote use only."""
import hashlib
TILE_BYTES=24576
MAX_BATCH_BYTES=16<<20

def pack_batch(batch,read_rows):
    """At most128 original rows plus one16MiB transfer file in memory."""
    import numpy as np
    nr,nc=batch['shape'];height,width=batch['height'],batch['width']
    assert batch['bytes']==height*width*TILE_BYTES<=MAX_BATCH_BYTES
    packed=np.zeros((height,width,6144),dtype='<u4');halves=packed.view('<u2').reshape(height,width,96,128)
    for y,s in enumerate(batch['rows']):
        first=128*s['group'];valid_rows=min(128,nr-first);original=read_rows(batch['tensor'],first,valid_rows)
        assert original.dtype==np.dtype('<u2') and original.shape==(valid_rows,nc)
        for ordinal in range(width):
            begin=ordinal*96;end=min(begin+96,nc)
            halves[y,ordinal,:end-begin,:valid_rows]=original[:,begin:end].T
    return packed

def validate_packed_batch(batch,packed,read_rows):
    """Compare every stored bit and every padded halfword with original rows.

    This deliberately reads original rows independently of pack_batch's copy
    operation. The caller must pin the same immutable original source before
    and after both passes. It is a remote preparation check, never a Mac test.
    """
    import numpy as np
    nr,nc=batch['shape'];height,width=batch['height'],batch['width']
    assert packed.dtype==np.dtype('<u4') and packed.shape==(height,width,6144)
    assert packed.nbytes==batch['bytes']<=MAX_BATCH_BYTES
    original_elements=0;padded_elements=0;row_pins=[]
    for i,row in enumerate(batch['rows']):
        first=128*row['group'];valid_rows=min(128,nr-first)
        original=read_rows(batch['tensor'],first,valid_rows)
        assert original.dtype==np.dtype('<u2') and original.shape==(valid_rows,nc)
        row_pins.append(dict(group=row['group'],bytes=original.nbytes,
            sha256=hashlib.sha256(original.tobytes(order='C')).hexdigest()))
        for ordinal in range(width):
            # Decode from individual packed32-bit words, independently of the
            # writer's uint16 view/transpose. For each input column, adjacent
            # output rows occupy the low then high halfword of a word.
            words=packed[i,ordinal].reshape(96,64)
            decoded=np.empty((128,96),dtype='<u2')
            decoded[0::2,:]=(words&0xffff).T
            decoded[1::2,:]=(words>>16).T
            begin=ordinal*96;valid_columns=max(0,min(96,nc-begin))
            assert np.array_equal(decoded[:valid_rows,:valid_columns],
                original[:,begin:begin+valid_columns]),(batch['file'],row['group'],ordinal)
            assert np.all(decoded[valid_rows:,:]==0)
            assert np.all(decoded[:valid_rows,valid_columns:]==0)
            original_elements+=valid_rows*valid_columns
            padded_elements+=128*96-valid_rows*valid_columns
    assert 2*(original_elements+padded_elements)==packed.nbytes
    return dict(passed=True,original_BF16_elements=original_elements,
        zero_padding_elements=padded_elements,original_row_pins=row_pins,
        verification='Independent packed-u32 low/high halfword decode against original BF16 rows')
