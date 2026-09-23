"""Original final norm and complete vocabulary into bounded physical ROIs.

The admitted remote caller supplies OriginalRows and the unchanged packer and
independent decoder. Finish, verify, fsync and release one <=124MB owner row
before touching the next; the complete 2.57GB matrix is never resident.
"""
from pathlib import Path
import hashlib,json,math,os,time
from stage_contract import require,VOCABULARY,HIDDEN_WORDS
from head_host_plan import build


def geometry(region):
    require(region['kind']=='original_final_norm_and_full_vocabulary' and region['groups']==1940 and
            region['matrix_PEs']==104760 and region['packed_matrix_weight_bytes']==2574581760 and
            region['original_tensors']['lm_head.weight']['shape']==[VOCABULARY,HIDDEN_WORDS] and
            region['original_tensors']['model.language_model.norm.weight']['shape']==[HIDDEN_WORDS],
            'Exact two original final tensors and full vocabulary geometry')
    items=build(region)['uploads'];weights=[p for p in items if p['symbol']=='weights'];points={};files={}
    for item in weights:
        source=item['prepared'];name=source['file'];shape=(item['height'],item['width'],6144)
        require(name not in files and math.prod(shape)*4<=16<<20,'Unique bounded physical head ROI')
        files[name]=(shape,item);seen=0
        for yy in range(item['height']):
            for xx in range(item['width']):
                x,y=item['x']+xx,item['y']+yy;row=(y-4)//55;ordinal=(y-4)%55;group=(x-region['x']-1)*21+row
                require(0<=row<21 and 0<=ordinal<54 and 0<=group<1940 and (x,y) not in points,
                        'Independent actual final-head PE/group/ordinal ownership')
                require(row==source['owner_row'] and ordinal==source['ordinal_begin']+yy,'Writer ROI metadata matches independent fabric coordinate')
                points[x,y]=(name,yy,xx,group,ordinal);seen+=1
        require(seen*24576==source['native_bytes'],'Exact physical head ROI payload extent')
    require(len(points)==104760 and sum(math.prod(shape)*4 for shape,_ in files.values())==2574581760,
            'Complete original final-head matrix PE and native byte inventory')
    require(len(region['stripes'])==1940,'Every original output group')
    for group,stripe in enumerate(region['stripes']):
        require(stripe['group']==group and stripe['tensor']=='lm_head.weight' and stripe['participants']==54 and
                stripe['output_rows']==[group*128,(group+1)*128] and stripe['valid_columns_last']==32 and
                (stripe['x'],stripe['y'])==(region['x']+1+group//21,4+55*(group%21)),
                'Original group row order and complete padded input columns')
    return items,files,points


def prepare(destination,region,reader,pack_batch,validate_packed_batch,check,progress):
    import numpy as np
    started=time.monotonic();destination=Path(destination);items,shapes,physical=geometry(region)
    require(not destination.exists(),'Fresh exact full-head preparation; no implicit overwrite or retry');destination.mkdir()
    records={};checks=[];total=0
    def close(value):value._mmap.close()
    def batch(group):return dict(file=f'original-head-group-{group:04}',tensor='lm_head.weight',shape=[VOCABULARY,HIDDEN_WORDS],
                                 width=54,height=1,bytes=54*24576,rows=[dict(group=group)])
    def seal(name,shape,dtype):
        nonlocal total
        check();path=destination/name
        with path.open('rb') as stream:os.fsync(stream.fileno())
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        require(value.shape==tuple(shape) and value.dtype==np.dtype(dtype) and value.flags.c_contiguous and value.nbytes<=16<<20,
                'Exact bounded prepared native head array')
        native=hashlib.sha256(memoryview(value).cast('B')).hexdigest();close(value);path.chmod(0o444)
        digest=hashlib.sha256()
        with path.open('rb',buffering=0) as stream:
            for block in iter(lambda:stream.read(1<<20),b''):
                digest.update(block);os.posix_fadvise(stream.fileno(),stream.tell()-len(block),len(block),os.POSIX_FADV_DONTNEED)
        size=path.stat().st_size;total+=size
        require(name not in records and len(records)<512 and size<=(16<<20)+4096 and total<3<<30,'Complete finite original head storage')
        records[name]=dict(bytes=size,sha256=digest.hexdigest(),native_sha256=native,shape=list(shape),dtype=np.dtype(dtype).str)
        progress('sealed',name,len(records),total)
    # All dirty pages are confined to one physical owner row. Sealing drops
    # its clean file-cache pages before proceeding to the next owner row.
    for owner_row in range(21):
        row_files={name:(shape,item) for name,(shape,item) in shapes.items() if item['prepared']['owner_row']==owner_row}
        require(sum(math.prod(shape)*4 for shape,_ in row_files.values())<=124<<20,'Bounded dirty head owner-row working set')
        for name,(shape,item) in row_files.items():
            check();mapped=np.lib.format.open_memmap(destination/name,mode='w+',dtype='<u4',shape=shape);close(mapped)
        stripes=[stripe for stripe in region['stripes'] if stripe['group']%21==owner_row]
        for stripe in stripes:
            group=stripe['group'];check();packed=pack_batch(batch(group),reader.read_rows)
            # The writer scatters the declared logical stripe into host ROIs.
            for name,(shape,item) in row_files.items():
                column=stripe['x']-item['x'];begin=item['y']-stripe['y'];height=item['height']
                require(0<=column<item['width'] and 0<=begin<begin+height<=54,'Declared original stripe intersects its complete owner-row ROI')
                mapped=np.load(destination/name,mmap_mode='r+',allow_pickle=False)
                mapped[:,column]=packed[0,begin:begin+height];close(mapped)
            progress('packed',str(group),len(records),total)
        for stripe in stripes:
            group=stripe['group'];check();gathered=np.empty((1,54,6144),dtype='<u4');selections={}
            # Independently recompute fabric coordinates from original output
            # group and input ordinal; never consume the writer's offsets.
            for ordinal in range(54):
                x,y=region['x']+1+group//21,4+55*(group%21)+ordinal
                name,yy,xx,got_group,got_ordinal=physical[x,y]
                require((got_group,got_ordinal)==(group,ordinal),'Independent original head coordinate reconstruction')
                selections.setdefault(name,[]).append((ordinal,yy,xx))
            for name,points in selections.items():
                mapped=np.load(destination/name,mmap_mode='r',allow_pickle=False)
                for ordinal,yy,xx in points:gathered[0,ordinal]=mapped[yy,xx]
                close(mapped)
            checks.append(dict(group=group,**validate_packed_batch(batch(group),gathered,reader.read_rows)))
            progress('independent_original_decode',str(group),len(records),total)
        for name,(shape,item) in row_files.items():seal(name,shape,'<u4')
    for item in items:
        if item['symbol']=='weights':continue
        source=item['prepared'];name=source['file']
        if item['symbol']=='descriptor':
            value=np.zeros((item['height'],item['width'],6),dtype='<u2')
            for ordinal in range(54):
                for column in range(item['width']):
                    x,y=item['x']+column,item['y']+ordinal;group=(x-region['x']-1)*21+(y-4)//55
                    require(0<=group<1940 and (y-4)%55==ordinal,'Original descriptor physical owner')
                    value[ordinal,column]=[0,group,0,0,128,32 if ordinal==53 else 96]
        else:
            require(item['symbol']=='head_norm_weights' and source['original_tensor']=='model.language_model.norm.weight',
                    'Only the unchanged original final norm support tensor')
            value=reader.read_tensor('model.language_model.norm.weight')
            require(value.shape==(5120,) and np.array_equal(value,reader.read_tensor('model.language_model.norm.weight')),
                    'All original final norm BF16 bits independently re-read')
        with (destination/name).open('xb') as stream:np.save(stream,value,allow_pickle=False);stream.flush();os.fsync(stream.fileno())
        seal(name,value.shape,value.dtype)
    require(set(records)=={item['prepared']['file'] for item in items} and len(checks)==1940,'Every original head execution file and all output groups')
    original=reader.finish()
    require(original['original_range_bytes']==2*region['original_weight_bytes'],'Every original final norm/matrix bit independently re-read once')
    manifest=dict(kind='direct_original_final_head_rows_to_ROIs',source_namespace='original-final-head',layer=64,family='head',
                  application=[95,1160],files=records,original_source=original,matrix_checks=checks,all_original_bits_checked=True,
                  original_geometry_checked=True,original_tensors=region['original_tensors'],original_weight_bytes=region['original_weight_bytes'],
                  packed_matrix_bytes=region['packed_matrix_weight_bytes'],array_file_bytes=total,reference_input_files=0,
                  SDK_invocations=0,model_forward=False,whole_shard_rehash=False,seconds=time.monotonic()-started)
    raw=(json.dumps(manifest,separators=(',',':'))+'\n').encode();require(len(raw)<=2<<20,'Bounded complete original-head preparation manifest')
    path=destination/'prepared.json'
    with path.open('xb') as stream:stream.write(raw);stream.flush();os.fsync(stream.fileno())
    path.chmod(0o444);fd=os.open(destination,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(files=len(records),array_file_bytes=total,original_tensors=2,original_range_bytes=reader.read_bytes,
                manifest=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),seconds=time.monotonic()-started)
