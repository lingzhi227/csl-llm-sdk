"""Original selected-layer bits directly into the two qualified ROI families.

No reference activations, SDK or model forward. Existing layer0..3 files are
reused separately. The bounded remote caller supplies pinned original headers,
prior whole-file identities, unchanged pack/decoder and configuration sources.
"""
from pathlib import Path
import hashlib,json,math,os,time


def require(ok,message):
    if not ok:raise ValueError(message)


def stamp(path):
    s=path.stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]


class OriginalRows:
    def __init__(self,cold,tensors,headers,shards,check,maximum_bytes):
        import numpy as np
        self.np=np;self.cold=Path(cold);self.tensors=tensors;self.shards=shards;self.check=check
        self.used=set();self.read_bytes=0;self.maximum_bytes=maximum_bytes;self.identities={}
        require(set(shards)=={t['shard'] for t in tensors.values()},'Only own original shards')
        for name,pin in shards.items():
            path=self.cold/name;require(Path(name).name==name and path.is_file() and not path.is_symlink(),'Exact original regular shard')
            before=stamp(path);require(before==pin['stamp'] and before[0]==2049 and before[2]==pin['bytes'],'Prior full-hash source identity unchanged')
            evidence=Path(pin['hash_receipt']['path']);raw=evidence.read_bytes()
            require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==pin['hash_receipt']['pin'],'Pinned accepted whole-shard hash receipt')
            receipt=json.loads(raw)
            require(receipt['status']=='complete' and receipt['source_identity']==before and receipt['sha256']==pin['sha256'],
                    'Complete full-byte digest and unchanged source stamp, without another whole-shard read')
            with path.open('rb',buffering=0) as f:
                prefix=f.read(8);require(len(prefix)==8,'Original header prefix');length=int.from_bytes(prefix,'little')
                require(length+8==pin['data_start'] and length<=1<<20,'Original bounded header extent')
                raw=f.read(length);require(len(raw)==length and json.loads(raw)==headers[name],'Actual original shard header matches pinned metadata')
            self.identities[name]=before
        for name,tensor in tensors.items():
            require(tensor['dtype']=='BF16' and headers[tensor['shard']][name]=={k:tensor[k] for k in ('dtype','shape','data_offsets')} and
                    math.prod(tensor['shape'])*2==tensor['data_offsets'][1]-tensor['data_offsets'][0],'Every original tensor extent/type/shape')
        self.unchanged()

    def unchanged(self):
        for name,before in self.identities.items():
            require(not (self.cold/name).is_symlink() and stamp(self.cold/name)==before,'Original source unchanged across both packing passes')

    def read(self,name,offset,shape):
        self.check();require(name in self.tensors,'Own original tensor namespace')
        tensor=self.tensors[name];size=math.prod(shape)*2
        require(0<size<=128*17408*2 and 0<=offset and offset+size<=tensor['data_offsets'][1]-tensor['data_offsets'][0],
                'Bounded original rows inside the exact tensor')
        require(self.read_bytes+size<=self.maximum_bytes,'Finite total original range reads')
        shard=tensor['shard'];path=self.cold/shard;self.unchanged()
        value=self.np.empty(shape,dtype='<u2');view=memoryview(value).cast('B')
        start=self.shards[shard]['data_start']+tensor['data_offsets'][0]+offset
        fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW)
        try:
            require(stamp(path)==self.identities[shard],'Stable original file before read')
            with os.fdopen(fd,'rb',buffering=0,closefd=False) as stream:
                stream.seek(start);copied=0
                while copied<len(view):
                    count=stream.readinto(view[copied:copied+(1<<20)]);require(count,'No original row truncation')
                    os.posix_fadvise(fd,start+copied,count,os.POSIX_FADV_DONTNEED);copied+=count
        finally:os.close(fd)
        self.unchanged();self.read_bytes+=size;self.used.add(name);return value

    def read_rows(self,name,first,count):
        shape=self.tensors[name]['shape'];require(len(shape)==2,'Original matrix')
        rows,columns=shape;require(0<=first<first+count<=rows and count<=128,'Original 128-row maximum')
        return self.read(name,first*columns*2,(count,columns))

    def read_tensor(self,name):
        shape=self.tensors[name]['shape'];require(math.prod(shape)*2<=1<<20,'Only small support tensor whole reads')
        return self.read(name,0,tuple(shape))

    def finish(self):
        self.unchanged();require(self.used==set(self.tensors),'All and only own original layer tensors consumed')
        return dict(tensors=sorted(self.used),original_range_bytes=self.read_bytes,whole_shard_rehash=False,
                    original_shards={n:dict(identity=s,sha256=self.shards[n]['sha256'],hash_receipt=self.shards[n]['hash_receipt']) for n,s in self.identities.items()})


def geometry(plan,host,tensors,family):
    linear=family=='linear';layer=plan['layer'];width=30 if linear else 29
    require(0<=layer<64 and plan['application']==[width,1160],'Qualified full original layer family geometry')
    expected={};scatter={};physical={};files={};slots={}
    for index,stripe in enumerate(plan['stripes']):
        name=stripe['tensor' if linear else 'name']
        require(name.startswith(f'model.language_model.layers.{layer}.') and name in tensors,'Own original stripe tensor')
        require(stripe['columns']==math.ceil(tensors[name]['shape'][1]/96),'Original column tile count')
        for ordinal in range(stripe['columns']):
            xy=(stripe['x'],stripe['y']+ordinal);require(xy not in expected,'Unique logical matrix PE')
            expected[xy]=(index,ordinal)
    grouped=host.get('prepared_weight_groups')
    if grouped:
        for group in grouped:
            require(group['file'] not in files,'Unique packed group file');files[group['file']]=(group['tiles'],6144)
    else:
        for item in host['matrix_uploads']:
            require(item['file'] not in files,'Unique packed ROI file');files[item['file']]=tuple(item['host_shape'])
    for name,shape in files.items():
        require(Path(name).name==name and not name.startswith('reference-') and 0<math.prod(shape)*4<=16<<20,'Bounded final original packed file')
        slots[name]=set()
    for item in host['matrix_uploads']:
        name=item['file'];base=item.get('file_flat_begin',0)
        require(item['count']==6144 and item['bits']==32 and item['host_shape']==[item['height'],item['width'],6144] and
                item['host_bytes']==item['height']*item['width']*24576<=16<<20,'Qualified exact host copy extent')
        for y in range(item['height']):
            for x in range(item['width']):
                xy=(item['x']+x,item['y']+y);slot=base+y*item['width']+x
                require(xy in expected and xy not in physical and slot not in slots[name] and slot<math.prod(files[name][:-1]),
                        'Every actual PE maps to one unique output tile')
                physical[xy]=(name,slot);slots[name].add(slot)
        seen=set()
        for segment in item['segments']:
            index=segment['logical_stripe'];require(0<=index<len(plan['stripes']),'Actual logical stripe index')
            for j in range(segment['tile_count']):
                dx,dy=segment['destination_x'],segment['destination_y']+j
                require(0<=dx<item['width'] and 0<=dy<item['height'] and (dx,dy) not in seen,'Unique complete writer segment')
                seen.add((dx,dy));xy=(item['x']+dx,item['y']+dy)
                require(expected[xy]==(index,segment['ordinal_begin']+j),'Writer segment equals independent actual PE/ordinal map')
            scatter.setdefault(index,[]).append((item,segment))
        require(len(seen)==item['width']*item['height'],'Every ROI word will be initialized')
    require(set(physical)==set(expected) and len(expected)==(31548 if linear else 30576),'Complete qualified original matrix population')
    for name,shape in files.items():require(slots[name]==set(range(math.prod(shape[:-1]))),'Complete final packed file tile coverage')
    return scatter,physical,files


def prepare(destination,plan,transport,host,tensors,reader,family,pack_batch,validate_packed_batch,configuration_arrays,frequency_bits,check,progress):
    import numpy as np
    started=time.monotonic();destination=Path(destination);linear=family=='linear'
    scatter,physical,shapes=geometry(plan,host,tensors,family)
    require(not destination.exists(),'Fresh original layer output, no implicit overwrite or retry');destination.mkdir()
    files={};checks=[];total=0
    def batch(index):
        stripe=plan['stripes'][index];name=stripe['tensor' if linear else 'name']
        return dict(file=f'logical-stripe-{index:03}',tensor=name,shape=tensors[name]['shape'],width=stripe['columns'],height=1,
                    bytes=stripe['columns']*24576,rows=[dict(group=stripe['group'])])
    def close(value):value._mmap.close()
    def seal(name,shape,dtype):
        nonlocal total
        check();path=destination/name
        with path.open('rb') as f:os.fsync(f.fileno())
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        require(value.flags.c_contiguous and value.dtype==np.dtype(dtype) and value.shape==tuple(shape) and value.nbytes<=16<<20,'Exact original prepared native array')
        native=hashlib.sha256(memoryview(value).cast('B')).hexdigest();close(value);path.chmod(0o444)
        digest=hashlib.sha256()
        with path.open('rb',buffering=0) as f:
            for block in iter(lambda:f.read(1<<20),b''):digest.update(block);os.posix_fadvise(f.fileno(),f.tell()-len(block),len(block),os.POSIX_FADV_DONTNEED)
        size=path.stat().st_size;total+=size
        require(size<=(16<<20)+4096 and total<=1<<30 and len(files)<198,'Existing finite per-layer output limits')
        files[name]=dict(bytes=size,sha256=digest.hexdigest(),native_sha256=native,shape=list(shape),dtype=np.dtype(dtype).str)
        progress('sealed',name,len(files),total)
    for name,shape in shapes.items():
        check();mapped=np.lib.format.open_memmap(destination/name,mode='w+',dtype='<u4',shape=shape);close(mapped)
    for index in range(len(plan['stripes'])):
        check();packed=pack_batch(batch(index),reader.read_rows)
        for item,segment in scatter[index]:
            mapped=np.load(destination/item['file'],mmap_mode='r+',allow_pickle=False);flat=mapped.reshape(-1,6144)
            for j in range(segment['tile_count']):
                slot=item.get('file_flat_begin',0)+(segment['destination_y']+j)*item['width']+segment['destination_x']
                flat[slot]=packed[0,segment['ordinal_begin']+j]
            del flat;close(mapped)
        del packed;progress('packed',str(index),len(files),total)
    # Reconstruct every stripe from independently enumerated actual PE
    # coordinates. Never use writer segments as the validation expectation.
    for index,stripe in enumerate(plan['stripes']):
        check();gathered=np.empty((1,stripe['columns'],6144),dtype='<u4');groups={}
        for ordinal in range(stripe['columns']):
            name,slot=physical[stripe['x'],stripe['y']+ordinal];groups.setdefault(name,[]).append((ordinal,slot))
        for name,selections in groups.items():
            mapped=np.load(destination/name,mmap_mode='r',allow_pickle=False);flat=mapped.reshape(-1,6144)
            for ordinal,slot in selections:gathered[0,ordinal]=flat[slot]
            del flat;close(mapped)
        checks.append(dict(logical_stripe=index,tensor=batch(index)['tensor'],**validate_packed_batch(batch(index),gathered,reader.read_rows)))
        del gathered;progress('independent_original_decode',str(index),len(files),total)
    for name,shape in shapes.items():seal(name,shape,'<u4')
    prefix=f'model.language_model.layers.{plan["layer"]}.'
    def read(suffix,shape):
        value=reader.read_tensor(prefix+suffix);require(value.shape==tuple(shape) and value.dtype==np.dtype('<u2'),'Own original support shape/type');return value
    arrays=configuration_arrays(plan,transport)
    arrays.update({'input-norm.npy':read('input_layernorm.weight',[5120]),'post-norm.npy':read('post_attention_layernorm.weight',[5120])})
    if linear:
        gains=read('linear_attn.norm.weight',[128]);a=read('linear_attn.A_log',[48]);bias=read('linear_attn.dt_bias',[48])
        arrays.update({'convolution-weights.npy':read('linear_attn.conv1d.weight',[10240,1,4]).reshape(80,512),
                       'head-gains.npy':np.tile(gains,(48,1)),'head-parameters.npy':np.column_stack((a,bias))})
        require(np.array_equal(arrays['convolution-weights.npy'].reshape(10240,1,4),read('linear_attn.conv1d.weight',[10240,1,4])),'All own original convolution tap bits')
        gains=read('linear_attn.norm.weight',[128]);a=read('linear_attn.A_log',[48]);bias=read('linear_attn.dt_bias',[48])
        for head in range(48):require(np.array_equal(arrays['head-gains.npy'][head],gains) and arrays['head-parameters.npy'][head].tolist()==[int(a[head]),int(bias[head])],'Every original head gain/scalar bit')
    else:
        q=read('self_attn.q_norm.weight',[256]);k=read('self_attn.k_norm.weight',[256])
        bits=np.asarray(frequency_bits,dtype='<u4');require(bits.shape==(32,) and bits.view('<f4')[0]==1,'Qualified unchanged rotary frequency bits')
        arrays.update({'qk-norms.npy':np.tile(np.concatenate([q,k]),(24,1)),'frequencies.npy':np.tile(bits.view('<f4'),(24,1))})
        q=read('self_attn.q_norm.weight',[256]);k=read('self_attn.k_norm.weight',[256])
        for head in range(24):
            require(np.array_equal(arrays['qk-norms.npy'][head,:256],q) and np.array_equal(arrays['qk-norms.npy'][head,256:],k) and
                    np.array_equal(arrays['frequencies.npy'][head].view('<u4'),bits),'Every duplicated original Q/K gain and unchanged frequency bit')
    for name,suffix in [('input-norm.npy','input_layernorm.weight'),('post-norm.npy','post_attention_layernorm.weight')]:
        require(np.array_equal(arrays[name],read(suffix,[5120])),'Original normalization bits re-read independently')
    for name,value in arrays.items():
        check();require(name not in files and Path(name).name==name and not name.startswith('reference-'),'Only unique causal execution parameter files')
        with (destination/name).open('xb') as f:np.save(f,value,allow_pickle=False);f.flush();os.fsync(f.fileno())
        seal(name,value.shape,value.dtype)
    expected={item['file'] for kind in ('matrix_uploads','config_uploads','parameter_uploads') for item in host[kind]}
    require(set(files)==expected and len(files)==(198 if linear else 53),'All and only finite original execution parameters')
    origin=reader.finish();manifest=dict(kind='direct_original_rows_to_vertical_ROIs',layer=plan['layer'],family=family,
              application=plan['application'],files=files,original_source=origin,matrix_checks=checks,all_original_bits_checked=True,
              original_geometry_checked=True,original_tensors=len(tensors),original_weight_bytes=sum(t['bytes'] for t in tensors.values()),
              packed_matrix_bytes=sum(math.prod(s)*4 for s in shapes.values()),array_file_bytes=total,
              reference_input_files=0,SDK_invocations=0,model_forward=False,whole_shard_rehash=False,seconds=time.monotonic()-started)
    raw=(json.dumps(manifest,separators=(',',':'))+'\n').encode();require(len(raw)<=2<<20,'Small per-layer preparation manifest')
    path=destination/'prepared.json'
    with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    path.chmod(0o444);fd=os.open(destination,os.O_RDONLY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return dict(layer=plan['layer'],family=family,files=len(files),array_file_bytes=total,manifest=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),
                original_tensors=len(tensors),original_range_bytes=reader.read_bytes,seconds=manifest['seconds'])
