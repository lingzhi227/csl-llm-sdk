"""Bounded immutable prepared-file views; load original weights once to device.

Preparation and runtime execute remotely under separate admitted owners. No
original model file is opened by this provider and no neural forward exists.
"""
from pathlib import Path
import hashlib,json,stat

def require(ok,message):
    if not ok:raise ValueError(message)

def fingerprint(path):
    s=path.stat();return (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)

class Prepared:
    def __init__(self,root,manifest_pin):
        import numpy as np
        self.np=np;self.root=Path(root);self.cache={};self.stamps={};self.cache_bytes=0
        path=self.root/'prepared.json'
        require(not path.is_symlink() and path.stat().st_size<=2<<20,'Bounded derived preparation manifest')
        raw=path.read_bytes()
        require(len(raw)==manifest_pin['bytes'] and hashlib.sha256(raw).hexdigest()==manifest_pin['sha256'],
                'Exact independently reviewed preparation receipt')
        self.manifest=json.loads(raw);require(self.manifest['all_original_bits_checked'],'Independent original bit check')
        require(self.manifest['kind']=='derived_vertical_ROIs_from_accepted_original_words' and
                self.manifest['application']==[30,1160] and self.manifest['all_original_words_reused_checked'],
                'Derived original-word preparation for the actual vertical geometry')
        self.files=self.manifest['files'];require(len(self.files)==199,'190 ROI matrix +8 support/config +1 input files')
        require(self.manifest['original_weight_bytes']==766546368,'Complete original Layer0 tensors')
        self.verified=set();self.disk_bytes_read=0

    def array(self,name):
        np=self.np;require(name in self.files and Path(name).name==name,'Declared prepared basename')
        path=self.root/name;pin=self.files[name]
        require(not path.is_symlink() and path.is_file(),'Real immutable prepared file')
        stamp=fingerprint(path);require(not stamp[-1]&0o222,'Prepared file is read only')
        require(stamp[2]==pin['bytes'] and pin['bytes']<=(16<<20)+4096,'Bounded prepared file identity')
        if name in self.stamps:require(self.stamps[name]==stamp,'Prepared file changed between transfers')
        if name in self.cache:return self.cache[name],pin
        # Hash each immutable file once. Later logical-stripe visits map only
        # their requested tiles instead of rereading a full physical ROI.
        first=name not in self.verified
        if first:
            digest=hashlib.sha256()
            with path.open('rb') as f:
                for block in iter(lambda:f.read(1<<20),b''):
                    digest.update(block);self.disk_bytes_read+=len(block)
            require(digest.hexdigest()==pin['sha256'] and fingerprint(path)==stamp,'Prepared bytes and stable file')
        value=np.load(path,mmap_mode='r',allow_pickle=False)
        require(list(value.shape)==pin['shape'] and value.dtype.str==pin['dtype'] and value.nbytes<=16<<20,
                'Exact prepared native shape and dtype')
        require(not value.flags.writeable and fingerprint(path)==stamp,'Immutable mapped prepared value')
        if first:require(hashlib.sha256(value.tobytes()).hexdigest()==pin['native_sha256'],'Prepared logical value identity')
        self.stamps[name]=stamp;self.verified.add(name)
        if pin['bytes']<=1<<20 and self.cache_bytes+value.nbytes<=4<<20:
            value=np.array(value,copy=True,order='C');value.setflags(write=False)
            self.cache[name]=value;self.cache_bytes+=value.nbytes
        return value,pin

    def transfer(self,item):
        np=self.np;source=item['prepared'];name=source['file'];value,pin=self.array(name)
        h,w,n=item['height'],item['width'],item['count']
        if 'file_x' in source:
            x,y=source['file_x'],source['file_y'];selected=value[y:y+h,x:x+w,:n]
        elif 'file_row' in source:
            row=source['file_row']
            if value.ndim==1:
                require(row==0 and h==w==1,'One flat parameter owner');selected=value[:n]
            elif value.ndim==2:
                require(w==1,'One parameter owner per prepared row');selected=value[row:row+h,:n]
            elif value.ndim==3:selected=value[row:row+h,:w,:n]
            else:raise ValueError('Known immutable parameter rank')
        else:selected=value
        selected=np.ascontiguousarray(selected).reshape(h,w,n)
        receipt=dict(file=name,sha256=pin['sha256'],shape=list(selected.shape),dtype=selected.dtype.str,
            native_sha256=hashlib.sha256(selected.tobytes()).hexdigest(),
            selection={k:v for k,v in source.items() if k.startswith('file_')})
        return selected,receipt

    def input_row(self,index):
        value,_=self.array('reference-input.npy')
        require(value.shape==(2,5120) and value.dtype==self.np.dtype('<u2') and index in (0,1),
                'Two actual original prompt rows')
        return self.np.ascontiguousarray(value[index]).reshape(1,1,5120)

    def verify_unchanged(self):
        for name,stamp in self.stamps.items():require(fingerprint(self.root/name)==stamp,'Prepared file changed during runtime')
        require(self.verified==self.files.keys(),'Every prepared original file was consumed')
        return dict(files=len(self.verified),disk_bytes_read=self.disk_bytes_read,cache_bytes=self.cache_bytes,
            unchanged=True,original_weight_reads_during_runtime=0)
