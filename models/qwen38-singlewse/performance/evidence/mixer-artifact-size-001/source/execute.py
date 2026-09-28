import gzip,hashlib,json,tarfile,time
from pathlib import Path
class Counter:
 def __init__(self):self.bytes=0
 def write(self,b):self.bytes+=len(b);return len(b)
 def flush(self):pass
root=Path('/srv/model-storage/qwen38-singlewse/runs/layer-mlp-compile-018')
started=time.monotonic();files=sorted(p for p in (root/'out').rglob('*') if p.is_file())+sorted(root.glob('*.csl'))
stamps={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
sink=Counter()
with gzip.GzipFile(fileobj=sink,mode='wb',compresslevel=6,mtime=0) as packed:
 with tarfile.open(fileobj=packed,mode='w|') as tar:
  for p in files:
   name='csl/'+p.name if p.suffix=='.csl' and p.parent==root else str(p.relative_to(root))
   tar.add(p,arcname=name,recursive=False)
assert stamps=={str(p):(p.stat().st_size,p.stat().st_mtime_ns) for p in files}
result=dict(files=len(files),expanded_bytes=sum(s[0] for s in stamps.values()),gzip_bytes=sink.bytes,seconds=time.monotonic()-started,physical=False,payload_saved=False,source='layer-mlp-compile-018',scope='Host tar/gzip size estimate from existing local compiler output; server artifact format may differ.')
Path('COMPLETE.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
