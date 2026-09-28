import hashlib,json,sys,tarfile,time
from pathlib import Path
started=time.monotonic();config=json.loads(Path('payload.json').read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
try:
 for n,h in json.loads(Path('source-manifest.json').read_text())['files'].items():assert sha(Path(n))==h
 stamps={}
 for n,p in config['paths'].items():
  path=Path(p);assert 0<path.stat().st_size<=512<<20
  assert sha(path)==config['hashes'][n];stamps[n]=(path.stat().st_ino,path.stat().st_size,path.stat().st_mtime_ns)
 with tarfile.open(fileobj=sys.stdout.buffer,mode='w|') as tar:
  for n,p in config['paths'].items():tar.add(p,arcname=n,recursive=False)
 sys.stdout.buffer.flush()
 for n,p in config['paths'].items():
  t=Path(p).stat();assert (t.st_ino,t.st_size,t.st_mtime_ns)==stamps[n]
 for n,h in json.loads(Path('source-manifest.json').read_text())['files'].items():assert sha(Path(n))==h
 Path('COMPLETE.json').write_text(json.dumps(dict(passed=True,physical=False,neural_execution=False,streamed_bytes=sum(s[1] for s in stamps.values()),hashes=config['hashes'],seconds=time.monotonic()-started))+'\n')
except BaseException as error:
 Path('FAILURE.json').write_text(json.dumps(dict(error=type(error).__name__,message=str(error)))+'\n');raise
