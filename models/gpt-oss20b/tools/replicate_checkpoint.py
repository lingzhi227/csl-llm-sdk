"""Resumable bounded relay of the verified checkpoint to the physical host.

Only OS pipe buffers pass through the Mac. No model file is created locally.
Each remote file is SHA256-verified before its .partial name is committed.
"""
import hashlib,json,shlex,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE='/srv/model-storage/gpt-oss20b/model'
DEST='/srv/gpt-oss20b-hardware/model'
SESSION=['sh','/path/to/alcf-session.sh','host']

def remote(code,timeout=30):
    r=subprocess.run(SESSION+['python3 -c '+shlex.quote(code)],capture_output=True,text=True,check=True,timeout=timeout)
    return json.loads(r.stdout)

def main():
    # Use a persistent directory even if the invoking shell's old cwd vanished.
    import os,fcntl
    os.chdir(ROOT)
    lock=(ROOT/'evidence/acquisition/alcf-replica.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    complete=json.loads((ROOT/'evidence/acquisition/COMPLETE.json').read_text())
    assert complete['model']=='openai/gpt-oss-20b' and complete['tensors']==459
    receipt=ROOT/'evidence/acquisition/alcf-replica-progress.json';records=[]
    started=time.monotonic()
    for item in complete['files']:
        name=item['file'];assert Path(name).name==name and item['publisher_verified']
        final=DEST+'/'+name;partial=final+'.partial'
        stat=remote("from pathlib import Path;import json; p=Path("+repr(final)+"); q=Path("+repr(partial)+"); p.parent.mkdir(parents=True,exist_ok=True); print(json.dumps({'final':p.exists(),'bytes':q.stat().st_size if q.exists() else 0}))")
        if not stat['final']:
            offset=stat['bytes'];assert 0<=offset<=item['bytes']
            while offset<item['bytes']:
                count=min(256<<20,item['bytes']-offset)
                reader=subprocess.Popen(['ssh','workstation',shlex.join(['dd','if='+SOURCE+'/'+name,'iflag=skip_bytes,count_bytes','skip='+str(offset),'count='+str(count),'status=none'])],stdout=subprocess.PIPE)
                code="from pathlib import Path;import os,sys; p=Path("+repr(partial)+"); assert (p.stat().st_size if p.exists() else 0)=="+str(offset)+"; f=p.open('ab'); remaining="+str(count)+"\n"
                code+="while remaining:\n b=sys.stdin.buffer.read(min(1<<20,remaining))\n if not b: raise EOFError('Short checkpoint chunk')\n f.write(b);remaining-=len(b)\nf.flush();os.fsync(f.fileno());f.close()\n"
                try:subprocess.run(SESSION+['python3 -c '+shlex.quote(code)],stdin=reader.stdout,check=True,timeout=240)
                finally:
                    reader.stdout.close()
                    try:code=reader.wait(timeout=15)
                    except subprocess.TimeoutExpired:reader.terminate();reader.wait(timeout=10);raise
                    if code:raise RuntimeError('Checkpoint source stream failed')
                offset+=count
                receipt.write_text(json.dumps(dict(complete=False,current=name,bytes=offset,total=item['bytes'],verified=records,elapsed_seconds=time.monotonic()-started),indent=2)+'\n')
                print(json.dumps(dict(file=name,bytes=offset,total=item['bytes'])),flush=True)
        code="from pathlib import Path;import hashlib,json; p=Path("+repr(final if stat['final'] else partial)+"); assert p.stat().st_size=="+str(item['bytes'])+"; h=hashlib.sha256()\n"
        code+="with p.open('rb') as f:\n while True:\n  b=f.read(8<<20)\n  if not b:break\n  h.update(b)\n"
        code+="assert h.hexdigest()=="+repr(item['sha256'])+"; "
        if not stat['final']:code+="p.rename("+repr(final)+"); "
        code+="print(json.dumps({'file':p.name,'sha256':h.hexdigest(),'verified':True}))"
        verified=remote(code,timeout=240);records.append(dict(file=name,sha256=verified['sha256'],bytes=item['bytes']))
    raw=json.dumps(complete,indent=2)+'\n'
    subprocess.run(SESSION+['cat > '+shlex.quote(DEST+'/COMPLETE.json')],input=raw,text=True,check=True,timeout=30)
    result=dict(complete=True,remote=DEST,revision=complete['revision'],files=records,elapsed_seconds=time.monotonic()-started,local_payload_files_created=False)
    receipt.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)

if __name__=='__main__':main()
