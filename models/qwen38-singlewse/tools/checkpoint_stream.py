"""Bounded public-checkpoint transport over existing SSH, without Mac payload storage."""
import argparse, fcntl, hashlib, json, os, resource, shutil, sys, time
from pathlib import Path


def items(hub):
    support={'config.json','model.safetensors.index.json','generation_config.json',
             'tokenizer.json','tokenizer_config.json','chat_template.jinja','LICENSE','README.md'}
    result=sorted((s for s in hub['siblings'] if s['rfilename'] in support or s['rfilename'].endswith('.safetensors')),key=lambda x:x['rfilename'])
    assert len(result)==74 and sum(x['size'] for x in result)==30879968808
    assert all(Path(x['rfilename']).name==x['rfilename'] for x in result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['send','receive']);a=p.parse_args()
    # ALCF system Python maps272400 KiB before reading the manifest (RSS14952 KiB).
    # Keep bounded resident buffers; allow the measured interpreter mappings.
    resource.setrlimit(resource.RLIMIT_AS,(512<<20,512<<20))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))
    resource.setrlimit(resource.RLIMIT_FSIZE,(8<<30,8<<30))
    resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
    os.nice(10)
    hub=json.loads(Path('hub.json').read_text());entries=items(hub)
    root=Path('/srv/model-storage/qwen38-singlewse/model' if a.mode=='send' else '/srv/qwen38-singlewse-hardware/model')
    receipts=[];started=time.monotonic();total=0;last=0;error=None
    if a.mode=='receive':
        lock=(root/'download.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert shutil.disk_usage(root).free>sum(x['size'] for x in entries)+(32<<30)
    try:
        for item in entries:
            name=item['rfilename'];size=item['size'];sha=hashlib.sha256();blob=hashlib.sha1(f'blob {size}\0'.encode())
            path=root/name;partial=root/(name+'.partial');count=0
            if a.mode=='send':
                assert path.stat().st_size==size
                stream=path.open('rb');sink=sys.stdout.buffer
            else:
                # A fresh attempt never overwrites verified files or an older partial.
                assert not path.exists() and not partial.exists(), name
                stream=sys.stdin.buffer;sink=partial.open('xb')
            try:
                while count<size:
                    block=stream.read(min(1<<20,size-count))
                    if not block:raise EOFError(name)
                    sha.update(block);blob.update(block);sink.write(block)
                    count+=len(block);total+=len(block)
                    if a.mode=='receive':
                        delay=total/(16*(1<<20))-(time.monotonic()-started)
                        if delay>0:time.sleep(delay)
                        if time.monotonic()-last>5:
                            Path('PROGRESS.json').write_text(json.dumps(dict(file=name,file_bytes=count,file_total=size,total_bytes=total,completed_files=len(receipts),seconds=time.monotonic()-started))+'\n');last=time.monotonic()
                sink.flush()
                if a.mode=='receive':os.fsync(sink.fileno())
            finally:
                (stream if a.mode=='send' else sink).close()
            expected=item.get('lfs',{}).get('sha256')
            if expected:assert sha.hexdigest()==expected,name
            else:assert blob.hexdigest()==item['blobId'],name
            if a.mode=='receive':partial.replace(path)
            receipts.append(dict(file=name,bytes=size,sha256=sha.hexdigest(),publisher_verified=True))
            Path('receipts.json').write_text(json.dumps(receipts,indent=2)+'\n')
        if a.mode=='receive':
            # Exact framing and no appended payload.
            if sys.stdin.buffer.read(1):raise ValueError('Extra transport payload')
            (root/'COMPLETE.json').write_text(json.dumps(dict(repository=hub['id'],revision=hub['sha'],files=receipts),indent=2)+'\n')
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        Path('receipt.json').write_text(json.dumps(dict(passed=error is None,error=error,mode=a.mode,total_bytes=total,completed_files=len(receipts),seconds=time.monotonic()-started,physical_job=False),indent=2)+'\n')


if __name__=='__main__':main()
