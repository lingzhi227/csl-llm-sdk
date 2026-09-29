"""Compare published bytes with the functional baseline and prior frozen evidence."""
import argparse,hashlib,json,subprocess
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--repo',type=Path,required=True)
    p.add_argument('--previous',required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    if a.output.exists():p.error('Fresh immutable receipt required')
    prefix='models/qwen38-singlewse/'
    baseline='6c2f4f5685478ee100f167e42a9f7f22f57dca35'
    notices={prefix+'docs/STATUS.md',prefix+'docs/ACCEPTANCE.md'}
    def tree(ref,path):
        raw=subprocess.check_output(['git','-C',str(a.repo),'ls-tree','-rz','--full-tree',ref,'--',path])
        out={}
        for item in raw.split(b'\0'):
            if not item:continue
            meta,name=item.split(b'\t',1);mode,kind,digest=meta.split()
            if kind!=b'blob':raise ValueError('Unexpected non-file tree entry')
            out[name.decode()]=digest.decode()
        return out
    def identical(files,skip=()):
        count=0
        for name,expected in files.items():
            if name in skip:continue
            raw=(a.repo/name).read_bytes()
            actual=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            if actual!=expected:raise ValueError('Preserved byte identity changed: '+name)
            count+=1
        return count
    original=tree(baseline,prefix);frozen=tree(a.previous,prefix+'performance/evidence/')
    original_count=identical(original,notices);frozen_count=identical(frozen)
    manifest=json.loads((a.repo/'PUBLIC_MANIFEST.json').read_text())
    for name,digest in manifest.items():
        if hashlib.sha256((a.repo/name).read_bytes()).hexdigest()!=digest:raise ValueError('Public manifest mismatch: '+name)
    exported=json.loads((a.repo/prefix/'performance/SOURCE_EXPORT.json').read_text())['files']
    for name,receipt in exported.items():
        if hashlib.sha256((a.repo/prefix/'performance'/name).read_bytes()).hexdigest()!=receipt['published_sha256']:
            raise ValueError('Export manifest mismatch: '+name)
    report=dict(passed=True,functional_baseline=baseline,baseline_files=len(original),original_unchanged=original_count,
                authorized_notices=sorted(notices),previous_publication=a.previous,unchanged_frozen_evidence_files=frozen_count,
                pre_receipt_manifest_files=len(manifest),pre_receipt_export_files=len(exported))
    a.output.open('x').write(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
