"""Stream large compiled artifacts; never retain all members/ELFs in memory."""
import hashlib,heapq,json,re,sqlite3,tarfile,time
from pathlib import Path,PurePosixPath
from elf_inventory import inventory,admit_wse3_sram,placement

def sha_file(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
    return digest.hexdigest()

def audit(artifact,root,application_pes,*,application=(750,1160),compressed_limit=32<<30,expanded_limit=80<<30,member_limit=1800000,max_elf_bytes=256<<10):
    artifact=Path(artifact);root=Path(root)
    assert artifact.is_file() and not artifact.is_symlink() and 0<artifact.stat().st_size<=compressed_limit
    assert type(application_pes) is int and 0<application_pes<=870000
    assert 0<max_elf_bytes<=8<<20
    width,height=application
    assert width*height==application_pes
    coverage=bytearray(application_pes)
    stamp=(artifact.stat().st_size,artifact.stat().st_mtime_ns,artifact.stat().st_ino)
    started=time.monotonic();artifact_sha=sha_file(artifact)
    expected={p.name:sha_file(p) for p in root.glob('*.csl')};assert expected
    found=set();count=0;expanded=0;apps=0;failed=0;maximum=0;heaviest=[];error=None
    destination=root/'artifact-validation';destination.mkdir()
    connection=sqlite3.connect(destination/'members.sqlite')
    connection.execute('PRAGMA cache_size=-2048');connection.execute('PRAGMA journal_mode=OFF');connection.execute('PRAGMA synchronous=OFF')
    connection.execute('CREATE TABLE members (path TEXT PRIMARY KEY) WITHOUT ROWID')
    try:
        with (destination/'sram.jsonl').open('x') as records,tarfile.open(artifact,mode='r|*') as archive:
            for member in archive:
                # tarfile3.11 caches members even in streaming mode. Nothing in
                # this audit uses indexed lookup; drop that cache every entry.
                archive.members.clear()
                count+=1;expanded+=member.size
                if count>member_limit or expanded>expanded_limit:raise ValueError('Archive aggregate bound')
                path=PurePosixPath(member.name)
                if path.is_absolute() or '..' in path.parts or '\\' in member.name or len(member.name)>1024:
                    raise ValueError('Archive path profile')
                if not(member.isdir() or member.isfile()):raise ValueError('Archive link/special member rejected')
                connection.execute('INSERT INTO members(path) VALUES (?)',(member.name,))
                if count%4096==0:connection.commit()
                if not member.isfile():continue
                is_app='/out/bin/' in member.name and member.name.endswith('.elf')
                is_source='/csl/' in member.name and member.name.endswith('.csl')
                if is_app:
                    if not 0<member.size<=max_elf_bytes:raise ValueError('ELF member size profile')
                    stream=archive.extractfile(member);raw=stream.read(member.size+1)
                    if len(raw)!=member.size:raise ValueError('Truncated ELF')
                    gate=admit_wse3_sram(inventory(raw),stack_allowance=4096,ceiling=48128)
                    locations=placement(raw)
                    for x,y,w,h in locations['rectangles']:
                        x-=4;y-=1
                        if x<0 or y<0 or x+w>width or y+h>height:raise ValueError('Application rectangle outside declared layout')
                        for row in range(y,y+h):
                            start=row*width+x
                            if any(coverage[start:start+w]):raise ValueError('Overlapping application program regions')
                            coverage[start:start+w]=bytes([1])*w
                    record=dict(file=member.name,sha256=hashlib.sha256(raw).hexdigest(),placement=locations,**gate)
                    records.write(json.dumps(record,separators=(',',':'))+'\n');apps+=1;failed+=not gate['passed']
                    maximum=max(maximum,gate['low_section_end']);heapq.heappush(heaviest,(gate['low_section_end'],member.name))
                    if len(heaviest)>12:heapq.heappop(heaviest)
                    if apps>application_pes:raise ValueError('Too many application ELFs')
                elif is_source:
                    if path.name not in expected or path.name in found or member.size>8<<20:raise ValueError('Embedded source set/profile')
                    digest=hashlib.sha256();stream=archive.extractfile(member)
                    for chunk in iter(lambda:stream.read(1<<20),b''):digest.update(chunk)
                    if digest.hexdigest()!=expected[path.name]:raise ValueError('Embedded source identity')
                    found.add(path.name)
            connection.commit()
        if sum(coverage)!=application_pes:raise ValueError('Application PE coverage')
        if found!=set(expected):raise ValueError('Missing embedded sources')
        if failed:raise ValueError('Application SRAM ceiling')
        if stamp!=(artifact.stat().st_size,artifact.stat().st_mtime_ns,artifact.stat().st_ino):raise ValueError('Artifact mutated during validation')
    except BaseException as exc:
        error=repr(exc);raise
    finally:
        connection.close()
        result=dict(passed=error is None,error=error,artifact=str(artifact),artifact_sha256=artifact_sha,
            compressed_bytes=stamp[0],expanded_bytes=expanded,members=count,application_elfs=apps,application_pes=sum(coverage),expected_application_pes=application_pes,
            failed_sram_pes=failed,max_low_section_end=maximum,stack_allowance=4096,ceiling=48128,
            heaviest=[dict(low_section_end=n,file=p) for n,p in sorted(heaviest,reverse=True)],source_files=len(found),
            source_identity_pass=found==set(expected),seconds=time.monotonic()-started,
            scope='Every application ELF plus declared stack reserve; not measured dynamic stack peak or full-model numerical acceptance',
            member_cache_cleared=True,max_elf_buffer_bytes=max_elf_bytes,hash_buffer_bytes=1<<20)
        (destination/'summary.json').write_text(json.dumps(result,indent=2)+'\n')
    return result
