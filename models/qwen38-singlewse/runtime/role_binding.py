"""Bind every compiled application rectangle to its host initialization role.

This reads real ELF object extents. It does not infer SRAM from a source estimate.
The independent SDK rectangle qualification remains a required earlier gate.
"""
import argparse,hashlib,json,struct,tarfile,time
from pathlib import Path
import numpy as np
from elf_inventory import placement
from resident_plan import load

def objects(data):
    assert data[:6]==b'\x7fELF\x02\x01'
    offset=struct.unpack_from('<Q',data,40)[0];stride,count,_=struct.unpack_from('<HHH',data,58)
    sections=[struct.unpack_from('<IIQQQQIIQQ',data,offset+i*stride) for i in range(count)]
    result={}
    for table in sections:
        if table[1]!=2:continue
        strings=sections[table[6]];raw=data[strings[4]:strings[4]+strings[5]]
        assert table[9]==24 and table[5]%24==0
        for at in range(table[4],table[4]+table[5],24):
            index,info,other,section,address,size=struct.unpack_from('<IBBHQQ',data,at)
            if info&15!=1 or not size:continue
            name=raw[index:].split(b'\0',1)[0].decode('utf-8')
            # Compiler may attach an internal DSD address prefix.
            name=name.split('$$')[-1]
            assert name not in result or result[name]==[address,size]
            result[name]=[address,size]
    assert result;return result

def expected(role):
    common={'config':32,'trace':24}
    if role<=5:
        values=dict(common,weights={1:34816,2:35840,3:32768,4:10240}.get(role,2),
            scales=12 if role==1 else 4,data={4:10240,5:35072}.get(role,2))
        if role in (1,2):values['output']={1:1088,2:560}[role]
        return 'storage.csl',values
    if role<=8:
        values=dict(common,g_conv_weights=3072 if role==6 else 2,g_gate_parameters=4,g_gain=256 if role==6 else 2)
        if role==6:values['g_input']=1028
        else:values['g_state']=32768
        return 'gdn.csl',values
    if role<=13:
        values=dict(common,a_gain=1024 if role==9 else 2,
            a_frequencies=128 if role==9 else 4)
        if role==9:values['a_input']=7168
        if role>=10:values.update({'a_kv.key_cache':12288,'a_kv.value_cache':12288})
        return 'attention.csl',values
    assert role==14
    return 'controller.csl',dict(common,program=7064,matrices=7968,io=580,prompt=384,controls=16,generated=384,result=20)

def audit(root,artifact,destination):
    root=Path(root);destination=Path(destination);plan=load(root);coverage=np.zeros_like(plan['role'])
    started=time.monotonic();records=[];role_counts={};programs=0
    with tarfile.open(artifact,'r|gz') as archive:
        for member in archive:
            archive.members.clear()
            if not ('/out/bin/' in member.name and member.name.endswith('.elf')):continue
            assert 0<member.size<=8<<20
            raw=archive.extractfile(member).read(member.size+1);assert len(raw)==member.size
            where=placement(raw);symbols=objects(raw);seen=set()
            for x,y,w,h in where['rectangles']:
                x-=4;y-=1;assert 0<=x<x+w<=750 and 0<=y<y+h<=1160
                region=plan['role'][y:y+h,x:x+w];assert not coverage[y:y+h,x:x+w].any()
                for role in np.unique(region):
                    role=int(role);source,requirements=expected(role)
                    assert Path(where['source']).name==source,(member.name,role,where['source'])
                    for name,size in requirements.items():
                        assert name in symbols and symbols[name][1]==size,(member.name,role,name,size,symbols.get(name))
                        assert symbols[name][0]+size<=48128
                    count=int(np.count_nonzero(region==role));role_counts[str(role)]=role_counts.get(str(role),0)+count;seen.add(role)
                coverage[y:y+h,x:x+w]=1
            records.append(dict(file=member.name,roles=sorted(seen),symbol_extents_pass=True));programs+=1
    assert np.all(coverage==1) and programs==434
    with Path(artifact).open('rb') as stream:artifact_sha256=hashlib.file_digest(stream,'sha256').hexdigest()
    summary=dict(passed=True,application_pes=int(coverage.sum()),programs=programs,role_counts=role_counts,artifact_sha256=artifact_sha256,
        role_sha256=plan['role_sha256'],config_sha256=plan['config_sha256'],seconds=time.monotonic()-started,
        scope='Actual ELF source role and object extents on every application PE; independent of physical numerical acceptance')
    destination.mkdir();(destination/'programs.json').write_text(json.dumps(records,indent=2)+'\n')
    (destination/'COMPLETE.json').write_text(json.dumps(summary,indent=2)+'\n');return summary

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--compiled',type=Path,required=True);a=p.parse_args()
    artifact=json.loads((a.compiled/'artifact.json').read_text())['artifact']
    print(json.dumps(audit(a.compiled,artifact,Path.cwd()/'result')))
