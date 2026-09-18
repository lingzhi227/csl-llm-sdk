"""One explicit offline csdb shell; bounded role rectangles with actual ELF addresses."""
import json,re,runpy,subprocess,time
from pathlib import Path
ROOT=Path.cwd();started=time.monotonic();m=json.loads((ROOT/'plan.json').read_bytes())['metadata']
commands=['context select out','target create --core-file=corefile.cs1','rectangle deselect 0,0 --dimension=44,12']
requests=[]
for role,word_address,word_length in [(0,1762,38),(1,2612,459),(2,3484,1331),(3,1910,38)]:
 cells={(x,y) for y,row in enumerate(m) for x,v in enumerate(row) if v[0]==role}
 while cells:
  x,y=min(cells,key=lambda xy:(xy[1],xy[0]));w=1
  while (x+w,y) in cells:w+=1
  h=1
  while all((xx,y+h) in cells for xx in range(x,x+w)):h+=1
  cells-={(xx,yy) for xx in range(x,x+w) for yy in range(y,y+h)}
  rect=f'{x+4},{y+1}';dims=f'{w},{h}'
  requests.append(dict(role=role,physical=[x+4,y+1,w,h],word_address=word_address,word_length=word_length))
  commands+=['rectangle select '+rect+' --dimension='+dims,'memory read --address='+str(word_address)+' --length='+str(word_length)+' --stdoutput','rectangle deselect '+rect+' --dimension='+dims]
commands+=['exit'];stdin='\n'.join(commands)+'\n'
r=subprocess.run(['csdb','.'],input=stdin,capture_output=True,text=True,timeout=55)
assert len(r.stdout)+len(r.stderr)<1048576
(ROOT/'csdb-batch-stdout.txt').write_text(r.stdout)
(ROOT/'csdb-batch-receipt.json').write_text(json.dumps(dict(code=r.returncode,requests=requests,stdin=stdin,stderr=r.stderr,seconds=time.monotonic()-started),indent=2)+'\n')
assert r.returncode==0 and 'No rectangles selected' not in r.stdout and 'ERROR:' not in r.stdout
rows=[]
for match in re.finditer(r'^\((\d+),(\d+)\) @ (0x[0-9a-fA-F]+):((?: [0-9a-fA-F]{4})+)\s*$',r.stdout,re.M):
 x,y,address,words=match.groups();rows.append(dict(x=int(x),y=int(y),word_address=int(address,16),words=[int(v,16) for v in words.split()]))
assert len(rows)==370 and len({(r['x'],r['y']) for r in rows})==370
bounds={}
for request in requests:
 x,y,w,h=request['physical']
 for xx in range(x,x+w):
  for yy in range(y,y+h):
   assert (xx,yy) not in bounds
   bounds[xx,yy]=(request['word_address'],request['word_length'])
for row in rows:assert (row['word_address'],len(row['words']))==bounds[row['x'],row['y']]
assert sum(len(row['words']) for row in rows)==92244
(ROOT/'csdb-memory-words.json').write_text(json.dumps(rows,separators=(',',':'))+'\n')
runpy.run_path(str(ROOT/'csdb_audit.py'),run_name='__main__')
