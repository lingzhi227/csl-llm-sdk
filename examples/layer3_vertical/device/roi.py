"""Plan-derived legal ROI transfers for vertical weights, state and boundaries."""
from pathlib import Path
from collections import defaultdict
import json
HERE=Path(__file__).resolve().parent
MAX_HOST_BYTES=16<<20

def rectangles(points,count):
 """Merge equal horizontal runs, then bound their vertical host-slot size."""
 rows=defaultdict(list);runs=defaultdict(list)
 for x,y in points:rows[y].append(x)
 for y,xs in sorted(rows.items()):
  start=last=None
  for x in sorted(xs):
   if start is not None and x!=last+1:runs[start,last-start+1].append(y);start=None
   if start is None:start=x
   last=x
  if start is not None:runs[start,last-start+1].append(y)
 result=[]
 for (x,width),ys in sorted(runs.items()):
  limit=MAX_HOST_BYTES//(width*count*4);assert limit>=1
  first=last=None
  for y in ys:
   if first is not None and (y!=last+1 or y-first>=limit):result.append([x,first,width,last-first+1,count]);first=None
   if first is None:first=y
   last=y
  if first is not None:result.append([x,first,width,last-first+1,count])
 return sorted(result,key=lambda b:(b[1],b[0]))
def transfer(symbol,dtype,box,**extra):
 x,y,w,h,n=box;native=2 if dtype=='u16' else 4
 return dict(symbol=symbol,dtype=dtype,x=x,y=y,width=w,height=h,count=n,bits=native*8,
  order='ROW_MAJOR',host_shape=[h,w,n],host_bytes=w*h*n*4,native_bytes=w*h*n*native,**extra)
