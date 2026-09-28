"""Independent wire reconstruction and host ownership checks for bounded batches."""
import sys,unittest
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'runtime'))
from device_client import Encoder,Client,batches


class DeviceClientTests(unittest.TestCase):
 def test_interleaved_complete_bank_extents_survive_bounded_batches(self):
  sizes=[8545,8481,7133,7133,7149,7246,7246,7245];tags=[1,0,3509,3508,3470,3420,3421,3459]
  original={t:(np.arange(n,dtype=np.uint32)*np.uint32(2654435761)) for t,n in zip(tags,sizes)}
  destination={t:np.zeros(n,np.uint32) for t,n in zip(tags,sizes)};visits={t:np.zeros(n,np.uint8) for t,n in zip(tags,sizes)}
  encoder=Encoder(3554)
  def writes():
   for first in range(0,max(sizes),256):
    for t in tags:
     p=original[t][first:first+256]
     if len(p):yield encoder.command(t,0,0,first,len(p),payload=p)
  total=0;sequences={t:0 for t in tags};batch_count=0
  for batch in batches(writes(),1024):
   wire=[v for frame in batch for v in frame.wire];self.assertLessEqual(len(wire),1024);cursor=0;batch_count+=1
   while cursor<len(wire):
    selector,length,replies,sequence=wire[cursor:cursor+4];cursor+=4
    self.assertTrue(selector&0x80000000);t=selector&0x7fffffff;total+=1;self.assertEqual(sequence,total)
    body=wire[cursor:cursor+length];cursor+=length;sequences[t]+=1
    self.assertEqual(body[:3],[sequences[t],0,0]);self.assertEqual(replies,4);first,count=body[3:5];self.assertEqual(length,6+count)
    destination[t][first:first+count]=body[6:];visits[t][first:first+count]+=1
  self.assertEqual(total,239);self.assertGreater(batch_count,1)
  for t in tags:
   np.testing.assert_array_equal(destination[t],original[t]);np.testing.assert_array_equal(visits[t],1)

 def test_reject_invalid_body_and_halfword_alignment_before_sequence_changes(self):
  e=Encoder(3554)
  for args in [(0,0,0,0,2,0,[1]),(0,0,1,0,1,0,[65536]),(0,1,1,1,2,0,[])]:
   with self.assertRaises(ValueError):e.command(*args)
  self.assertEqual(e.total,0);self.assertEqual(e.sequences,{})
  with self.assertRaises(ValueError):e.operand(1,40,40,[0]*64)
  with self.assertRaises(ValueError):e.operand(0,0,40,[0]*64)

 def test_operand_is_inline_full_width_and_does_not_advance_worker_sequence(self):
  e=Encoder(3554);words=[0xffffffff,0x80000001,0xffff0001,0x0001ffff]*16
  f=e.operand(0xf1230001,47,48,words)
  self.assertEqual(f.wire[:7],(0x80000000|3554,67,0,1,0xf1230001,47,48))
  self.assertEqual(list(f.wire[7:]),words);self.assertEqual(e.sequences,{})

 def test_read_reply_owns_arena_and_invalid_batch_never_reaches_sdk(self):
  class Constants:ROW_MAJOR=0;MEMCPY_32BIT=0
  class Runner:
   def __init__(self):self.calls=0
   def get_id(self,name):return name
   def memcpy_h2d(self,*args,**kwargs):self.calls+=1;self.wire=args[1].copy()
   def memcpy_d2h(self,destination,name,*args,**kwargs):
    destination[:]=[1,0,1,1] if name in ('device_audit',15) else [1,0,2,16,0xffff8000]
  r=Runner();client=Client(r,Constants,Constants,(2,2));e=Encoder(3554)
  read=e.command(0,1,1,0,2);write=e.command(0,0,0,0,1,payload=[7])
  with self.assertRaises(ValueError):client.stream([read,write])
  self.assertEqual(r.calls,0);client.stream([read])
  self.assertEqual(int(r.wire[0]),0xc0000000)
  with self.assertRaises(ValueError):client.stream([write])
  np.testing.assert_array_equal(client.read_reply(),[0x8000,0xffff])
  with self.assertRaises(ValueError):client.stream([read])


if __name__=='__main__':unittest.main()
