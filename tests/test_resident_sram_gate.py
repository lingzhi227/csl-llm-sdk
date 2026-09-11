"""Exercise the new8PE compiled-state gate at its exact ordinary SRAM boundary."""
import struct,sys,tempfile,unittest
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'core'))
from qwen38 import resident_sdk_admission as module

class GateTests(unittest.TestCase):
 def test_all_eight_actual_footprints_require_the_one_KiB_spare(self):
  with tempfile.TemporaryDirectory() as tmp:
   work=Path(tmp).resolve();(work/'out/bin').mkdir(parents=True)
   for end,passed in ((44032,True),(44033,False),(44912,False)):
    # Real ELF64 header/section parsing; synthetic NOBITS SRAM extent.
    strings=b'\0.bss\0.shstrtab\0'
    raw=bytearray(256+len(strings));raw[:7]=b'\x7fELF\x02\x01\x01'
    struct.pack_into('<Q',raw,40,64);struct.pack_into('<HHH',raw,58,64,3,2)
    struct.pack_into('<IIQQQQIIQQ',raw,128,1,8,3,0,0,end,0,0,4,0)
    struct.pack_into('<IIQQQQIIQQ',raw,192,6,3,0,0,256,len(strings),0,0,1,0)
    raw[256:]=strings
    for rank in range(8):(work/f'out/bin/out_{rank}_0.elf').write_bytes(raw)
    result=module.compiled_state(work)
    self.assertEqual(result['passed'],passed)
    self.assertEqual(len(result['pes']),8)
    self.assertTrue(all(v['stack_allowance_bytes']==4096 and v['ordinary_address_ceiling']==48128 for v in result['pes'].values()))

if __name__=='__main__':unittest.main()
