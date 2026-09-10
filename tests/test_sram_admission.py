from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'core'))
from qwen38.elf_inventory import admit_wse3_sram


class SramAdmissionTests(unittest.TestCase):
    def footprint(self,end):
        return {'static_allocated_section_bytes':end+96,'allocated_sections':[
            {'name':'.bss','address':0,'bytes':end}, {'name':'.filters','address':63104,'bytes':96}]}

    def test_48k_includes_stack_at_exact_boundary(self):
        self.assertTrue(admit_wse3_sram(self.footprint(49152-4096))['passed'])
        self.assertFalse(admit_wse3_sram(self.footprint(49152-4096+1))['passed'])
        self.assertFalse(admit_wse3_sram(self.footprint(46624))['passed'])
        with self.assertRaises(ValueError):admit_wse3_sram(self.footprint(46624),ceiling=61440)
        with self.assertRaises(ValueError):admit_wse3_sram(self.footprint(46624),stack_allowance=2048)

    def test_high_address_application_section_is_not_configuration(self):
        f=self.footprint(20000)
        f['allocated_sections'].append({'name':'.text','address':60000,'bytes':2})
        self.assertFalse(admit_wse3_sram(f)['passed'])
