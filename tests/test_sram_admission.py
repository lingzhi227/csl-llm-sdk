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

    def test_fifo_configuration_keeps_the_original_sram_and_stack_limits(self):
        f=self.footprint(39088)
        f['allocated_sections'] += [
            {'name':'.s1ds','type':1,'address':64832,'bytes':8},
            {'name':'.dds','type':1,'address':64960,'bytes':8},
            {'name':'.xds','type':1,'address':64128,'bytes':24},
        ]
        result=admit_wse3_sram(f,4096,48128)
        self.assertEqual(result['low_section_end'],39088)
        self.assertEqual(result['low_section_end']+result['stack_allowance_bytes'],43184)
        self.assertTrue(result['passed'])
        f['allocated_sections'][0]['bytes']=44033
        self.assertFalse(admit_wse3_sram(f,4096,48128)['passed'])

    def test_unrecognized_fifo_initializer_is_not_ignored(self):
        for name,address,size in (('.s1ds',64832,8),('.dds',64960,8),('.xds',64128,24)):
            valid={'name':name,'type':1,'address':address,'bytes':size}
            for field,value in (('name',name+'_unknown'),('type',8),('address',address+2),('bytes',size+2)):
                with self.subTest(section=name,field=field):
                    f=self.footprint(39088)
                    f['allocated_sections'].append(dict(valid,**{field:value}))
                    self.assertFalse(admit_wse3_sram(f,4096,48128)['passed'])
