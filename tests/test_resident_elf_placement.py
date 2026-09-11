"""Actual placement gate with metadata faults; no native SDK or fabric execution."""
import importlib.util
import sys
import tempfile
import types
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'core'))
from qwen38.resident_elf_placement import inspect_placement
from qwen38.resident_sdk_driver import Evidence  # Import once before patch.dict(sys.modules).


class Reader:
    def __init__(self,path):self.rank=int(path.stem.split('_')[1])
    def get_fabric_dimensions(self):return (11,4)
    def iter_rectangles(self):
        yield (4+self.rank%4,1+self.rank//4,1,1)


class PlacementTests(unittest.TestCase):
    def test_complete_coverage_and_duplicate_segments_within_one_ELF(self):
        result=inspect_placement(Path('/synthetic'),Reader)
        self.assertEqual([v['fabric_xy'] for v in result['files']],
                         [[4,1],[5,1],[6,1],[7,1],[4,2],[5,2],[6,2],[7,2]])
        class Repeated(Reader):
            def iter_rectangles(self):
                yield from super().iter_rectangles();yield from super().iter_rectangles()
        self.assertTrue(all(v['segment_rectangle_count']==2 for v in inspect_placement(Path('/synthetic'),Repeated)['files']))

    def test_wrong_missing_overlapping_oversized_and_unbounded_rectangles(self):
        faults=[[],[(4,1,2,1)],[(5,1,1,1)],[(4,1,1,1),(4,2,1,1)],[(4,1,1,1)]*65]
        for rectangles in faults:
            class Broken(Reader):
                def iter_rectangles(self):return iter(rectangles) if self.rank==0 else super().iter_rectangles()
            with self.subTest(rectangles=rectangles[:2]),self.assertRaises(ValueError):
                inspect_placement(Path('/synthetic'),Broken)
        class Swapped(Reader):
            def __init__(self,path):
                super().__init__(path)
                if self.rank in (0,4):self.rank=4-self.rank
        with self.assertRaisesRegex(ValueError,'Wrong ELF placement'):inspect_placement(Path('/synthetic'),Swapped)

    def test_wrong_fabric_rejected(self):
        class Wrong(Reader):
            def get_fabric_dimensions(self):return (11,3)
        with self.assertRaisesRegex(ValueError,'fabric dimensions'):inspect_placement(Path('/synthetic'),Wrong)

    def test_actual_driver_blocks_Runtime_construction_on_bad_metadata(self):
        here=ROOT/'examples/wp16/resident/driver.py'
        if not here.exists():here=ROOT/'driver.py'
        spec=importlib.util.spec_from_file_location('placement_driver_test',here)
        driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)
        class Wrong(Reader):
            def get_fabric_dimensions(self):return (11,3)
        for reader,passes in ((Wrong,False),(Reader,True)):
            with self.subTest(passes=passes),tempfile.TemporaryDirectory() as tmp,ExitStack() as stack:
                work=Path(tmp).resolve()
                for name in ('verify','verify_compiled','current','load_prepared'):
                    stack.enter_context(patch.object(driver,name,return_value=(None,None,None) if name=='load_prepared' else None))
                stack.enter_context(patch.object(driver,'WORK',work));stack.enter_context(patch.object(Path,'cwd',return_value=work))
                stack.enter_context(patch.dict(driver.os.environ,WP16_SDK_STAGE='simulate'))
                elf=types.ModuleType('cerebras.elf.cself');elf.ELFMemory=reader
                runtime=types.ModuleType('cerebras.sdk.runtime.sdkruntimepybind')
                construct=stack.enter_context(patch.object(runtime,'SdkRuntime',create=True,side_effect=RuntimeError('construction sentinel')))
                runtime.MemcpyDataType=runtime.MemcpyOrder=object
                runtime.SimfabConfig=lambda **kwargs:None
                runtime.SdkTarget=types.SimpleNamespace(WSE3=3);runtime.get_platform=lambda *args:None
                stack.enter_context(patch.dict(sys.modules,{'cerebras.elf.cself':elf,'cerebras.sdk.runtime.sdkruntimepybind':runtime}))
                with self.assertRaisesRegex(RuntimeError if passes else ValueError,'construction sentinel' if passes else 'fabric dimensions'):
                    driver.main()
                self.assertEqual(construct.call_count,int(passes))
                self.assertEqual((work/'placement-admission.json').exists(),passes)


if __name__=='__main__':unittest.main()
