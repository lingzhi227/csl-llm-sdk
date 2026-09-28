"""Artifact framing must preserve chunks and reject oversized/truncated data."""
import importlib.util,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT.parent/'runtime'))
spec=importlib.util.spec_from_file_location('performance_bounded_compiler',ROOT/'runtime/bounded_compiler.py')
compiler=importlib.util.module_from_spec(spec);spec.loader.exec_module(compiler)


def response(block,total,name='artifact.tar.gz'):
    return SimpleNamespace(HasField=lambda key:False,data=SimpleNamespace(file_name=name,total_bytes=total,
        data_chunk=block,num_bytes=len(block)))


class BoundedCompilerTests(unittest.TestCase):
    def test_multiple_server_chunks_are_appended_and_exact(self):
        with tempfile.TemporaryDirectory() as root:
            path=compiler.receive(iter([response(b'abc',7),response(b'defg',7)]),root,'artifact',max_total=7,max_chunk=4)
            self.assertEqual(Path(path).read_bytes(),b'abcdefg')
            self.assertFalse(Path(root,'artifact.tar.gz.partial').exists())

    def test_bad_framing_cannot_become_a_final_artifact(self):
        cases=[[response(b'a',8)], [response(b'abc',7)],
               [response(b'abc',7),response(b'defg',6)],
               [response(b'abcd',4,'wrong.tar.gz')], [response(b'abcde',5)]]
        for blocks in cases:
            with self.subTest(blocks=blocks),tempfile.TemporaryDirectory() as root:
                with self.assertRaises(ValueError):compiler.receive(iter(blocks),root,'artifact',max_total=7,max_chunk=4)
                self.assertFalse(Path(root,'artifact.tar.gz').exists())


if __name__=='__main__':unittest.main()
