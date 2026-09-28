"""Host boundary and native-bit conventions for the complete MLP qualification."""
import importlib.util,sys,unittest,tempfile
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
spec=importlib.util.spec_from_file_location('run_layer_mlp',ROOT/'runtime/run_layer_mlp.py')
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


class MlpQualificationTests(unittest.TestCase):
    def test_bounded_bank_reads_preserve_bits_and_reject_file_changes(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'banks.npy';gold=np.arange(1234,dtype=np.uint32)*371
            np.save(path,gold);reader=driver.BankFile(path)
            self.assertEqual(reader.nbytes,gold.nbytes)
            np.testing.assert_array_equal(reader[13:1007],gold[13:1007])
            for part in [slice(-1,10),slice(0,1235),slice(0,3,2)]:
                with self.assertRaises(ValueError):reader[part]
            with path.open('ab') as f:f.write(b'bad')
            with self.assertRaises(ValueError):reader[0:3]

    def test_rectangular_copies_do_not_cross_holes_rows_or_element_extents(self):
        records=[dict(pe=[x,y],n=n) for x,y,n in [(4,2,16),(3,2,16),(1,2,16),(0,2,16),(0,3,16),(1,3,32)]]
        runs=list(driver.row_runs(records,lambda r:r['n']))
        self.assertEqual([[r['pe'] for r in run] for run in runs],[[[0,2],[1,2]],[[3,2],[4,2]],[[0,3]],[[1,3]]])

    def test_native_codes_equal_fp8_divided_by_256_in_half(self):
        codes=np.array([i for i in range(256) if (i&127)!=127],np.uint8)
        exponent=((codes&127)>>3).astype(np.int16);mantissa=(codes&7).astype(np.float64)
        truth=np.where(exponent==0,mantissa*2.**-9,(1+mantissa/8)*np.exp2(exponent-7))
        truth=np.copysign(truth,np.where(codes&128,-1.,1.))/256
        np.testing.assert_array_equal(driver.native_code(codes),truth.astype(np.float16).view(np.uint16))

    def test_physical_driver_uploads_only_weights_lut_setup_and_initial_boundary(self):
        import ast
        tree=ast.parse((ROOT/'runtime/run_layer_mlp.py').read_text());uploads=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Name) and node.func.id=='copy':
                if any(k.arg=='value' for k in node.keywords):uploads.add(node.args[0].value)
        self.assertEqual(uploads,{'bank','silu_lut','setup','quant_input'})
        code=(ROOT/'runtime/run_layer_mlp.py').read_text()
        self.assertIn("runner.launch('finish',nonblock=False)",code)


if __name__=='__main__':unittest.main()
