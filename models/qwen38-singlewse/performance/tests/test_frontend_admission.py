"""Reject out-of-scope component profiles before compiler/wafer allocation."""
import ast
import copy
from pathlib import Path
import unittest


class FrontendAdmissionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the actual pure gate without importing SDK/site execution modules.
        source=Path(__file__).resolve().parents[1]/'runtime/compile_component.py'
        tree=ast.parse(source.read_text())
        gate=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='admitted_component')
        backend=ast.parse((source.parent/'backend.py').read_text())
        recurrence=next(n for n in backend.body if isinstance(n,ast.FunctionDef) and n.name=='gdn_columns_profile')
        namespace={};exec(compile(ast.Module(body=[recurrence,gate],type_ignores=[]),str(source),'exec'),namespace)
        cls.admit=staticmethod(namespace['admitted_component'])
        cls.profile=dict(application=[16,2],application_pes=32,original_frontend=True,
                         frontend_shape=[16,3,128],revision='017b9c7af6b5689d5dd426a76e0bc077eb5ca20a',full_model=False)

    def test_exact_profile_and_existing_component(self):
        self.assertTrue(self.admit(self.profile))
        self.assertTrue(self.admit(dict(application=[4,5],application_pes=20)))

    def test_missing_or_expanded_identity_rejected(self):
        replacements=dict(application=[16,3],application_pes=33,original_frontend=False,
                          frontend_shape=[16,4,128],revision='unverified',full_model=True)
        for key,value in replacements.items():
            with self.subTest(field=key):
                candidate=copy.deepcopy(self.profile);candidate[key]=value
                self.assertFalse(self.admit(candidate))
                del candidate[key];self.assertFalse(self.admit(candidate))


if __name__=='__main__':unittest.main()
