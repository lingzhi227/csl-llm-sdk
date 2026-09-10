import importlib.util
from pathlib import Path
import tempfile
import unittest
spec = importlib.util.spec_from_file_location('sdk_container', Path(__file__).resolve().parents[1]/'tools/sdk_container.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContainerTests(unittest.TestCase):
    def test_parent_mount_precedes_tmp_and_container_tmp_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            work=Path(directory); image=work/'sdk.sif'; image.touch()
            cmd=module.command(image, work, ['python','driver.py'])
            self.assertLess(cmd.index('--bind='+str(work.resolve())+':'+str(work.resolve())),
                            cmd.index('--bind='+str(work.resolve()/'tmp')+':/tmp'))
            self.assertIn('--env=TMPDIR=/tmp',cmd)
            self.assertEqual(cmd[-2:], ['python','driver.py'])
