"""Frozen observed failures remain reproducible; active/adaptation errors fail."""
import hashlib,json
from pathlib import Path
import sys,tempfile,unittest
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tools import export_milestone as exporter


class FailedSourceExportTests(unittest.TestCase):
    def test_requires_pinned_original_failure_at_same_location(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(exporter,'ROOT',Path(tmp)):
            root=Path(tmp);path=root/'evidence/failed-001/source/execute.py';path.parent.mkdir(parents=True)
            raw=b"message='unfinished\n"
            receipt=path.parent.parent/'source-syntax-errors.json'
            receipt.write_text(json.dumps({'execute.py':dict(source_sha256=hashlib.sha256(raw).hexdigest(),line=1,message='unterminated string literal (detected at line 1)')}))
            exporter.validate_python(path,raw,raw.decode())
            with self.assertRaises(SyntaxError):exporter.validate_python(path,raw+b'\n',raw.decode())
            with self.assertRaises(SyntaxError):exporter.validate_python(path,raw,'\n'+raw.decode())
            with self.assertRaises(SyntaxError):exporter.validate_python(root/'tools/active.py',raw,raw.decode())

    def test_new_publication_corruption_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp,patch.object(exporter,'ROOT',Path(tmp)):
            root=Path(tmp);path=root/'evidence/failed-001/source/execute.py';path.parent.mkdir(parents=True)
            raw=b'message=1\n'
            (path.parent.parent/'source-syntax-errors.json').write_text(json.dumps({'execute.py':dict(source_sha256=hashlib.sha256(raw).hexdigest(),line=1,message='invented')}))
            with self.assertRaises(ValueError):exporter.validate_python(path,raw,"message='unfinished\n")
