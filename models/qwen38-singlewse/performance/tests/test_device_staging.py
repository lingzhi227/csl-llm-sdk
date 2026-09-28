"""Regression for the observed attempt-name shadowing during device adaptation."""
import contextlib,io,json,subprocess,sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'tools')]
import stage_mixer_resource_compile as stager

class DeviceStagingTests(unittest.TestCase):
 def test_selected_role_name_never_replaces_remote_attempt_identity(self):
  frozen=ROOT/'evidence/layer-backend-compile-014/source'
  files={p.name:p.read_bytes() for p in frozen.iterdir() if p.is_file()}
  profiles=json.loads(files['profiles.json'])['samples'];calls=[]
  def run(args,**kwargs):
   calls.append(args);return subprocess.CompletedProcess(args,0,stdout='',stderr='')
  with tempfile.TemporaryDirectory() as directory:
   root=Path(directory);(root/'performance/evidence').mkdir(parents=True)
   for relative in ['csl/device_control.csl','spatial/device_control.py']:
    target=root/'performance'/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/relative).read_bytes())
   with patch.object(stager,'ROOT',root),patch.object(stager,'build',return_value=(files,9,1,profiles)),patch.object(stager.subprocess,'run',side_effect=run),patch.object(sys,'argv',['stage_mixer_resource_compile.py','999','--device-control']),contextlib.redirect_stdout(io.StringIO()):
    stager.main()
   dispatch=json.loads((root/'performance/evidence/layer-backend-compile-999/dispatch.json').read_text())
   self.assertEqual(dispatch['remote'],'/srv/model-storage/qwen38-singlewse/runs/layer-backend-compile-999')
   self.assertEqual(dispatch['unit'],'qwen38-single-layer-backend-compile-999')
   self.assertIn(dispatch['remote'],calls[1][-1]);self.assertIn('--unit='+dispatch['unit'],calls[2][-1])
   self.assertNotIn('mixer_layer_norm_sender.csl',calls[2][-1])

if __name__=='__main__':unittest.main()
