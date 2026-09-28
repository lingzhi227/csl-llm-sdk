"""The parent must bound post-load stalls, including a blocked SDK call."""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
sys.path.insert(0,str(ROOT.parent))
from progress import ProgressDeadline,Progress
import progress_lifecycle


class ProgressTests(unittest.TestCase):
    def test_loading_uses_outer_limit_then_each_phase_has_its_own_limit(self):
        with tempfile.TemporaryDirectory() as temp:
            now=[0];guard=ProgressDeadline(temp,os.getpid(),30,lambda:now[0])
            guard.check();now[0]=400;guard.check()
            reporter=Progress(True,Path(temp)/'progress.json')
            reporter.mark('initialize');guard.check();now[0]+=29;guard.check()
            reporter.mark('finish',1);guard.check();now[0]+=30
            with self.assertRaisesRegex(TimeoutError,'finish'):guard.check()

    def test_foreign_regressed_missing_and_oversized_records_fail_closed(self):
        for change in ('foreign','regressed','missing','oversized'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as temp:
                p=Path(temp)/'progress.json';reporter=Progress(True,p)
                guard=ProgressDeadline(temp,os.getpid(),30,lambda:0)
                reporter.mark('initialize');reporter.mark('arm',1);guard.check()
                if change=='missing':p.unlink()
                elif change=='oversized':p.write_text(' '*4097)
                else:
                    record=json.loads(p.read_text())
                    if change=='foreign':record['pid']+=1
                    else:record['sequence']-=1
                    p.write_text(json.dumps(record))
                with self.assertRaises(ValueError):guard.check()

    def test_unbounded_limits_and_boolean_limits_rejected(self):
        for value in (None,True,0,9,61,30.0):
            with self.assertRaises(ValueError):ProgressDeadline('.',os.getpid(),value)

    def test_stalled_child_is_terminated_and_correlated_cleanup_runs(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'experiment.json').write_text(json.dumps(dict(complete_original_mlp=True,run_progress_timeout_seconds=10)))
            (root/'child.py').write_text(
                "import json,os,time\nfrom pathlib import Path\n"
                "Path('run.jobs').write_text('wsjob-testprogress\\n')\n"
                "Path('progress.json').write_text(json.dumps(dict(pid=os.getpid(),sequence=1,phase='finish')))\n"
                "time.sleep(60)\n")
            cleanup=progress_lifecycle.cleanup;released=[];children=[];now=[0]
            def clock():now[0]+=11;return now[0]
            def release(r,name,ids,**kwargs):
                released.extend(ids)
                return [dict(id=i,phase='CANCELLED',released=True,cancelled=True) for i in ids]
            def clean(r,name,baseline,proc):
                children.append(proc)
                return cleanup(r,name,baseline,proc,query_fn=lambda *a:dict(items=[]),release_fn=release)
            with patch.object(progress_lifecycle,'query',lambda *a:dict(items=[])), \
                 patch.object(progress_lifecycle,'cleanup',clean), \
                 patch.object(progress_lifecycle,'child_limits',lambda:None), \
                 patch.object(progress_lifecycle,'ProgressDeadline',lambda r,p,s:ProgressDeadline(r,p,s,clock)):
                with self.assertRaisesRegex(RuntimeError,'Stage failed'):
                    progress_lifecycle.stage(root,'run','child.py',60)
            self.assertIsNotNone(children[0].poll())
            self.assertEqual(released,['wsjob-testprogress'])
            receipt=json.loads((root/'run-audit.json').read_text())
            self.assertEqual(receipt['error'],'TimeoutError: Post-load progress deadline: finish')
            self.assertFalse(receipt['cleanup_errors'])


if __name__=='__main__':unittest.main()
