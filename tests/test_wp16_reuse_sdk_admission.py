"""Exact new SDK admission boundaries; no systemd/container/SDK calls."""
import copy
from contextlib import ExitStack, nullcontext
import importlib.util
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.wp16_reuse_sdk_admission import CANDIDATE, PROFILE, STEPS, GENERATED, application_elf_names, compare_compiled_states, validate, stage_limits, THREAD_ENV


ROOT=Path(__file__).resolve().parents[1]
HERE=ROOT/'examples/wp16/connected'
if not HERE.exists():HERE=ROOT
def load(name):
    spec=importlib.util.spec_from_file_location('connected_'+name+'_admission_test',HERE/(name+'.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


class ConnectedSDKAdmissionTests(unittest.TestCase):
    def test_runtime_generated_allowlist_does_not_relax_original_identity(self):
        frozen=dict(files={'out/bin/out_0_0.elf':dict(bytes=1,sha256='synthetic')},
                    compiled_bytes=1,footprints={},pes={},passed=True)
        actual=copy.deepcopy(frozen)
        actual['files'].update(copy.deepcopy(GENERATED));actual['compiled_bytes']+=54952
        self.assertEqual(compare_compiled_states(actual,frozen,after_runtime=True),GENERATED)
        with self.assertRaisesRegex(ValueError,'before simulator'):compare_compiled_states(actual,frozen)
        for name in ('out/bin/out_0_0.elf','out/generated/coord.elf'):
            bad=copy.deepcopy(actual);bad['files'][name]['sha256']='changed'
            with self.assertRaises(ValueError):compare_compiled_states(bad,frozen,after_runtime=True)
        bad=copy.deepcopy(actual);bad['files']['out/generated/unexpected.elf']=dict(bytes=1,sha256='other')
        with self.assertRaisesRegex(ValueError,'auxiliary'):compare_compiled_states(bad,frozen,after_runtime=True)

    def test_edge_ELFs_stay_distinct_from_exact_application_PE_set(self):
        application = {'out/bin/out_' + str(pe) + '_0.elf' for pe in range(4)}
        support = {'out/' + side + '/bin/out_' + str(pe) + '_0.elf'
                   for side, count in [('west', 3), ('east', 2)] for pe in range(count)}
        files = application | support | {'out/out.json', 'out/bin/out_rpc.json'}
        self.assertEqual(set(application_elf_names(files)), application)
        self.assertEqual(len(files), 11)
        with self.assertRaises(ValueError):
            application_elf_names(files - {'out/bin/out_1_0.elf'})
        for extra in ('out/bin/out_4_0.elf', 'out/bin/unexpected.elf'):
            with self.assertRaises(ValueError):
                application_elf_names(files | {extra})

    def admission(self):
        return dict(package='WP16', candidate=CANDIDATE, candidate_limit=1,
                    sdk_execution_authorized=True, manifest_sha256='test-only-not-real',
                    profile=copy.deepcopy(PROFILE), steps=copy.deepcopy(STEPS),
                    original_model_rerun_authorized=False, new_weight_preparation_authorized=False)

    def test_scope_resources_and_admission_are_exact(self):
        validate(self.admission(), 'test-only-not-real')
        for name, value in [('cpu_affinity', [0, 1]), ('simulation_seconds', 421),
                            ('swap_bytes', 1), ('input_indices', [0, 1]), ('generation_input_contraction', [[1,0,1],[2,1,2],[3,3,2]]), ('pes', 5),
                            ('full17408_down_output', True)]:
            altered = self.admission(); altered['profile'][name] = value
            with self.assertRaises(ValueError):
                validate(altered, 'test-only-not-real')
        for name, value in [('candidate_limit', 2), ('sdk_execution_authorized', False),
                            ('manifest_sha256', 'different'), ('new_weight_preparation_authorized', True)]:
            altered = self.admission(); altered[name] = value
            with self.assertRaises(ValueError):
                validate(altered, 'test-only-not-real')

    def test_reorder_or_command_change_cannot_bypass_serial_entry(self):
        for change in ('reverse', 'argv', 'time'):
            altered = self.admission()
            if change == 'reverse':
                altered['steps']['steps'].reverse()
            elif change == 'argv':
                altered['steps']['steps'][1]['argv'] = ['python', 'driver.py']
            else:
                altered['steps']['steps'][1]['seconds'] = 421
            with self.assertRaises(ValueError):
                validate(altered, 'test-only-not-real')

    def test_distinct_stage_limits_and_changed_profile_refused(self):
        self.assertEqual(stage_limits('compile'),{'memory.max':'1073741824','memory.swap.max':'0','pids.max':'128'})
        self.assertEqual(stage_limits('simulate'),{'memory.max':'536870912','memory.swap.max':'0','pids.max':'128'})
        for name in ('source','simulation','',None):
            with self.assertRaises(ValueError):stage_limits(name)
        for value in ({'compile':1073741824,'simulate':1073741824},
                      {'compile':536870912,'simulate':536870912},
                      {'compile':1073741824}, {'compile':1073741824,'simulate':536870912,'extra':1}):
            altered=self.admission();altered['profile']['memory_bytes_by_stage']=value
            with self.assertRaises(ValueError):validate(altered,'test-only-not-real')
        altered=self.admission();altered['profile']['memory_bytes']=1073741824
        with self.assertRaises(ValueError):validate(altered,'test-only-not-real')

    def test_host_entry_checks_actual_stage_limit_before_container_exec(self):
        entry=load('entry_sdk');original_read=Path.read_text
        for stage in ('compile','simulate'):
            for bad in (False,True):
                with self.subTest(stage=stage,bad=bad),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                    work=Path(folder);(work/'tmp').mkdir();unit='qwen38-job-synthetic.service'
                    actual=stage_limits(stage)
                    if bad:actual['memory.max']=stage_limits('simulate' if stage=='compile' else 'compile')['memory.max']
                    def read(path,*args,**kwargs):
                        if str(path)=='/proc/self/cgroup':return '0::/synthetic/'+unit+'\n'
                        if str(path).startswith('/sys/fs/cgroup/synthetic/'):
                            return actual[path.name]+'\n'
                        return original_read(path,*args,**kwargs)
                    stack.enter_context(patch.object(entry,'WORK',work));stack.enter_context(patch.object(Path,'cwd',return_value=work))
                    stack.enter_context(patch.object(Path,'read_text',read));stack.enter_context(patch.object(entry,'verify',return_value='synthetic'))
                    stack.enter_context(patch.object(entry,'image_stamp',return_value={}));stack.enter_context(patch.object(entry,'current',return_value=[0]))
                    compiled=stack.enter_context(patch.object(entry,'verify_compiled'))
                    launch=stack.enter_context(patch.object(entry.os,'execv'))
                    stack.enter_context(patch.dict(entry.os.environ,dict(THREAD_ENV,WP16_SDK_STAGE=stage,WP16_SDK_UNIT=unit),clear=True))
                    stack.enter_context(patch.object(sys,'argv',['entry_sdk.py',stage]))
                    if bad:
                        with self.assertRaisesRegex(ValueError,'Actual SDK hard limits'):entry.main()
                        launch.assert_not_called();compiled.assert_not_called()
                        self.assertFalse((work/('runtime-admission-'+stage+'.json')).exists())
                    else:
                        entry.main();launch.assert_called_once()
                        self.assertEqual(json.loads((work/('runtime-admission-'+stage+'.json')).read_text())['actual_limits'],actual)
                        self.assertEqual(compiled.call_count,int(stage=='simulate'))

    def test_container_entry_rejects_wrong_outer_memory_before_running_script(self):
        inner=load('container_entry')
        for stage in ('compile','simulate'):
            for bad in (False,True):
                with self.subTest(stage=stage,bad=bad),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                    work=Path(folder);unit='qwen38-job-synthetic.service';actual=stage_limits(stage)
                    if bad:actual['memory.max']=stage_limits('simulate' if stage=='compile' else 'compile')['memory.max']
                    (work/('runtime-admission-'+stage+'.json')).write_text(json.dumps(dict(source_manifest_sha256='synthetic',stage=stage,unit=unit,affinity=[0],actual_limits=actual)))
                    stack.enter_context(patch.object(inner,'WORK',work));stack.enter_context(patch.object(Path,'cwd',return_value=work))
                    stack.enter_context(patch.object(inner,'verify',return_value='synthetic'));stack.enter_context(patch.object(inner,'current',return_value=[0]))
                    compiled=stack.enter_context(patch.object(inner,'verify_compiled'))
                    execute=stack.enter_context(patch.object(inner.runpy,'run_path'))
                    stack.enter_context(patch.dict(inner.os.environ,dict(THREAD_ENV,TMPDIR='/tmp',WP16_SDK_STAGE=stage,WP16_SDK_UNIT=unit),clear=True))
                    stack.enter_context(patch.object(sys,'argv',['container_entry.py',stage]))
                    if bad:
                        with self.assertRaisesRegex(ValueError,'Matching host actual-resource'):inner.main()
                        execute.assert_not_called();compiled.assert_not_called()
                        self.assertFalse((work/('container-admission-'+stage+'.json')).exists())
                    else:
                        inner.main();execute.assert_called_once_with('compile_sdk.py' if stage=='compile' else 'driver.py',run_name='__main__')
                        self.assertEqual(json.loads((work/('container-exit-'+stage+'.json')).read_text())['status'],'passed')

    def test_supervisor_stage_properties_actual_mismatch_and_cleanup(self):
        guard=load('guard_sdk');original_read=Path.read_text
        for bad_stage in (None,'compile','simulate'):
            with self.subTest(bad_stage=bad_stage),tempfile.TemporaryDirectory() as folder,ExitStack() as stack:
                work=Path(folder);launched=[];stopped=set();active={}
                def ctl(*args):
                    if args[0]=='stop':stopped.add(args[1])
                    return ''
                def run(command,**kwargs):
                    unit=next(arg.split('=',1)[1] for arg in command if arg.startswith('--unit='))
                    stage=command[-1];launched.append((stage,command));active.update(unit=unit,stage=stage)
                    (work/('container-exit-'+stage+'.json')).write_text(json.dumps(dict(status='passed',affinity=[0])))
                    if stage=='simulate':
                        (work/'device-evidence').mkdir()
                        (work/'device-evidence/result.json').write_text(json.dumps(dict(status='connected_reuse_passed',normal_stop=True)))
                def properties(unit):
                    return dict(MainPID='0',ActiveState='inactive' if unit in stopped else 'active',SubState='exited',
                        Result='success',ExecMainStatus='0',ControlGroup='' if unit in stopped else '/synthetic/'+unit)
                def read(path,*args,**kwargs):
                    if str(path).startswith('/sys/fs/cgroup/synthetic/'):
                        actual=stage_limits(active['stage'])
                        if active['stage']==bad_stage:actual['memory.max']=stage_limits('simulate' if bad_stage=='compile' else 'compile')['memory.max']
                        return dict(actual,**{'memory.peak':'1024','memory.events':'max 0\noom 0\noom_kill 0\noom_group_kill 0\n','pids.events':'max 0\n'})[path.name]
                    return original_read(path,*args,**kwargs)
                for name,value in [('WORK',work),('verify',lambda work:'synthetic'),('image_stamp',lambda:{}),
                        ('heavy_lock',lambda path:nullcontext({})),('ctl',ctl),('properties',properties),
                        ('measure',lambda:(16*1024**3,64*1024**3,1024,1)),('verify_compiled',lambda *args,**kwargs:{}),
                        ('observe',lambda *args:dict(samples=[dict(pid=1,tid=1,start_ticks=1)],exited_during_observation=0))]:
                    stack.enter_context(patch.object(guard,name,value))
                stack.enter_context(patch.object(Path,'cwd',return_value=work));stack.enter_context(patch.object(Path,'read_text',read))
                stack.enter_context(patch.object(guard.os,'sched_getaffinity',return_value={0},create=True))
                stack.enter_context(patch.object(guard.subprocess,'run',run))
                self.assertEqual(guard.main(),0 if bad_stage is None else 1)
                report=json.loads((work/'supervisor.json').read_text())
                self.assertEqual(len(stopped),len(launched))
                self.assertEqual([name for name,_ in launched],['compile'] if bad_stage=='compile' else ['compile','simulate'])
                for stage,command in launched:
                    self.assertIn('MemoryMax='+stage_limits(stage)['memory.max'],command)
                    self.assertIn('RuntimeMaxSec='+str(300 if stage=='compile' else 420),command)
                    for field in ('MemorySwapMax=0','TasksMax=128','CPUAffinity=0','LimitFSIZE=2097152','LimitCORE=0'):self.assertIn(field,command)
                if bad_stage is not None:
                    self.assertIn('Runtime actual SDK limits changed',report['steps'][-1]['message'])
                    self.assertEqual(report['steps'][-1]['cleanup'],'owned_unit_quiescent')


if __name__ == '__main__':
    unittest.main()
