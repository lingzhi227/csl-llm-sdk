"""Independently audit retained synthetic bridge captures and exact cohost identity."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.build_frontend_stage import verified_frozen


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def audit(attempt, output):
    if not re.fullmatch(r'gdn-bridge-sim-\d{3}', attempt) or output.exists():
        raise ValueError('A new audit of a frozen bridge simulation is required')
    folder = ROOT/'evidence'/attempt
    manifest = json.loads((folder/'source-manifest.json').read_text())['files']

    def source(name):
        return verified_frozen(folder/'source'/name, manifest[name])

    plan = json.loads(source('probe-plan.json'))
    result_raw = (folder/'result.json').read_bytes()
    result = json.loads(result_raw)
    assert result['passed'] and result['normal_stop'] and result['synthetic_wire']
    assert not result['physical'] and not result['neural_execution'] and not result['full_model']
    assert not plan.get('diagnostic_simprint', False)
    assert not plan.get('diagnostic_microthread_change', False)
    assert json.loads((folder/'workstation-release.json').read_text())['workstation_released']
    assert json.loads((folder/'COMPLETE.json').read_text())['passed']
    assert json.loads((folder/'sram.json').read_text())['passed']

    resource = ROOT/'evidence'/plan['resource_compile']
    rm = json.loads((resource/'source-manifest.json').read_text())['files']
    def resource_source(name):
        return verified_frozen(resource/'source'/name, rm[name])
    samples = json.loads(resource_source('profiles.json'))['samples']
    matches = [s for s in samples if s['original_pe'] == plan['original_cohost']
               and s['variant'] == 'candidate']
    assert len(matches) == 1
    sample = matches[0]
    parameters = sample['parameters']
    frames, workers, group = (parameters[k] for k in
                             ('bridge_frames', 'bridge_workers', 'bridge_group'))
    original = resource_source(sample['source'])
    wrappers = b'''\nfn probe_arm() void {bridge_arm();}
fn probe_send() void {sys.unblock_cmd_stream();}
fn probe_wait() void {sys.unblock_cmd_stream();}
comptime {@export_symbol(probe_arm);@export_symbol(probe_send);@export_symbol(probe_wait);}
'''
    if plan.get('direct_fabric_streams', False):
        assert plan['explicit_invocation_argument']
        assert plan['resource_compile'] == 'layer-backend-compile-045'
        assert parameters.get('bridge_switch', False) is False
        assert b'param bridge_switch:bool=false;' in original
        wrappers = wrappers.replace(b'fn probe_arm() void {bridge_arm();}',
                                    b'fn probe_arm(token:u32) void {bridge_stream_arm(token);}')
        segments = list(map(int, parameters['bridge_segments'].split('{')[1].rstrip('}').split(',')))
        assert segments == plan['segments'] and len(segments) == workers and sum(segments) == frames
    assert source(sample['source']) == original + wrappers
    assert sha(original) == plan['actual_cohost_source_sha256']
    assert plan['bank_words'] == parameters['bank_words'] == result['original_bank_words']
    layout = source('layout.csl').decode()
    declaration = next(s for s in layout.splitlines() if '@set_tile_code(1,0,' in s)
    for name, value in parameters.items():
        value = str(value).lower() if not isinstance(value, str) else value
        assert re.search(r'\.' + re.escape(name) + '=' + re.escape(value) + r'[,}]', declaration)

    dispatch = json.loads((folder/'dispatch.json').read_text())
    remote = '/srv/model-storage/qwen38-singlewse/runs/' + attempt
    assert dispatch['remote'] == remote
    # Only small, already retained captures are read. No compiler/runtime job,
    # model weights, new simulations, or hardware resources are allocated.
    script = 'root=' + repr(remote) + '\n' + r'''
from pathlib import Path
import hashlib,json
import numpy as np
r=Path(root);result_raw=(r/'result.json').read_bytes();result=json.loads(result_raw)
p=r/'actual.npz';assert p.stat().st_size<131072
assert hashlib.sha256(p.read_bytes()).hexdigest()==result['capture_sha256']
with np.load(p,allow_pickle=False)as a:
 assert set(a.files)=={prefix+str(t)for t in (1,2,99,100)for prefix in ('received_','bridge_')}
 all_words=0
 for run in result['invocations']:
  token=run['token'];n=run['sink'][1];workers=run['sink'][3];group=GROUP
  actual=a['received_'+str(token)];assert actual.shape==(n,5)and actual.dtype==np.uint32
  # Elementwise construction independent of the vectorized runner oracle.
  expected=np.array([[token,384*group+2*i,2,(token*104729)^i^0x3f807e11,5]
                     for i in range(n)],dtype=np.uint32)
  np.testing.assert_array_equal(actual,expected)
  np.testing.assert_array_equal(a['bridge_'+str(token)],[0,0,0,9,n,workers,token,1])
  assert run['bridge']==a['bridge_'+str(token)].tolist()
  assert run['source'][:4]==[1,5*n,workers,1]and run['source'][4]>0
  assert run['sink'][:4]==[1,n,3,workers]
  all_words+=actual.size
 print(json.dumps(dict(passed=True,result_sha256=hashlib.sha256(result_raw).hexdigest(),
   capture_sha256=result['capture_sha256'],exact_retained_return_words=all_words,
   local_callback_snapshots=4,early_return_words=[v['source'][4]for v in result['invocations']])))
'''.replace('GROUP', str(group))
    capture = json.loads(subprocess.check_output(
        ['ssh', 'workstation', '/usr/bin/python3 -c ' + shlex.quote(script)],
        text=True, timeout=30))
    assert capture['result_sha256'] == sha(result_raw)
    assert capture['exact_retained_return_words'] == result['exact_return_words'] == 4*5*frames
    assert result['exact_forward_words'] == 4*1161
    assert result['raw_markers_received'] == 4*workers
    assert result['weighted_markers_emitted'] == 4
    assert [v['token'] for v in result['invocations']] == [1, 2, 99, 100]
    report = dict(passed=True, attempt=attempt, group=group, frames=frames, workers=workers,
                  bank_words=parameters['bank_words'], original_cohost=sample['original_pe'],
                  compiled_cohost_sha256=sha(original), compiled_cohost_exact=True,
                  source_manifest_sha256=sha((folder/'source-manifest.json').read_bytes()),
                  capture=capture, physical=False, neural_execution=False, full_model=False,
                  direct_fabric_streams=plan.get('direct_fabric_streams', False),
                  switch_parent_executed=False,
                  scope='Independent retained-return/status audit; forward words and bank sentinels '
                        'were checked by the source-bound runner but are not retained in actual.npz. '
                        'Three diagnostic RPC wrappers; no original neural execution or speed acceptance.')
    output.write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('attempt')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    audit(args.attempt, args.output)
