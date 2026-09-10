"""Stream the metadata-pinned selected original attention head; never retry."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'core'))
from qwen38.ranged import RangeReader


def download(plan_path, output):
    plan = json.loads(plan_path.read_text())
    if (plan['model'], plan['revision'], plan['shard'], plan['total_size']) != (
        'Qwen/Qwen3.8-27B', '1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0',
        'model-00001-of-00018.safetensors', 3966730552):
        raise ValueError('Pinned source identity mismatch')
    expected = [('q-with-norm.bf16', 2907879576, 5243392),
                ('k-with-norm.bf16', 2834478744, 2621952),
                ('v-row-major.bf16', 3033709208, 2621440)]
    if [(r['file'], r['start'], r['length']) for r in plan['ranges']] != expected:
        raise ValueError('Audited selected-head mapping mismatch')
    url = f"https://huggingface.co/{plan['model']}/resolve/{plan['revision']}/{plan['shard']}"
    if plan['url'] != url or plan['expected_payload_bytes'] != 10486784:
        raise ValueError('Payload identity mismatch')
    output.mkdir(parents=True, exist_ok=False)
    reader = RangeReader(url, plan['total_size'], budget=16*1024*1024)
    report = {'status': 'started', 'plan': plan,
              'plan_sha256': hashlib.sha256(plan_path.read_bytes()).hexdigest(), 'files': {}}
    try:
        for item in plan['ranges']:
            digest = hashlib.sha256()
            partial = output/(item['file']+'.part')
            with partial.open('xb') as stream:
                for offset in range(0, item['length'], 1024*1024):
                    chunk = reader.read(item['start']+offset, min(1024*1024, item['length']-offset))
                    stream.write(chunk)
                    digest.update(chunk)
            partial.rename(output/item['file'])
            report['files'][item['file']] = {'bytes': item['length'], 'sha256': digest.hexdigest()}
        if reader.received != 10486784:
            raise ValueError('Total received bytes mismatch')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=str(exc))
        raise
    finally:
        report.update(payload_bytes_received=reader.received, payload_bytes_reserved=reader.reserved,
                      ranges=reader.receipts)
        (output/'download.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    download(args.plan, args.output)
