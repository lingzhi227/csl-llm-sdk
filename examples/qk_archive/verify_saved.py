"""Verify published synthetic captures without importing the SDK or allocating a device."""
from pathlib import Path
import hashlib
import json
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE/'qk-archive-hw-run-002'))
from check_capture import check_epoch, check_negative, require


def main():
    root = HERE.parents[1]/'evidence/qk-archive'
    result = json.loads((root/'result.json').read_bytes())
    summary = json.loads((root/'acceptance-summary.json').read_bytes())
    for name, digest in summary['public_receipt_hashes'].items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == digest, 'Receipt hash')
    values = {}
    require(len(result['capture_files']) == 13, 'Thirteen raw captures')
    for label, spec in result['capture_files'].items():
        relative = Path(spec['path'])
        require(not relative.is_absolute() and '..' not in relative.parts, 'Safe capture path')
        raw = (root/relative).read_bytes()
        require(len(raw) == spec['bytes'] and hashlib.sha256(raw).hexdigest() == spec['sha256'], 'Raw hash and size')
        record = json.loads(raw)
        require(record['label'] == label, 'Raw label')
        values[label] = record['values'] if record['shape'][0] > 1 else record['values'][0]
    for epoch in (1, 2):
        row = check_epoch(*(values[f'epoch{epoch}_{name}'] for name in
            ('evidence','archive','reference','workspace','output','baseline_output')), epoch)
        require(row == result['epochs'][epoch-1], 'Published epoch result')
    require(values['epoch1_archive'][704:] != values['epoch2_archive'][704:] and
            values['epoch1_output'] != values['epoch2_output'], 'Reset changed actual values')
    require(check_negative(values['negative_evidence'], values['epoch2_evidence']) == result['negative'], 'Negative result')
    journal = [json.loads(line) for line in (root/'journal.jsonl').read_text().splitlines()]
    require(len(journal) == 43 and [v['sequence'] for v in journal] == list(range(43)), 'Complete journal sequence')
    require(journal[-2]['kind'] == 'stop_enter' and journal[-1]['kind'] == 'stop_exit', 'Normal stop recorded')
    require(result['status'] == 'passed' and result['normal_stop'], 'Passing final runtime')
    print(json.dumps(dict(status='saved_capture_verification_passed', raw_files=13,
        resets=2, archive_words_per_reset=960, output_halfwords_per_reset=256,
        poisoned_words_per_reset=1280, SDK_invocations=0, hardware_invocations=0)))


if __name__ == '__main__':
    main()
