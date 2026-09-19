"""Persistence interruption and boundedness checks; no SDK or neural test replay."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import capture_store as store


def main():
    start = time.monotonic()
    checks = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        label = 'epoch1_evidence'
        data = [[0xffffffff] * 128 for _ in range(45)]
        first = store.save_capture(root, label, data)
        saved = (root / first['path']).read_bytes()
        assert hashlib.sha256(saved).hexdigest() == first['sha256']
        assert json.loads(saved)['values'] == data
        checks.append('Preserve all32 bits including invalid16-bit host containers')
        for bad_label in ('../escape', 'epoch4_evidence'):
            try:
                store.save_capture(root, bad_label, data)
            except ValueError:
                pass
            else:
                raise AssertionError('Invalid capture label accepted')
        checks.append('Reject unplanned or escaping capture labels')
        try:
            store.save_capture(root, label, [[0] * 128 for _ in range(45)])
        except FileExistsError:
            pass
        else:
            raise AssertionError('Immutable raw capture overwritten')
        assert (root / first['path']).read_bytes() == saved
        checks.append('Duplicate save cannot overwrite original evidence')
        for function in ('fsync', 'link'):
            with patch.object(store.os, function, side_effect=OSError('injected interrupted publication')):
                try:
                    store.save_capture(root, 'epoch2_evidence', data)
                except OSError:
                    pass
                else:
                    raise AssertionError('Interrupted persistence accepted')
            assert not (root / 'captures/epoch2_evidence.json').exists()
            assert not list((root / 'captures').glob('*.pending'))
            assert (root / first['path']).read_bytes() == saved
        checks.append('Interrupted write or publication preserves earlier complete captures')
        store.save_progress(root, {label:first}, [], dict(copies=1), 11)
        old_progress = (root / 'progress.json').read_bytes()
        with patch.object(store.os, 'replace', side_effect=OSError('injected interruption')):
            try:
                store.save_progress(root, {label:first}, [], dict(copies=2), 13)
            except OSError:
                pass
            else:
                raise AssertionError('Interrupted progress replacement accepted')
        assert (root / 'progress.json').read_bytes() == old_progress
        assert (root / first['path']).read_bytes() == saved
        checks.append('Interrupted progress update leaves raw capture and previous progress intact')
        for next_label, (rows, columns) in store.SPECS.items():
            if next_label == label:
                continue
            store.save_capture(root, next_label, [[0xffffffff] * columns for _ in range(rows)])
        total = sum(p.stat().st_size for p in (root / 'captures').glob('*.json'))
        assert total <= store.TOTAL_CAPTURE_LIMIT
        assert len(list((root / 'captures').glob('*.json'))) == 13
        checks.append('All13 maximumuint32 captures fit cumulative1MiB and per-file128KiB')
    return dict(status='passed', checks=checks, worst_case_capture_bytes=total,
                seconds=time.monotonic()-start, SDK_imported_or_executed=False)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
