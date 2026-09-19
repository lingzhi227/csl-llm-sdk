"""Persist each completed raw D2H before a later SDK call can block or time out."""
import hashlib
import json
import os
from pathlib import Path

SPECS = {f'epoch{epoch}_{kind}': shape for epoch in (1, 2, 3)
         for kind, shape in [('evidence', (45, 128)), ('qk', (6, 516)),
                             ('input', (6, 1028)), ('native_sources', (4, 132))]}
SPECS['negative_evidence'] = (45, 128)
CAPTURE_LIMIT = 131072
PROGRESS_LIMIT = 16384
TOTAL_CAPTURE_LIMIT = 1048576


def _sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _atomic(path, raw, replace=False):
    temporary = path.with_name(path.name + '.pending')
    # Exclusive creation fails closed after an interrupted or duplicate write.
    with temporary.open('xb') as stream:
        try:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink()
            raise
    try:
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        _sync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def save_capture(root, label, values):
    if label not in SPECS:
        raise ValueError('Known immutable capture label required')
    rows, columns = SPECS[label]
    if len(values) != rows or any(len(row) != columns for row in values):
        raise ValueError('Actual D2H shape required')
    # Preserve every bit of the uint32 host container, including bad halfwords.
    if any(type(value) is not int or not 0 <= value <= 0xffffffff
           for row in values for value in row):
        raise ValueError('Unsigned32 host containers required')
    raw = (json.dumps(dict(label=label, shape=[rows, columns], values=values),
                      separators=(',', ':')) + '\n').encode()
    if len(raw) > CAPTURE_LIMIT:
        raise ValueError('Bounded single raw capture')
    directory = Path(root) / 'captures'
    directory.mkdir(exist_ok=True)
    if directory.is_symlink():
        raise ValueError('Owned capture directory required')
    existing = list(directory.glob('*.json'))
    if len(existing) >= len(SPECS) or sum(p.stat().st_size for p in existing) + len(raw) > TOTAL_CAPTURE_LIMIT:
        raise ValueError('Bounded complete capture inventory')
    path = directory / (label + '.json')
    _atomic(path, raw)
    return dict(path=str(path.relative_to(root)), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def save_progress(root, captures, epochs, counts, journal_events):
    raw = (json.dumps(dict(status='incomplete', captures=captures, epochs=epochs,
            counts=counts, journal_events=journal_events), separators=(',', ':')) + '\n').encode()
    if len(raw) > PROGRESS_LIMIT:
        raise ValueError('Bounded partial progress receipt')
    _atomic(Path(root) / 'progress.json', raw, replace=True)
