"""Fault checks for runtime auxiliary lifetime gating; no SDK or ELF payloads."""
import hashlib
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch
import reuse_gate


def main():
    start = time.monotonic()
    names = sorted(reuse_gate.AUXILIARY)
    rejected = []
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / 'out/generated').mkdir(parents=True)
        (root / 'out/base').write_bytes(b'base')
        base = {'out/base'}
        reuse_gate.check_auxiliary(root, base, 'absent')

        def reject(label, phase):
            try:
                reuse_gate.check_auxiliary(root, base, phase)
            except ValueError:
                rejected.append(label)
            else:
                raise AssertionError(label)

        reject('Missing final generated files', 'exact')
        (root / names[0]).write_bytes(b'partial')
        reuse_gate.check_auxiliary(root, base, 'live')
        reject('Existing generated file before first load', 'absent')
        (root / names[0]).write_bytes(b'x' * (reuse_gate.AUXILIARY[names[0]]['bytes'] + 1))
        reject('Oversized live auxiliary', 'live')
        (root / names[0]).unlink()
        unexpected = root / 'out/unexpected'
        unexpected.write_bytes(b'x')
        reject('Unexpected generated filename', 'live')
        unexpected.unlink()
        (root / names[0]).symlink_to(root / 'out/base')
        reject('Auxiliary symlink', 'live')
        (root / names[0]).unlink()
        for name in names:
            (root / name).write_bytes(b'wrong')
        reject('Incorrect final identity', 'exact')
        specs = {name:dict(bytes=5,sha256=hashlib.sha256(b'valid').hexdigest()) for name in names}
        for name in names:
            (root / name).write_bytes(b'valid')
        with patch.object(reuse_gate, 'AUXILIARY', specs):
            reuse_gate.check_auxiliary(root, base, 'exact')
        (root / 'out/base').unlink()
        reject('Missing immutable compiled base', 'live')
    return dict(status='passed', rejected=rejected, positive_lifetimes=['absent','partial_live','exact_synthetic'],
                seconds=time.monotonic()-start, SDK_or_ELF_payload_read=False)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2))
