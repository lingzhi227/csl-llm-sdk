"""Create a small synthetic slab for offline reproduction; never model evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def create(destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    # Exactly representable finite BF16 values; vary both axes and include zeros.
    raw = bytearray()
    for row in range(128):
        for col in range(5120):
            value = ((row * 13 + col * 7) % 31 - 15) / 128
            bits = struct.unpack('<I', struct.pack('<f', value))[0]
            assert bits & 65535 == 0
            raw.extend(struct.pack('<H', bits >> 16))
    (destination / 'slab-row-major.bf16').write_bytes(raw)
    receipt = {
        'status': 'complete', 'kind': 'synthetic_reproduction_fixture',
        'scope': 'locally generated; no original model identity or model acceptance',
        'shape': [128, 5120], 'dtype': 'BF16', 'payload_bytes': len(raw),
        'slab_sha256': hashlib.sha256(raw).hexdigest(), 'network_requests': 0,
    }
    (destination / 'download.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return destination


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    print(create(parser.parse_args().destination))
