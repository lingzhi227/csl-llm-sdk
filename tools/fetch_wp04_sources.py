"""Fetch two pinned upstream text files; bounded bytes, time and verified hashes."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
from prepare_wp04 import SOURCES

COMMIT = '4815a0a6a064214f2d8208c094464a5a6b76ca8d'
PATHS = {'modeling_qwen3_5.py': 'models/qwen3_5/modeling_qwen3_5.py',
         'import_utils.py': 'utils/import_utils.py'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    destination = parser.parse_args().destination
    destination.mkdir(parents=True, exist_ok=False)
    receipt = {'status': 'started', 'commit': COMMIT, 'files': [], 'payload_bytes': 0}
    try:
        for name, digest in SOURCES.items():
            url = f'https://raw.githubusercontent.com/huggingface/transformers/{COMMIT}/src/transformers/{PATHS[name]}'
            with urllib.request.urlopen(url, timeout=30) as response:
                content = response.read(512*1024+1)
            receipt['payload_bytes'] += len(content)
            if len(content) > 512*1024 or hashlib.sha256(content).hexdigest() != digest:
                raise ValueError('Oversized or unverified upstream source: '+name)
            content.decode('utf-8')
            (destination/name).write_bytes(content)
            receipt['files'].append({'name': name, 'url': url, 'bytes': len(content), 'sha256': digest})
        receipt['status'] = 'complete'
    except BaseException as error:
        receipt.update(status='failed', error=str(error))
        raise
    finally:
        (destination/'acquisition.json').write_text(json.dumps(receipt, indent=2)+'\n')
