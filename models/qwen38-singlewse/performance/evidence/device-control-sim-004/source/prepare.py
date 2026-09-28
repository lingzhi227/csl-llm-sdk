"""Read one unchanged full original P31 bank from the qualified remote fixture."""
import hashlib,json
from pathlib import Path
import numpy as np

config=json.loads(Path('peer.json').read_text());old=Path(config['original_banks_path'])
h=hashlib.sha256()
with old.open('rb') as f:
 for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
assert h.hexdigest()==config['original_banks_sha256']
original=np.load(old,mmap_mode='r',allow_pickle=False);r=config['bank_record']
bank=np.array(original[r['offset']:r['offset']+r['words']],dtype=np.uint32)
assert bank.size==config['parameters']['bank_words'];del original
setup=np.array(config['setup'],np.uint16)
with Path('fixture.npz').open('xb') as f:np.savez(f,bank=bank,setup=setup)
Path('fixture.json').write_text(json.dumps(dict(original_pe=r['pe'],original_bank_words=r['words'],
 original_bank_extent_unchanged=True,original_fixture_sha256=h.hexdigest(),bank_sha256=hashlib.sha256(bank.tobytes()).hexdigest(),
 fixture_sha256=hashlib.sha256(Path('fixture.npz').read_bytes()).hexdigest(),physical=False,
 scope='Full original selected bank initialization/readback and credited device commands. No neural/fabric projection or model throughput acceptance.'),indent=2)+'\n')
