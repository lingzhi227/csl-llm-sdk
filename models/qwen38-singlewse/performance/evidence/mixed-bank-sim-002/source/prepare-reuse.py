import hashlib,json
from pathlib import Path
p=json.loads(Path("fixture-source.json").read_text());m=json.loads(Path("fixture.json").read_text());assert hashlib.sha256(Path("fixture.npz").read_bytes()).hexdigest()==p["fixture_sha256"]==m["fixture_sha256"]
print("Verified exact previously frozen fixture")
