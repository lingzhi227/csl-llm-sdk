"""Actual native ELF metadata for new physical geometry, no runtime creation."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "core"))
from cerebras.elf.cself import ELFMemory
from qwen38.resident_spatial import TILES
from hw00.store import atomic_json

rows = []
for tile in TILES:
    name = f"out/bin/out_{tile.rank}_0.elf"
    reader = ELFMemory(ROOT / "compiled" / name)
    if tuple(reader.get_fabric_dimensions()) != (762, 1172):
        raise ValueError("Hardware fabric mismatch")
    rectangles = list(reader.iter_rectangles())
    if not 0 < len(rectangles) <= 64 or {tuple(r) for r in rectangles} != {(tile.x+4, tile.y+1, 1, 1)}:
        raise ValueError("Unexpected physical placement: " + name)
    rows.append(dict(file=name, rank=tile.rank, rectangles=rectangles))
atomic_json(ROOT / "placement.json", dict(passed=True, geometry=[762, 1172], files=rows, runtime_created=False))
