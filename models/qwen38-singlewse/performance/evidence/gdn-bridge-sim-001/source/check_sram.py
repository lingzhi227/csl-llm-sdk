"""Actual ELF SRAM plus complete coordinate coverage, including shared ELF images."""
import hashlib
import json
from pathlib import Path
from elf_inventory import inventory, admit_wse3_sram
from placement import placement


def cover(placements, application, offset):
    width, height = application; ox, oy = offset
    required = {(x, y) for x in range(ox, ox + width) for y in range(oy, oy + height)}
    actual = {}
    for name, rectangles in placements.items():
        for x, y, w, h in rectangles:
            if w <= 0 or h <= 0:
                raise ValueError('Empty load rectangle')
            for yy in range(y, y + h):
                for xx in range(x, x + w):
                    xy = (xx, yy)
                    if xy not in required or xy in actual:
                        raise ValueError('Application coverage out of bounds or overlapping: ' + str(xy))
                    actual[xy] = name
    if set(actual) != required:
        raise ValueError('Application PE coverage incomplete')
    return actual


def check(root, application=None, fabric=(762, 1172), offset=(4, 1)):
    root = Path(root)
    if application is None:
        application = json.loads((root / 'experiment.json').read_text())['application']
    records = []; placements = {}
    for path in sorted(root.glob('out/bin/*.elf')):
        raw = path.read_bytes(); name = str(path.relative_to(root))
        gate = admit_wse3_sram(inventory(raw), stack_allowance=4096, ceiling=48128)
        positions = placement(raw, fabric); placements[name] = positions['rectangles']
        records.append(dict(file=name, sha256=hashlib.sha256(raw).hexdigest(),
                            rectangles=positions['rectangles'], source=positions['source'], **gate))
    actual = cover(placements, application, offset)
    result = dict(passed=all(r['passed'] for r in records), application_pes=len(actual),
                  compiled_elf_images=len(records), application=list(application), fabric=list(fabric),
                  offset=list(offset), records=records,
                  note='A compiler ELF may cover several PEs. All complete low-memory PT_LOAD rectangles form the exact application once; SRAM is admitted for every mapped image.')
    (root / 'sram.json').write_text(json.dumps(result, indent=2) + '\n')
    if not result['passed']:
        raise ValueError('Actual SRAM limit failed')
    return result
