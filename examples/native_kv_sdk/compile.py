"""Compile once, then inspect actual45-PE ELF coverage and real leased storage."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'core'))
from qwen38.elf_inventory import inventory, admit_wse3_sram
from cerebras.elf.cself import ELFMemory
from elf_symbols import symbols
from check_capture import PLACEMENTS


def actual_array(raw, name, size, alignment=4):
    found = {}
    for item in symbols(raw):
        canonical = item['name'].rsplit('$$', 1)[-1] if item['name'].startswith('$$csl_base_address$$') else item['name']
        if (canonical == name or (size == 4 and canonical == name + '.0')) and item['bytes'] > 0:
            found[item['address'], item['bytes']] = item
    if len(found) != 1:
        raise ValueError('One actual leased/exported array: ' + name)
    item = next(iter(found.values()))
    if item['bytes'] != size or item['address'] % alignment:
        raise ValueError('Exact aligned array extent: ' + name)
    return item


def main():
    if sorted(os.sched_getaffinity(0)) != [0] or (ROOT / 'out').exists():
        raise ValueError('CPU0 and one attempt required')
    width, height = 9, 5
    dims = width + 7, height + 2
    command = ['sdk_debug_shell', 'compile', 'layout.csl', '--arch=wse3',
               f'--fabric-dims={dims[0]},{dims[1]}', '--fabric-offsets=4,1',
               '-o=out', '--memcpy', '--channels=1', '--max-parallelism=1']
    subprocess.run(command, check=True, timeout=60)
    files = {}
    for path in sorted((ROOT / 'out').rglob('*')):
        if path.is_file():
            raw = path.read_bytes()
            files[str(path.relative_to(ROOT))] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    application = []
    covered = set()
    for path in sorted((ROOT / 'out/bin').glob('*.elf')):
        reader = ELFMemory(path)
        rectangles = [tuple(v) for v in reader.iter_rectangles()]
        if tuple(reader.get_fabric_dimensions()) != dims:
            raise ValueError('Actual fabric dimensions')
        coords = set()
        for x, y, w, h in rectangles:
            coords |= {(xx - 4, yy - 1) for xx in range(x, x + w) for yy in range(y, y + h)}
        if not coords or coords & covered or not coords <= set(PLACEMENTS):
            raise ValueError('Exact disjoint application placement')
        covered |= coords
        raw = path.read_bytes()
        gate = admit_wse3_sram(inventory(raw), 4096, 48128)
        arrays = {}
        for point in sorted(coords):
            role, index = PLACEMENTS[point]
            expected = {'evidence': (512, 4), 'packets.header_buffer': (4, 4), 'packets.receive_buffer': (124, 4)}
            if role in (1, 2, 3):
                expected['tx'] = 32, 4
            if role == 1:
                expected.update(qk_storage=(1032, 2), input_storage=(2056, 2))
                expected['hc.counts'] = 16, 2
            if role in (3, 4):
                expected['source_storage'] = 264, 2
            if role == 3:
                expected['rows.tx'] = 124, 4
            if role in (1, 5):
                expected['observer.buffer'] = 128, 4
            if role == 5:
                expected['observer.snapshot'] = 128, 4
            if role == 2 or (role == 4 and index == 3):
                expected['raw_word'] = 4, 4
            for name, (size, alignment) in expected.items():
                arrays[name] = actual_array(raw, name, size, alignment)
        values = list(arrays.values())
        for i, left in enumerate(values):
            for right in values[i + 1:]:
                if max(left['address'], right['address']) < min(left['address'] + left['bytes'], right['address'] + right['bytes']):
                    raise ValueError('Actual retained/packet/observer/source buffers overlap')
        application.append(dict(file=str(path.relative_to(ROOT)), rectangles=rectangles, sram=gate, arrays=arrays))
    if covered != set(PLACEMENTS) or not 1 <= len(application) <= 45:
        raise ValueError('All45 application PEs and bounded ELF inventory')
    result = dict(passed=all(item['sram']['passed'] for item in application), files=files,
                  application=application, geometry=dims, PEs=len(covered), runtime_created=False,
                  DSR_scope='Explicit source leases; concurrent implicit compiler use is additionally exercised by runtime, not proved by array symbols.')
    with (ROOT / 'compiled.json').open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    if not result['passed']:
        raise ValueError('Actual SRAM ceiling')


if __name__ == '__main__':
    main()
