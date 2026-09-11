"""Validate actual SDK-loaded rectangles before constructing a simulator.

The reader is SDK2.10.1 cerebras.elf.cself.ELFMemory. Its iter_rectangles
uses native ELFObject.iterate_segments, the same placement source used by
SdkElfGenerator. Filename suffixes do not encode application coordinates.
"""
from itertools import islice

from qwen38.resident_spatial import TILES


def inspect_placement(work, elf_memory_type):
    """Require exactly the eight expected singleton placements in an 11x4 fabric.

    ELFMemory documents possible repeated segment rectangles. Identical
    rectangles within one ELF are allowed; overlap between ELFs is not.
    This is metadata validation, not proof that fabric execution completed.
    """
    rows = []
    occupied = set()
    for tile in TILES:
        name = f'out/bin/out_{tile.rank}_0.elf'
        reader = elf_memory_type(work / name)
        dimensions = tuple(reader.get_fabric_dimensions())
        if dimensions != (11, 4):
            raise ValueError(f'Unexpected ELF fabric dimensions: {name}: {dimensions}')
        raw = list(islice(reader.iter_rectangles(), 65))
        if not raw or len(raw) > 64:
            raise ValueError(f'Bounded nonempty ELF segment rectangles required: {name}')
        rectangles = {tuple(rect) for rect in raw}
        expected = (tile.x + 4, tile.y + 1, 1, 1)
        if rectangles != {expected}:
            raise ValueError(f'Wrong ELF placement: {name}: {sorted(rectangles)} != {expected}')
        coordinate = expected[:2]
        if coordinate in occupied:
            raise ValueError('Application ELF placements overlap')
        occupied.add(coordinate)
        rows.append(dict(path=name, rank=tile.rank, role=tile.role,
                         application_xy=[tile.x, tile.y], fabric_xy=list(coordinate),
                         segment_rectangles=[list(r) for r in sorted(rectangles)],
                         segment_rectangle_count=len(raw)))
    if occupied != {(x, y) for y in (1, 2) for x in range(4, 8)}:
        raise ValueError('Incomplete 4x2 application ELF coverage')
    return dict(status='actual_SDK_ELF_placement_checked', fabric=[11, 4],
                application=[4, 2], offset=[4, 1], files=rows,
                reader='cerebras.elf.cself.ELFMemory.iter_rectangles',
                filename_coordinates_assumed=False, simulator_constructed=False)
