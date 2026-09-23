"""Post-release actual raw views, with no reference-fed intermediate input."""
from pathlib import Path
import copy
import hashlib
import stat
from runtime_boundary import require


def box(item):
    return [item[k] for k in ('x', 'y', 'width', 'height', 'count')]


class RawEvidence:
    def __init__(self, root, capture):
        self.root, self.capture, self.read_bytes = Path(root), capture, 0
        require(capture['normal_stop'] is True and
                capture['completed'] == ([1, 2, 3] if capture['mode'] == 'baseline' else [1]),
                'Complete normally stopped physical schedule')

    def raw(self, name):
        import numpy as np
        r = self.capture['raw_files'][name]
        require(Path(r['path']).parts == ('raw', name), 'Exact raw evidence path')
        path = self.root/r['path']
        require(not path.is_symlink() and not path.parent.is_symlink(), 'No raw evidence symlink')
        before = path.stat()
        require(stat.S_ISREG(before.st_mode) and before.st_size == r['bytes'] and
                0 < r['bytes'] <= 16 << 20, 'Bounded regular native raw file')
        raw = path.read_bytes()
        after = path.stat()
        require((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) ==
                (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns) and
                hashlib.sha256(raw).hexdigest() == r['sha256'], 'Actual raw hash and stable identity')
        self.read_bytes += len(raw)
        x, y, w, h, n = r['rectangle']
        require(0 <= x < x+w <= 119 and 0 <= y < y+h <= 1160 and n > 0, 'Application raw geometry')
        value = np.frombuffer(raw, dtype={'u16': '<u2', 'u32': '<u4', 'f32': '<f4'}[r['dtype']])
        require(value.size == w*h*n, 'Exact native raw extent')
        return value.reshape(h, w, n)

    def checked(self, name, item):
        r = self.capture['raw_files'][name]
        require(r['rectangle'] == box(item) and r['symbol'] == item['symbol'] and
                r['dtype'] == item['dtype'], 'Exact independently scheduled raw identity')
        return self.raw(name)


class Evidence(RawEvidence):
    """One original layer at a time; retain global raw file ordinals."""
    def __init__(self, root, capture, host, lowered, layer):
        super().__init__(root, capture)
        require(capture['mode'] == 'baseline' and not capture['mathematical_audit_passed'],
                'Unmodified baseline evidence for conditional operator audit')
        self.layer, self.index, self.upstream = layer, {}, None
        regions = {r['layer']: r for r in lowered['regions']}
        region = regions[layer]
        edge = next((e for e in lowered['hidden_edges'] if e['destination_layer'] == layer), None)
        source_point = None if edge is None else tuple(edge['source'])
        for ordinal, item in enumerate(host['copies']['diagnostic_copies']):
            if item['symbol'] == 'hidden' and source_point is not None:
                x, y = source_point
                if item['x'] <= x < item['x']+item['width'] and item['y'] <= y < item['y']+item['height']:
                    require(self.upstream is None and item['layer_id'] == layer-1,
                            'One independently retained preceding-layer final hidden')
                    self.upstream = ordinal, x-item['x'], y-item['y'], item
            if item['layer_id'] != layer:
                continue
            for serial in (1, 2, 3):
                r = capture['raw_files'][f's{serial}-diagnostic-{ordinal:04}.bin']
                require(r['rectangle'] == box(item) and r['dtype'] == item['dtype'] and
                        r['symbol'] == item['symbol'], 'Global ordinal and physical diagnostic provenance')
            for y in range(item['height']):
                for x in range(item['width']):
                    key = item['symbol'], item['x']+x-region['x'], item['y']+y
                    require(key not in self.index, 'One native diagnostic owner per original-layer coordinate')
                    self.index[key] = ordinal, x, y
        require((layer == 0) == (self.upstream is None), 'Only Layer0 obtains a host embedding')
        self.cached_name = self.cached_value = None

    def coordinate(self, serial, symbol, x, y):
        ordinal, ix, iy = self.index[symbol, x, y]
        name = f's{serial}-diagnostic-{ordinal:04}.bin'
        if name != self.cached_name:
            self.cached_value, self.cached_name = self.raw(name), name
        return self.cached_value[iy, ix].copy()

    def rectangle(self, serial, symbol, x, y, width, height, count):
        import numpy as np
        owners, result = {}, None
        for yy in range(height):
            for xx in range(width):
                ordinal, ix, iy = self.index[symbol, x+xx, y+yy]
                owners.setdefault(ordinal, []).append((xx, yy, ix, iy))
        for ordinal, points in owners.items():
            value = self.raw(f's{serial}-diagnostic-{ordinal:04}.bin')
            require(value.shape[2] == count, 'Exact diagnostic element count')
            if result is None:
                result = np.empty((height, width, count), dtype=value.dtype)
            for xx, yy, ix, iy in points:
                result[yy, xx] = value[iy, ix]
        require(result is not None, 'Nonempty actual arithmetic view')
        return result

    def input(self, serial):
        if self.layer == 0:
            return self.raw(f's{serial}-original-input.bin').reshape(5120).copy()
        ordinal, ix, iy, item = self.upstream
        value = self.checked(f's{serial}-diagnostic-{ordinal:04}.bin', item)[iy, ix]
        require(value.size == 5120 and value.dtype.str == '<u2', 'Full actual preceding-layer BF16 vector')
        return value.copy()

    def input_rows(self):
        import numpy as np
        return np.stack([self.input(1), self.input(2)])


class AuditPrepared:
    """Own original weights plus an actual device input for downstream audits."""
    def __init__(self, prepared, evidence):
        self.layer, self.prepared, self.evidence = evidence.layer, prepared, evidence

    def array(self, name):
        return self.prepared.layers[self.layer].array(name)

    def input_row(self, index):
        require(index in (0, 1), 'Only the two captured causal positions')
        return (self.prepared.input_row(index) if self.layer == 0 else
                self.evidence.input(index+1).reshape(1, 1, 5120))


def numeric_inputs(families, lowered, host, layer):
    """Only tensor namespace and physical sampling offset change the oracle."""
    region = next(r for r in lowered['regions'] if r['layer'] == layer)
    plan = copy.deepcopy(families[region['family']]['plan'])
    if region['family'] == 'linear':
        def rename(value):
            if isinstance(value, str):
                return value.replace('model.language_model.layers.0.', f'model.language_model.layers.{layer}.')
            if isinstance(value, list):
                return [rename(v) for v in value]
            if isinstance(value, dict):
                return {k: rename(v) for k, v in value.items()}
            return value
        plan = rename(plan)
    plan['layer'], plan['physical_x_offset'] = layer, region['x']
    copies = dict(uploads=[p for p in host['copies']['uploads'] if p['layer_id'] == layer])
    return plan, copies
