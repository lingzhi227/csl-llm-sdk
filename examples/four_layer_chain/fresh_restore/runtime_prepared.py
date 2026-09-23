"""Own-layer immutable parameters; only Layer0 provides normal hidden input."""
from pathlib import Path
import hashlib, json
from qualified_attention.runtime_prepared import Prepared as QualifiedViews, fingerprint
from runtime_boundary import require


class Layer(QualifiedViews):
    def __init__(self, layer, root, manifest_pin):
        import numpy as np
        self.np, self.layer, self.root = np, layer, Path(root)
        self.cache, self.stamps, self.cache_bytes = {}, {}, 0
        self.verified, self.disk_bytes_read = set(), 0
        self.manifest_pin = manifest_pin
        path = self.root/'prepared.json'
        require(not path.is_symlink() and not path.stat().st_mode & 0o222 and
                path.stat().st_size <= 2 << 20, 'Immutable bounded layer preparation receipt')
        raw = path.read_bytes()
        require(dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) == manifest_pin,
                'Exact independently accepted original-layer preparation')
        self.manifest = json.loads(raw)
        self.files = self.manifest['files']
        require(self.manifest['all_original_bits_checked'], 'Complete original bit verification')
        require(self.manifest['application'] == [29 if layer == 3 else 30, 1160] and
                len(self.files) == [199, 198, 198, 55][layer], 'Exact family array inventory')
        if layer in (1, 2):
            require(self.manifest['kind'] == 'direct_original_rows_to_vertical_ROIs' and
                    self.manifest['layer'] == layer and self.manifest['original_tensors'] == 14 and
                    self.manifest['original_weight_bytes'] == 766546368 and
                    self.manifest['original_geometry_checked'], 'Own original Layer1/2 tensor ranges')
            require(not {'reference-input.npy', 'reference-output.npy'} & self.files.keys(),
                    'Intermediate CPU hidden inputs are not prepared')
        else:
            require(self.manifest['kind'] == 'derived_vertical_ROIs_from_accepted_original_words' and
                    self.manifest['all_original_words_reused_checked'], 'Accepted original-word derived preparation')
            if layer == 0:
                require(self.manifest['original_weight_bytes'] == 766546368, 'Complete original Layer0 tensors')
            else:
                require(self.manifest['packed_matrix_bytes'] == 751435776, 'Complete original Layer3 matrices')
        self.manifest_stamp = fingerprint(path)

    def verify_unchanged(self):
        require(fingerprint(self.root/'prepared.json') == self.manifest_stamp, 'Manifest unchanged during runtime')
        for name, before in self.stamps.items():
            require(fingerprint(self.root/name) == before, 'Prepared original bytes unchanged during runtime')
        expected = self.files.keys() - ({'reference-input.npy', 'reference-output.npy'} if self.layer == 3 else set())
        require(self.verified == expected, 'Every execution parameter consumed; only explicit audit references excluded')
        return dict(layer=self.layer, files=len(self.verified), disk_bytes_read=self.disk_bytes_read,
                    cache_bytes=self.cache_bytes, unchanged=True, original_shard_reads_during_runtime=0)


class Prepared:
    def __init__(self, inputs):
        require(set(inputs) == {'0', '1', '2', '3'}, 'Four independently accepted original preparations')
        self.layers = {int(i): Layer(int(i), value['root'], value['pin']) for i, value in inputs.items()}
        require(len({str(layer.root.resolve()) for layer in self.layers.values()}) == 4,
                'Distinct original-layer payload directories')

    def transfer(self, item):
        layer = item['layer_id']
        require(layer in self.layers and item['prepared']['source_layer'] == layer and
                item['prepared']['source_namespace'] == f'original-layer-{layer}', 'Exact own-layer upload source')
        require(item['prepared']['x'] == item['x'] and item['prepared']['y'] == item['y'],
                'Actual physical ROI agrees with nested asynchronous copy')
        require(not item['prepared']['file'].startswith('reference-'), 'No intermediate reference in parameter transfer')
        value, receipt = self.layers[layer].transfer(item)
        receipt.update(original_layer=layer, prepared_manifest=self.layers[layer].manifest_pin)
        return value, receipt

    def input_row(self, index):
        import numpy as np
        value, _ = self.layers[0].array('reference-input.npy')
        require(value.shape == (2, 5120) and value.dtype == np.dtype('<u2') and index in (0, 1),
                'Only two accepted original Layer0 embedding rows')
        return np.ascontiguousarray(value[index]).reshape(1, 1, 5120)

    def verify_unchanged(self):
        return {str(i): layer.verify_unchanged() for i, layer in self.layers.items()}
