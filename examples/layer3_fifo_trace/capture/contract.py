"""Finite, role-bounded copy plan for one original layer3 diagnostic epoch."""
import hashlib
import json

WIDTH, HEIGHT, EPOCHS = 750, 45, 1

def spec(name, x, y, width, height, count, dtype):
    return dict(name=name, x=x, y=y, width=width, height=height,
                count=count, dtype=dtype, bits=16 if dtype == 'u16' else 32)


def parameters(plan):
    for i, batch in enumerate(plan['weight_batches']):
        yield f'weights-{i:03}.npy', spec('weights', batch['x'], batch['y'],
            batch['width'], batch['height'], 6144, 'u32'), None
    for batch in plan['weight_batches']:
        box = [batch[k] for k in ('x', 'y', 'width', 'height')]
        yield 'descriptor.npy', spec('descriptor', *box, 6, 'u16'), box
    yield 'route.npy', spec('route', 0, 0, WIDTH, HEIGHT, 4, 'u16'), None
    yield 'input-norm.npy', spec('norm_gains', 1, 0, 1, 1, 5120, 'u16'), None
    yield 'post-norm.npy', spec('norm_gains', 2, 0, 1, 1, 5120, 'u16'), None
    yield 'qk-norms.npy', spec('qk_gains', 16, 0, 24, 1, 512, 'u16'), None
    yield 'frequencies.npy', spec('frequencies', 16, 0, 24, 1, 32, 'f32'), None


def operations(plan):
    for _, item, _ in parameters(plan):
        yield dict(kind='copy', direction='h2d', **item)
    yield dict(kind='launch', name='initialize')
    for _, item, _ in parameters(plan):
        yield dict(kind='copy', direction='d2h', **item)
    state = spec('state', 0, 0, WIDTH, HEIGHT, 12, 'u32')
    yield dict(kind='copy', direction='d2h', **state)
    yield dict(kind='copy', direction='h2d', **spec('hidden', 1, 0, 1, 1, 5120, 'u16'))
    yield dict(kind='launch', name='prepare')
    yield dict(kind='copy', direction='d2h', **state)
    yield dict(kind='launch', name='compute')
    yield dict(kind='copy', direction='d2h', **spec('observer', 748, 0, 2, 25, 160, 'u32'))


def summary(plan):
    ops = list(operations(plan)); copies = [x for x in ops if x['kind'] == 'copy']
    return dict(copies=len(copies), launches=len(ops)-len(copies),
        maximum_host_transfer_bytes=max(x['width']*x['height']*x['count']*4 for x in copies),
        host_transfer_bytes=sum(x['width']*x['height']*x['count']*4 for x in copies),
        operation_sha256=hashlib.sha256(json.dumps(ops, sort_keys=True,
            separators=(',', ':')).encode()).hexdigest(),
        all_descriptors_matrix_only=True, all_diagnostics_role_bounded=True,
        observer_delay_seconds=20, observer_deadline_seconds=35, observer_host_bytes=32000, post_compute_all_PE_read=False, unconditional_stop_after_snapshot=True)
