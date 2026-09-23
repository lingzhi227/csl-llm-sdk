"""Independent finite operation ledger for both physical contexts."""
from runtime_boundary import require


def schedule(mode):
    require(mode in ('baseline', 'restore'), 'Known physical context')
    return [(1, 1, 0, 0), (2, 1, 1, 1), (3, 2, 0, 0)] if mode == 'baseline' else [(1, 1, 1, 1)]


def scheduled(host, observations, mode):
    copies = host['copies']
    steps = schedule(mode)
    observed = {(o['serial'], o['layer']): o for o in observations}
    require(len(observed) == len(observations) == 4*len(steps) and
            set(observed) == {(s[0], layer) for s in steps for layer in range(4)},
            'Exactly one final observation for every layer and serial')

    def transfers(direction, items, prefix, digits=4):
        for index, item in enumerate(items):
            yield dict(kind='copy', direction=direction, item=item, label=f'{prefix}-{index:0{digits}}')

    def command(name):
        return dict(kind='launch', name=name)

    weights_done = False
    for index, item in enumerate(copies['uploads']):
        if item['symbol'] == 'weights':
            if not weights_done:
                yield dict(kind='weight_upload', items=[p for p in copies['uploads'] if p['symbol'] == 'weights'])
                weights_done = True
        else:
            yield dict(kind='copy', direction='h2d', item=item, label=f'upload-once-{index:04}')
    if mode == 'restore':
        yield from transfers('h2d', copies['checkpoint_copies'], 'restore-bank', 3)
    yield command('initialize')
    yield from transfers('d2h', copies['uploads'], 'initial-retention')
    yield from transfers('d2h', copies['state_copies'], 's0-initialized-state')
    if mode == 'baseline':
        yield from transfers('d2h', copies['persistent_copies'], 'initialized-persistent')
    for serial, generation, position, _ in steps:
        if generation == 2:
            yield command('reset')
            yield from transfers('d2h', copies['state_copies'], 's2-reset-state')
            yield from transfers('d2h', copies['persistent_copies'], 'reset-persistent')
        yield dict(kind='copy', direction='h2d', item=copies['hidden_input'], label=f's{serial}-original-input')
        yield command('prepare')
        yield from transfers('d2h', copies['state_copies'], f's{serial}-prepared-state')
        yield command('compute')
        for item in copies['observers']:
            attempts = observed[serial, item['layer_id']]['attempts']
            require(type(attempts) is int and 1 <= attempts <= 128, 'Finite actual observer prefix')
            for attempt in range(attempts):
                yield dict(kind='copy', direction='d2h', item=item,
                           label=f"s{serial}-layer{item['layer_id']}-observer-{attempt:03}")
        yield from transfers('d2h', copies['state_copies'], f's{serial}-complete-state')
        for snapshot in range(2):
            yield from transfers('d2h', copies['transport_copies'], f's{serial}-transport-{snapshot}')
        yield from transfers('d2h', copies['control_copies'], f's{serial}-control', 3)
        yield from transfers('d2h', copies['handoff_copies'], f's{serial}-handoff', 3)
        yield from transfers('d2h', copies['diagnostic_copies'], f's{serial}-diagnostic')
        if generation == 1:
            yield from transfers('d2h', copies['checkpoint_copies'], f's{serial}-semantic', 3)
    yield from transfers('d2h', copies['uploads'], 'terminal-retention')
