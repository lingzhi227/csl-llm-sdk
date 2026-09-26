"""Fixed control-only host phases for the implemented resident model."""
from dataclasses import dataclass
from enum import IntEnum
from resident_geometry import LAYERS,CONTEXT

class Phase(IntEnum):
    RESET=0
    TOKEN=1
    EMBEDDING_SLICE=2
    QKV_PROJECT=3
    QKV_GATHER=4
    ROPE=5
    KV=6
    O_PROJECT=7
    O_GATHER=8
    ATTENTION_RESIDUAL=9
    ROUTER=10
    EXPERT_CONFIG=11
    EXPERT_EXECUTE=12
    EXPERT_JOIN=13
    HANDOFF=14
    HEAD_PROJECT=15
    HEAD_MAX=16

@dataclass(frozen=True)
class Command:
    phase:Phase
    layer:int=0
    index:int=0

def token_schedule(position):
    if not 0<=position<CONTEXT:raise ValueError('96-token resident runtime capacity exceeded; no silent cache wrap')
    yield Command(Phase.TOKEN)
    for k in range(30):yield Command(Phase.EMBEDDING_SLICE,index=k)
    for layer in range(LAYERS):
        yield Command(Phase.QKV_PROJECT,layer)
        for row in range(40):yield Command(Phase.QKV_GATHER,layer,row)
        yield Command(Phase.ROPE,layer)
        for head in range(8):yield Command(Phase.KV,layer,head)
        yield Command(Phase.O_PROJECT,layer)
        for row in range(23):yield Command(Phase.O_GATHER,layer,row)
        for phase in (Phase.ATTENTION_RESIDUAL,Phase.ROUTER,Phase.EXPERT_CONFIG,Phase.EXPERT_EXECUTE,Phase.EXPERT_JOIN,Phase.HANDOFF):
            yield Command(phase,layer)
    yield Command(Phase.HEAD_PROJECT)
    for row in range(38):yield Command(Phase.HEAD_MAX,index=row)

def contract():
    commands=list(token_schedule(0))
    return dict(implemented=True,commands_per_token=len(commands),all_layers=LAYERS,context=CONTEXT,
                host_neural_input='one token ID at west',host_control_fields=['phase','layer','index'],
                host_neural_output='one greedy token ID at east',
                intermediates='all hidden states, routing, expert outputs and KV remain on wafer',
                weight_uploads='initialization only; no per-layer or per-token reload')
