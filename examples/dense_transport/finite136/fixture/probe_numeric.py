"""Future dense fixture audit against exact integer arithmetic witnesses.

All captured FP32 values must be provided as their original u32 bits, never
converted through rounded decimal JSON. Arrays stay on the admitted remote
host. This module is source-only and does not import or invoke the SDK.
"""
from probe_fixture import (descriptor, expected_row, f32_bits, frame_words,
    input_numerator, weight_words)
from probe_boundary import exact, require


def retained(actual,initial,key):
    exact(actual['weights'],initial['weights'],key+' actual resident weight retention')
    exact(actual['descriptor'],initial['descriptor'],key+' actual descriptor retention')


def validate(snapshot, position, parameter_snapshot):
    require(position in (0,1), 'Fixture position')
    source_FMAs=0; output_rows=0
    for phase in range(2):
        for y in range(4):
            for ordinal in range(4):
                key=f'{1+5*phase+ordinal},{y}'; actual=snapshot[key]
                params=parameter_snapshot[key]
                retained(actual,params,key)
                exact(params['weights'],weight_words(phase,y,ordinal),key+' resident packed weights')
                exact(params['descriptor'],descriptor(phase,y),key+' descriptor')
                active=[f32_bits(input_numerator(position,phase,ordinal,c)/32) for c in range(96)]
                final=[f32_bits(input_numerator(position,1,ordinal,c)/32) for c in range(96)]
                exact(actual['active_input'],active,key+' active input bits')
                exact(actual['input'],final,key+' final frame bits')
            expected=[expected_row(position,phase,y,row) for row in range(128)]
            for ordinal in range(4):
                key=f'{1+5*phase+ordinal},{y}'; actual=snapshot[key]
                exact(actual['partial'],[r['partial_bits'][ordinal] for r in expected],key+' local96 FMA bits')
                exact(actual['result'],[r['result_bits'] for r in expected],key+' collective bits')
                source_FMAs+=128*96
            root=f'{1+5*phase},{y}'
            rounded=[r['rounded_bf16'] for r in expected]
            exact(snapshot[root]['rounded'],rounded,root+' BF16 RNE')
            receiver=snapshot[f'11,{(y+phase+1)%4}']['received_rows']
            require(len(receiver)==256,'Receiver full two-slot capture')
            exact(receiver[phase*128:(phase+1)*128],rounded,root+' delivered full row')
            output_rows+=1
    exact(snapshot['11,0']['frame_input'],frame_words(position),'Origin two retained frames')
    require(source_FMAs==393216 and output_rows==8,'Complete arithmetic coverage')
    return dict(scope='Finite exact-dyadic transport fixture only; not original model or expanded arithmetic domain',
        position=position, matrix_PEs=32, source_FMAs=source_FMAs,
        collective_adds=3072, BF16_outputs=1024, exact_bit_mismatches=0)
