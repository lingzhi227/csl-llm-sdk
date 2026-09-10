"""Account for a proposed synchronous output-shard schedule; never launch SDK."""
import json


def symbolic_sequence(pes,calls):
    """Full5120 physical ordering, with a stable ordinal for every copy and launch."""
    sequence=[]
    copies=launches=0
    def copy(direction,name,pe,count,bits=32):
        nonlocal copies
        copies+=1
        sequence.append(dict(kind='copy',sequence=copies,direction=direction,name=name,
                             pe=pe,count=count,bits=bits,host_bytes=count*4,native_bytes=count*bits//8))
    def launch(name):
        nonlocal launches
        launches+=1
        sequence.append(dict(kind='launch',ordinal=launches,name=name))
    for pe in range(pes):
        copy('h2d','output_storage',pe,130)
        copy('h2d','bf16_storage',pe,130,16)
    for call in range(calls):
        for pe in range(pes):copy('h2d','control',pe,5)
        launch('begin')
        for tile in range(46):
            for pe in range(pes):
                copy('h2d','weights_storage',pe,14338,16)
                copy('h2d','input_storage',pe,114)
                copy('h2d','control',pe,5)
            launch('accumulate')
            if tile in (0,44):
                for pe in range(pes):
                    copy('d2h','output_storage',pe,130)
                    copy('d2h','state',pe,13)
        launch('finalize')
        for pe in range(pes):
            for name,count,bits in [('output_storage',130,32),('bf16_storage',130,16),
                ('state',13,32),('weight_guard_snapshot',5,32),('input_storage',114,32),('timing',276,16)]:
                copy('d2h',name,pe,count,bits)
            if call==calls-1:copy('d2h','weights_storage',pe,14338,16)
        launch('release')
        for pe in range(pes):copy('d2h','state',pe,13)
    return sequence


def plan(pes, calls, columns=5120):
    tiles = (columns+111)//112
    # Every physical transfer has width=1, even with multiple compute PEs.
    initial = [(130, 32), (130, 16)]  # guarded FP32 and BF16 output
    tile = [(14338, 16), (114, 32), (5, 32)]
    boundary = [(130, 32), (13, 32)]
    final = [(130, 32), (130, 16), (13, 32), (5, 32), (114, 32), (276, 16)]
    frames = initial + calls*([(5, 32)]+tiles*tile+2*boundary+final+[(13,32)]) + [(14338, 16)]
    assert all(count*4 <= 65536 for count, _ in frames)
    return dict(pes=pes, calls=calls, columns=columns, outputs=128*pes,
                tiles=tiles, last_valid=columns-112*(tiles-1),
                copies=pes*len(frames), host_slots=pes*sum(n for n, _ in frames),
                host_bytes=pes*sum(n*4 for n, _ in frames),
                payload_bytes=pes*sum(n*b//8 for n, b in frames),
                launches=calls*(tiles+3), max_physical_host_buffer=57352,
                fanout='Separate width1 original-input H2D to each PE; no implicit broadcast or inter-PE transport.',
                aggregation='Read each retained shard independently in global Q256/gate256/K256/V256 order; host comparison only.',
                lifecycle='One runtime. Initialize guarded outputs; each call begin,46 tiles,finalize,read,release,read release-state. Full four-call runs reuse accumulators, zero follows nonzero; preliminary has one case only.',
                boundary='Read guarded FP32 output and state after tiles0 and44; final includes casts, state, guard snapshot, input, all timing; last call also full resident weight tile.')


if __name__ == '__main__':
    sequential = plan(1,4)
    print(json.dumps({'sdk_authorized':False, 'preliminary':plan(4,1),
                      'full_single_pe_run':sequential,
                      'full_eight_sequential_runs':{k:8*sequential[k] for k in ('copies','host_slots','host_bytes','payload_bytes','launches')},
                      'full_parallel_eight_pe_run':plan(8,4),
                      'static_declared_bytes_per_pe':31080,
                      'stack_allowance':4096, 'sram_ordinary_end_ceiling':49152,
                      'actual_elf_admission':'Not compiled. Sum of declared arrays is not SRAM placement qualification.'},indent=2))
