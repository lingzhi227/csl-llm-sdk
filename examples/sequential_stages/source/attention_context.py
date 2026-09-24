"""Bind the exact accepted observer checks to one scheduled stage context.

Record fields, error checks, protocol counters and numeric domains are reused.
The old three-execution allowlist, head/archive capacity and origin sequence
capacity are replaced by the explicit context and matching derived device limits.
"""
from pathlib import Path
import hashlib,types
from stage_contract import Context,CONTEXT_CAPACITY,require
from observer_capacity import wire_limit

PINS={
 'observer_decode.py':'4b3a683674916066d15f80ca9f491905fd9a9444ab6682eecbdac832147ac090',
 'archive_decode.py':'4dc0bbd7c001f3da71e3f52832644f07ad1bb04be08dd23b1f2ce4d18c9b9d9b',
 'runtime_boundary.py':'61eefe32fee886d5b226473b8aa66b88d4b8b4b5f916a67af26d3846df44bbed'}


def source(directory,name):
    raw=(Path(directory)/name).read_bytes()
    require(hashlib.sha256(raw).hexdigest()==PINS[name],'Exact accepted attention host source: '+name)
    return raw.decode()


def replace(text,old,new):
    require(text.count(old)==1,'One explicit context adaptation: '+old)
    return text.replace(old,new,1)


def bind(directory,context):
    require(isinstance(context,Context) and 0<len(context.positions)<=CONTEXT_CAPACITY and
            all(type(p) is int and 0<=p<CONTEXT_CAPACITY for p in context.positions) and
            len(set(context.positions))==len(context.positions), 'Scheduled finite stage context')
    def scheduled(serial,position=None):
        require(type(serial) is int and 1<=serial<=len(context.positions),'Local serial belongs to this exact runtime context')
        if position is not None:
            require(type(position) is int and position==context.positions[serial-1],
                    'Logical position comes from the scheduled context, not local serial')
        return context.positions[serial-1]
    text=source(directory,'observer_decode.py')
    text=replace(text,'require((serial, position) in ((1,0),(2,1),(3,0),(1,1)), "Exact scheduled head identity")',
                 'scheduled(serial,position)')
    text=replace(text,'require(serial in (1,2,3), "Exact scheduled global serial")','scheduled(serial)')
    text=replace(text,'values[0] <= 390','values[0] <= 2*65*8')
    text=replace(text,'values[5] == 192','values[5] == 64*8')
    text=replace(text,'values[0]<=8192','values[0]<=ORIGIN_WIRE_LIMIT')
    decoder=types.ModuleType('scheduled_attention_observer');decoder.scheduled=scheduled
    decoder.ORIGIN_WIRE_LIMIT=wire_limit('attention')
    exec(compile(text,'<pinned-context-attention-observer>','exec'),decoder.__dict__)
    archive=types.ModuleType('scheduled_attention_archive')
    exec(compile(source(directory,'archive_decode.py'),'<pinned-attention-archive>','exec'),archive.__dict__)
    def decode_archive(status,raw_u32,*,head,identity,generation):
        position=scheduled(generation)
        require(identity==[context.request_id,context.generation,position,context.stage.stage_id,0],
                'Archive request, logical generation, position, stage and operation are context-bound')
        return archive.decode_archive(status,raw_u32,head=head,identity=identity,generation=generation,
                                      stream_limit=65*CONTEXT_CAPACITY)
    text=source(directory,'runtime_boundary.py')
    text=replace(text,'from .observer_decode import decode_global,semantic_state,decode_head','')
    text=replace(text,'from .archive_decode import decode_archive','')
    text=replace(text,'a[0]<=8192','a[0]<=ORIGIN_WIRE_LIMIT')
    # Each new namespace gets context-bound functions; the frozen package and
    # another runtime's decoder globals are never modified in place.
    module=types.ModuleType('scheduled_attention_boundary');module.__package__='qualified_attention'
    module.ORIGIN_WIRE_LIMIT=wire_limit('attention')
    module.decode_global=decoder.decode_global;module.semantic_state=decoder.semantic_state
    module.decode_head=decoder.decode_head;module.decode_archive=decode_archive
    exec(compile(text,'<pinned-context-attention-boundary>','exec'),module.__dict__)
    module.context=context
    return module
