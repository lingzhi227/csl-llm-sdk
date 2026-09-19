"""Decode complete 33-word snapshots; preserve all original protocol checks."""
from check_native import check_native

def split_snapshot(words):
    if len(words)!=2 or any(len(row)!=33 for row in words):
        raise ValueError('Complete two-PE 33-word snapshot required')
    if any(type(v) is not int or not 0<=v<2**32 for row in words for v in row):
        raise ValueError('Every snapshot word must be complete u32')
    return [row[:30] for row in words],[row[30:31] for row in words],[row[31:33] for row in words]

def check_snapshot(case,initial,final):
    _,initial_tail,initial_native=split_snapshot(initial)
    state,tail,native=split_snapshot(final)
    if tail!=initial_tail or native!=initial_native:
        raise ValueError('Snapshot counter trailer changed in final observation window')
    if tail!=[[1 if case==5 else 0],[0]]:
        raise ValueError('Exact ordinary-tail completion count required')
    check_native(case,native)
    return state,tail,native
