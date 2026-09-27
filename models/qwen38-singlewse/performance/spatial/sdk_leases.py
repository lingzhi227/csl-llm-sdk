"""Reject explicit application leases that overlap the default SDK memcpy ABI.

This narrow source gate complements compilation; it does not prove dynamic
buffer lifetimes, compiler-internal descriptors or queue liveness. The installed
SDK 2.10.1 memcpy/wse3/memcpy.csl defaults dest_dsr_ids and src1_dsr_ids to [0].
Reserve the entire descriptor index 0 conservatively, plus queues/microthreads
0 and 1. This gate applies to the explicit-literal full-matrix engine only.
"""
import hashlib
import re


def audit_explicit_sdk_leases(sources):
    leases = []
    patterns = {
        'descriptor': r'@get_dsr\(\s*(dsr_dest|dsr_src0|dsr_src1)\s*,\s*(\d+)\s*\)',
        'input_queue': r'@get_input_queue\(\s*(\d+)\s*\)',
        'output_queue': r'@get_output_queue\(\s*(\d+)\s*\)',
        'microthread': r'@get_ut_id\(\s*(\d+)\s*\)',
    }
    for name, raw in sources.items():
        source = re.sub(r'//[^\n]*', '', raw.decode())
        for kind, pattern in patterns.items():
            matches = list(re.finditer(pattern, source))
            builtin = 'get_dsr' if kind == 'descriptor' else 'get_ut_id' if kind == 'microthread' else 'get_' + kind
            if len(matches) != len(re.findall('@' + builtin + r'\s*\(', source)):
                raise ValueError(f'{name}: nonliteral {kind} lease requires an explicit audit')
            for match in matches:
                index = int(match.groups()[-1])
                if index in ({0} if kind == 'descriptor' else {0, 1}):
                    raise ValueError(f'{name}: application {kind} {index} overlaps SDK reservation')
                if not 0 <= index < 8:
                    raise ValueError(f'{name}: invalid WSE-3 {kind} {index}')
                leases.append(dict(file=name, resource=kind, index=index,
                                   descriptor_kind=match.group(1) if kind == 'descriptor' else None))
    if not leases:
        raise ValueError('No explicit application leases checked')
    return dict(schema='wse-explicit-sdk-lease-audit-v1', passed=True,
                source_sha256={name: hashlib.sha256(raw).hexdigest() for name, raw in sources.items()},
                leases=leases, sdk_reserved_descriptor_indices=[0],
                sdk_reserved_queue_and_microthread_indices=[0, 1],
                scope='Explicit literal application leases avoid default SDK memcpy reservations; dynamic lifetimes and queue liveness require execution.')
