"""Decode first-fault telemetry; failure evidence never qualifies protocol success."""
import hashlib,struct
WIDTH=178
FIELDS=('begin','magic','site','role_column','first_error','protocol_error','rx_phase_after',
    'rx_phase_before','tx_phase','network_header','receive_length','control_word','leases',
    'pending_length','pending_xy','source_frame_validity','headers','bodies','control_tails',
    'errors','TX_frames','TX_tails','ordinary_tail_events','native_control_entries',
    'unexpected_data_entries','ready_count','rows_done','peer_rows','request',
    'requests_received','application_flags','end')
def inspect_faults(first,final):
    if len(first)!=WIDTH or len(final)!=WIDTH or any(len(v)!=128 for v in first+final):
        raise ValueError('Complete178x128 fault capture required')
    changed=[];incomplete=[];faults=[]
    for x,(a,b) in enumerate(zip(first,final)):
        a=[int(v) for v in a];b=[int(v) for v in b]
        if any(v<0 or v>0xffffffff for v in a+b):raise ValueError('Unsigned full-word telemetry')
        if a!=b:changed.append(x)
        if not any(b):continue
        role=1 if x==0 else 2 if x==2 else 3 if 40<=x<176 else 0
        if b[0]!=2 or b[31]!=2 or b[1]!=0x46313336 or b[2] not in (1,2,3,4) or b[3]!=(role|(x<<16)) or role==0 or b[63]!=0 or b[127]!=0:
            incomplete.append(x)
        source_capacity=31 if role==2 else 8
        source_valid=bool(b[12]&1) and 1<=b[13]<=source_capacity
        frame_valid=bool(b[12]&1) and not bool(b[12]&2) and b[8] in (1,2)
        invalid_length=bool(b[12]&1) and not 1<=b[13]<=source_capacity
        expected_flags=int(source_valid)|(int(frame_valid)<<1)|(int(bool(b[12]&2))<<2)|(int(invalid_length)<<3)
        source_length=b[13] if source_valid else 0
        site_valid=b[2]==1 or b[2] in (2,4) and role==1 or b[2]==3 and role==2
        lease_valid=not bool(b[12]&2) or bool(b[12]&1)
        if b[15]!=expected_flags or any(b[96+source_length:127]) or invalid_length or not site_valid or not lease_valid or b[12]&~15 or b[8]>2 or b[6]>5 or b[7]>5 or b[4]==0:
            if x not in incomplete:incomplete.append(x)
        row=dict(column=x,source_lease_valid=bool(b[15]&1),frame_issued_and_lease_held=bool(b[15]&2),source_queued=bool(b[15]&4),metadata=dict(zip(FIELDS,b[:32])),bank_sha256=hashlib.sha256(struct.pack('<128I',*b)).hexdigest())
        # At most four full selected first-fault payloads in the small report.
        # Both complete rectangles remain in the raw captures onALCF/Mass1.
        if len(faults)<4:row.update(RX_words=b[32:63],TX_frame=b[64:96],current_source=b[96:127])
        faults.append(row)
    return dict(scope='First-fault scalar point snapshot; no repair or continuous DMA claim',
        capture_stable=not changed,changed_columns=changed,incomplete_columns=incomplete,
        latched_faults=len(faults),faults=faults)
