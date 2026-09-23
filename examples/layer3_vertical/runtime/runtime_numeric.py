"""Accepted Layer3 arithmetic with actual vertical-to-logical raw views only.

Source operator formulas/order/gates are preserved. Geometry, original packed
file lookup and predetermined physical-coordinate sample selection differ.
All transport/state/journal checks live in the separate evidence audit.
"""
import numpy as np
import time
from runtime_boundary import require
from numeric_views import Views
from qualified_math.projection_checks import decode,encode,equal,normal,norm_gate,nonlinear_gate
from qualified_math.local_bounds import local_bounds
from qualified_math.exact_arithmetic import exact_dot
from qualified_math.qk_native_conditional import validate_qk
from qualified_math.attention_conditional import validate_attention
SCHEDULE=((1,1,0,0),(2,1,1,1),(3,2,0,0))

def audit(evidence,prepared,physical,copies,math,nominal_output):
    import json
    from pathlib import Path
    started=time.monotonic()
    plan=json.loads((Path(__file__).resolve().parent/'logical-plan.json').read_bytes())
    views=Views(evidence,prepared,physical,copies,plan)
    require(nominal_output.shape==(8,5120) and nominal_output.dtype==np.dtype('<u2'),'Eight pinned original nominal boundary outputs')
    original_input = prepared.array('reference-input.npy')[0]
    original_output = nominal_output
    gains = [prepared.array(name)[0].reshape(-1) for name in ('input-norm.npy', 'post-norm.npy')]
    qkg = prepared.array('qk-norms.npy')[0]
    freq = prepared.array('frequencies.npy')[0]
    prior_cache = np.zeros((24, 4096), np.uint16)
    results = []; rows = samples = 0
    first_cache=first_output=None
    for epoch,reset,position,input_row in SCHEDULE:
        archives=views.archives(epoch,reset,position)
        peers=views.peers(epoch)
        for name, value in peers.items():
            if value.dtype == np.float32:
                normal(value, 'Finite normal peer diagnostics: '+name)
        # Canonical normalized BF16 input vectors remain in actual active lanes.
        vectors = {}
        for i, b in enumerate(plan['weight_batches']):
            phase = next(r['frame_phase'] for r in plan['matrix_regions'] if r['name'] == b['name'])
            if phase in (0, 2) and phase not in vectors:
                data = views.matrix(epoch,i)['active_input'][0]
                require(not np.any(data.view(np.uint32)&65535), 'Norm handoff is exactly BF16 expanded')
                vectors[phase] = (data.reshape(-1)[:5120].view(np.uint32) >> 16).astype(np.uint16)
        vectors[1] = peers['attention_casts'].reshape(24, 768)[:, 512:].reshape(-1)
        vectors[3] = peers['mlp_product'].reshape(-1)
        projections = {r['name']: np.zeros(r['shape'][0], np.uint16) for r in plan['matrix_regions']}
        covered = {name: set() for name in projections}
        for i, b in enumerate(plan['weight_batches']):
            data = views.matrix(epoch,i)
            h, w = b['height'], b['width']; n = b['shape'][1]
            phase = next(r['frame_phase'] for r in plan['matrix_regions'] if r['name'] == b['name'])
            operand = np.zeros(w*96, np.float32); operand[:n] = decode(vectors[phase])
            current = np.broadcast_to(operand.reshape(1, w, 96), (h, w, 96)).copy()
            equal(data['active_input'], current, 'Every active lane receives same canonical neural vector and padding')
            final = np.zeros(w*96, np.float32); count = min(final.size, 17408)
            final[:count] = decode(vectors[3][:count])
            equal(data['input'], np.broadcast_to(final.reshape(1, w, 96), (h, w, 96)).copy(), 'Final broadcast period retained on all lanes')
            counters = np.broadcast_to(np.array([epoch, epoch, epoch, 4*epoch], np.uint32), (h, w, 4)).copy()
            equal(data['lane_counters'], counters, 'Every lane consumed four frames and exactly one GEMV/line per position')
            weights=views.weight_batch(i).view(np.uint16).reshape(h,w,96,128)
            low, high = local_bounds(weights, data['active_input'])
            part = data['partial']; normal(part, 'Finite normal partials')
            require(np.all(part.astype(np.float64) >= low) and np.all(part.astype(np.float64) <= high), 'All actual-operand96FMA local-row enclosures')
            rows += part.size
            for yy in range(h):
                valid = min(128, b['shape'][0]-128*b['row_groups'][yy])
                for xx in range(w):
                    actual_x,actual_y=views.physical_coordinate(i,yy,xx)
                    chosen = (actual_y*37+actual_x*17+epoch*13) % valid
                    expected = exact_dot(weights[yy, xx, :, chosen],
                        (data['active_input'][yy, xx].view(np.uint32) >> 16).astype(np.uint16))
                    require(expected == int(part[yy, xx, chosen:chosen+1].view(np.uint32)[0]), 'Predetermined exact integer FP32 forward96FMA sample')
                    samples += 1
            reduced = part[:, -1].copy()
            for xx in range(w-2, -1, -1):
                reduced = np.add(reduced, part[:, xx], dtype=np.float32)
                normal(reduced, 'Normal or zero right-to-left reduction intermediate')
            equal(data['result'], np.broadcast_to(reduced[:, None], (h, w, 128)).copy(), 'Exact right-to-left chain and every result receiver')
            equal(data['rounded'][:, 0], encode(reduced), 'All root BF16 RNE conversions')
            for yy, group in enumerate(b['row_groups']):
                require(group not in covered[b['name']], 'Projection row coverage disjoint')
                covered[b['name']].add(group); start = 128*group; count = min(128, b['shape'][0]-start)
                require(not np.any(data['rounded'][yy, 0, count:]), 'Padded rows zero')
                projections[b['name']][start:start+count] = data['rounded'][yy, 0, :count]
        require(all(len(covered[r['name']]) == r['row_groups'] for r in plan['matrix_regions']), 'All seven projection row groups')
        def projection(suffix):
            return projections['model.language_model.layers.3.'+suffix+'.weight']
        q = projection('self_attn.q_proj').reshape(24, 512)
        k = np.repeat(projection('self_attn.k_proj').reshape(4, 256), 6, axis=0)
        v = np.repeat(projection('self_attn.v_proj').reshape(4, 256), 6, axis=0)
        qi = peers['qk_input'].reshape(24, 512); ai = peers['attention_input'].reshape(24, 1024)
        equal(qi[:, :256].copy(), q[:, :256].copy(), 'All24Q projection root handoffs')
        equal(qi[:, 256:].copy(), k, 'All4K roots fanout to all24queries')
        equal(ai[:, 512:768].copy(), v, 'All4V roots fanout to all24queries')
        equal(ai[:, 768:].copy(), q[:, 256:].copy(), 'All24rawgate projection handoffs')
        equal(peers['qk_output'].reshape(24, 512), ai[:, :512].copy(), 'Retained QK output alias identity')
        head_reports = []
        for head in range(24):
            actual = dict(stats=peers['qk_stats'].reshape(24,10)[head].tolist(),
                          output=peers['qk_output'].reshape(24,512)[head].tolist())
            for name,record in archives[head]['arrays'].items():
                raw=np.asarray(record['bits'],dtype=np.uint32 if record['dtype']=='f32' else np.uint16)
                actual[name]=(raw.view(np.float32) if record['dtype']=='f32' else raw).tolist()
            token = dict(position=position, q=decode(qi[head, :256]).tolist(), k=decode(qi[head, 256:]).tolist())
            checked = validate_qk(token, decode(qkg[head, :256]).tolist(), decode(qkg[head, 256:]).tolist(), freq[head].tolist(), actual,
                input_bits=qi[head].tolist(),gain_bits=qkg[head].tolist())
            require(checked['passed'], 'Complete actual-operand QK/RoPE gates head '+str(head))
            observed = {name: peers['attention_'+export].reshape(24, -1)[head].tolist() for name, export in [
                ('scores', 'scores'), ('score_casts', 'score_casts'), ('stats', 'stats'),
                ('stages', 'stages'), ('output_casts', 'casts')]}
            atoken = dict(token=position+1, q=decode(ai[head, :256]).tolist(), k=decode(ai[head, 256:512]).tolist(),
                v=decode(ai[head, 512:768]).tolist(), gate=decode(ai[head, 768:]).tolist())
            prior_cache[head, position*256:(position+1)*256] = ai[head, 256:512]
            prior_cache[head, 2048+position*256:2048+(position+1)*256] = ai[head, 512:768]
            att = validate_attention(atoken, decode(prior_cache[head]).tolist(), observed)
            require(att['passed'], 'Complete actual-operand attention gates head '+str(head))
            head_reports.append(dict(qk=checked, attention=att))
        equal(peers['cache'].reshape(24, 4096), prior_cache, 'All cache appends, old prefixes and untouched tails')
        for kv in range(4):
            equal(prior_cache[6*kv:6*kv+6], np.broadcast_to(prior_cache[6*kv], (6, 4096)).copy(), 'All six canonical KV replicas')
        hidden = peers['hidden'].reshape(2, 5120); saved = peers['norm_saved'].reshape(2, 5120)
        equal(saved[0], original_input[input_row], 'Exact original layer input boundary')
        residual = encode(np.add(decode(saved[0]), decode(projection('self_attn.o_proj')), dtype=np.float32))
        equal(hidden[0], residual, 'First full5120residual conversion')
        equal(saved[1], residual, 'First residual to post norm handoff')
        final = encode(np.add(decode(saved[1]), decode(projection('mlp.down_proj')), dtype=np.float32))
        equal(hidden[1], final, 'Second full5120residual conversion')
        norms = [norm_gate(saved[i], gains[i], peers['norm_stats'].reshape(2, 4)[i], vectors[2*i]) for i in range(2)]
        equal(projection('mlp.gate_proj'), peers['mlp_gate'].reshape(-1), 'Every gate root to MLP owner')
        equal(projection('mlp.up_proj'), peers['mlp_up'].reshape(-1), 'Every up root to MLP owner')
        nonlinear = nonlinear_gate(peers)
        fc = peers['frame_counters'].reshape(-1)
        require(0 <= int(fc[0]) <= 3*epoch and np.array_equal(fc[1:], [4*epoch, 264*epoch, 69888*epoch, epoch-1]), 'Device origin frames/chunks/retirement counters')
        actual = decode(final).astype(np.float64); nominal = decode(original_output[input_row]).astype(np.float64)
        difference = np.abs(actual-nominal); nonzero = nominal != 0
        if epoch==1:
            first_cache=prior_cache.copy();first_output=final.copy()
        elif epoch==3:
            equal(final,first_output,'Device reset replay exactly reproduces first output')
            equal(prior_cache[:,:256],first_cache[:,:256],'Replay active K slot exactly reproduces first execution')
            equal(prior_cache[:,2048:2304],first_cache[:,2048:2304],'Replay active V slot exactly reproduces first execution')
        # The next serial starts from this actual independently checked capture.
        # Reset retains inactive suffix bytes; it does not replace cache with zeros.
        prior_cache=peers['cache'].reshape(24,4096).copy()
        results.append(dict(serial=epoch,reset_generation=reset,position=position,input_row=input_row, norms=norms, nonlinear=nonlinear, heads=head_reports,
            nominal_original_bf16_mismatches=int(np.count_nonzero(final != original_output[input_row])),
            nominal_max_absolute_error=float(np.max(difference)), nominal_max_relative_error_nonzero=float(
                np.max(difference[nonzero]/np.abs(nominal[nonzero]))), nominal_zero_reference_absolute_error=float(
                np.max(difference[~nonzero], initial=0)), nominal_bitwise_match=bool(np.array_equal(final, original_output[input_row]))))
    require(rows == 3*30576*128 and samples == 3*30576, 'Complete finite matrix check counts')
    return dict(status='passed', layer=3, positions=2,serial_executions=3,device_reset_replay_exact=True,local_conditional_rows=rows, exact_FMA_samples=samples,
        sample_formula='((actual_y*37)+(actual_x*17)+(serial*13))%valid_rows',
        exact_FMA_multiply_slots=samples*96, results=results, seconds=time.monotonic()-started,
        original_whole_layer_nominal_compared=True, all_operator_gates=True,
        archived_standalone_trig_probes_numeric_gate=False,
        QK_RMS_observed_statistics=10,QK_normalization_uses_conditional_enclosure=True,
        overwritten_RMS_stages_used_as_observations=False, full_model=False,
        source_propagated_whole_layer_enclosure=False, SDK_imported=False)
