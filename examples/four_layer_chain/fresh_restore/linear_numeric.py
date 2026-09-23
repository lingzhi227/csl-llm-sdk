"""Full Layer0 actual-operand oracle composed from qualified primitive checks.

The math argument supplies the exact frozen Layer3 projection/norm/MLP
validators and Layer0 SIM002 preprocess/gated/full128 recurrent validators.
Their source closure must be pinned by the separately admitted offline owner.
No SDK or device write occurs. Whole-layer propagated bounds are not claimed.
"""
import hashlib,time

def audit(evidence,prepared,plan,copies,math,nominal_output):
    import numpy as np
    started=time.monotonic()
    def require(ok,message):
        if not ok:raise ValueError(message)
    def equal(a,b,message):
        require(a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes(),message)
    def decode(a):return (np.asarray(a,dtype='<u4')<<16).view('<f4')
    def encode(a):
        a=np.asarray(a,dtype='<f4');require(np.all(np.isfinite(a)),'Finite BF16 cast source')
        b=a.view('<u4');return ((b+32767+((b>>16)&1))>>16).astype('<u2')
    def normal(a):
        b=np.asarray(a,dtype='<f4').view('<u4')
        require(np.all(np.isfinite(a)) and not np.any((((b>>23)&255)==0)&((b&0x7fffffff)!=0)),
                'Qualified finite normal-or-zero arithmetic domain')
    require(nominal_output.shape==(2,5120) and nominal_output.dtype==np.dtype('<u2'),'Original nominal prompt output rows0/1')
    required={'local_bounds','exact_dot','norm_gate','nonlinear_gate','preprocess','gated','recurrent_step','recurrent_check','recurrent_replay'}
    require(required<=math.keys() and all(callable(math[k]) for k in required),'Exact qualified arithmetic dependencies')
    weight_items={a['prepared']['file']:a for a in copies['uploads'] if a['symbol']=='weights'}
    require(len(weight_items)==190,'Every bound original physical weight ROI')
    # Each ROI-to-logical-tile edge is part of the frozen host plan.
    weight_owners={}
    for item in weight_items.values():
        for segment in item['prepared']['segments']:
            for offset in range(segment['tile_count']):
                key=(segment['logical_stripe'],segment['ordinal_begin']+offset)
                require(key not in weight_owners,'One prepared original tile owner')
                weight_owners[key]=(item['prepared']['file'],segment['destination_y']+offset,segment['destination_x'])
    require(len(weight_owners)==31548,'All original matrix tiles bound')
    gains=[prepared.array(n)[0].reshape(-1) for n in ('input-norm.npy','post-norm.npy')]
    conv_weights=prepared.array('convolution-weights.npy')[0]
    head_gains=prepared.array('head-gains.npy')[0];head_parameters=prepared.array('head-parameters.npy')[0]
    prior_history=np.zeros((80,128,4),dtype='<u2')
    ideal_states=[[0.0]*16384 for _ in range(48)];state_bounds=[[0.0]*16384 for _ in range(48)]
    exact_states=[[0.0]*16384 for _ in range(48)]
    rows=samples=0;reports=[];first_output=None
    for step in plan['schedule']:
        serial=step['serial'];position=step['position']
        if serial==3:
            prior_history.fill(0);ideal_states=[[0.0]*16384 for _ in range(48)]
            state_bounds=[[0.0]*16384 for _ in range(48)];exact_states=[[0.0]*16384 for _ in range(48)]
        def at(name,xy):return evidence.coordinate(serial,name,*xy)
        def owners(name,points):return np.stack([at(name,p) for p in points])
        mlp_points=plan['support']['MLP_owners'];head_points=[h['owner'] for h in plan['heads']]
        vectors={1:owners('head_casts',head_points)[:,256:].reshape(-1),3:owners('mlp_product',mlp_points).reshape(-1)}
        for phase in (0,2):
            stripe=next(s for s in plan['stripes'] if s['phase']==phase)
            a=evidence.rectangle(serial,'active_input',stripe['x'],stripe['y'],1,stripe['columns'],96).reshape(-1)
            require(not np.any(a.view('<u4')&65535),'Norm vector consists of expanded actual BF16')
            vectors[phase]=(a[:5120].view('<u4')>>16).astype('<u2')
        projections={r['name']:np.zeros(r['shape'][0],dtype='<u2') for r in plan['matrix_regions']};covered={n:set() for n in projections}
        for ordinal,stripe in enumerate(plan['stripes']):
            width=stripe['columns'];x,y=stripe['x'],stripe['y'];phase=stripe['phase']
            def block(name,count):return evidence.rectangle(serial,name,x,y,1,width,count).transpose(1,0,2)
            active=block('active_input',96);wanted=np.zeros((1,width,96),dtype='<f4')
            source=vectors[phase];wanted.reshape(-1)[:len(source)]=decode(source)
            equal(active,wanted,'Every projection PE receives the actual phase vector and zero tail')
            packed=np.empty((1,width,6144),dtype='<u4')
            grouped={}
            for logical_ordinal in range(width):
                filename,iy,ix=weight_owners[ordinal,logical_ordinal]
                grouped.setdefault(filename,[]).append((logical_ordinal,iy,ix))
            for filename,points in grouped.items():
                source,_=prepared.array(filename)
                for logical_ordinal,iy,ix in points:packed[0,logical_ordinal]=source[iy,ix]
            weights=packed.view('<u2').reshape(1,width,96,128)
            low,high=math['local_bounds'](weights,active)
            partial=block('partial',128);normal(partial)
            require(np.all(partial.astype(float)>=low) and np.all(partial.astype(float)<=high),'Every actual96FMA row conditional enclosure')
            rows+=partial.size
            for xx in range(width):
                chosen=((y+xx)*37+(x+plan['physical_x_offset'])*17+serial*13)%stripe['valid_rows']
                exact=math['exact_dot'](weights[0,xx,:,chosen],(active[0,xx].view('<u4')>>16).astype('<u2'))
                require(exact==int(partial[0,xx,chosen:chosen+1].view('<u4')[0]),'Predetermined exact96FMA sample on every actual matrix PE')
                samples+=1
            reduced=partial[:,-1].copy()
            for xx in range(width-2,-1,-1):reduced=np.add(reduced,partial[:,xx],dtype='<f4');normal(reduced)
            equal(block('result',128),np.broadcast_to(reduced[:,None],(1,width,128)).copy(),'Exact right-to-left reduction and result broadcast')
            rounded=at('rounded',[x,y]);equal(rounded,encode(reduced[0]),'Every root BF16 RNE conversion')
            valid=stripe['valid_rows'];require(not np.any(rounded[valid:]),'Zero padded projection rows')
            group=stripe['group'];name=stripe['tensor'];require(group not in covered[name],'Disjoint projection group')
            covered[name].add(group);projections[name][128*group:128*group+valid]=rounded[:valid]
        require(all(len(covered[r['name']])==r['row_groups'] for r in plan['matrix_regions']),'Complete eight original projection matrices')
        def projection(name):return projections[f"model.language_model.layers.{plan['layer']}."+name+'.weight']
        qkv=projection('linear_attn.in_proj_qkv').reshape(80,128)
        conv_points=[c['owner'] for c in plan['convolution']]
        conv_input=owners('conv_input',conv_points);equal(conv_input,qkv,'All10240 original QKV projection handoffs')
        prior_history[:,:,:3]=prior_history[:,:,1:];prior_history[:,:,3]=conv_input
        histories=owners('conv_history',conv_points).reshape(80,128,4)
        equal(histories,prior_history,'Every retained convolution tap and causal shift')
        conv={n:owners('conv_'+n,conv_points) for n in ('stages','casts','unscaled','output','stats')}
        equal(conv['output'][16:],conv['unscaled'][16:],'Unscaled K/V outputs')
        require(not np.any(conv['stats'][32:]),'Value convolution has no normalization statistics')
        heads=[];sign_differences=0
        for index,head in enumerate(plan['heads']):
            xy=head['owner'];channels=[index//3,16+index//3,32+index]
            query,key,value=(at('head_'+n,xy) for n in ('query','key','value'))
            for v,c in zip((query,key,value),channels):equal(v,conv['output'][c],'Exact conv-to-head FP32 fanout')
            z=at('head_raw_gate',xy);equal(z,projection('linear_attn.in_proj_z')[128*index:128*(index+1)],'Original Z projection handoff')
            gates=at('head_gate_stages',xy);ab=np.array([projection('linear_attn.in_proj_a')[index],projection('linear_attn.in_proj_b')[index],*head_parameters[index]],dtype='<u2')
            equal(gates[11:],decode(ab),'Original A/B projections and unchanged scalar parameters')
            stages=conv['stages'][channels];casts=conv['casts'][channels]
            token=dict(zip(('a','b','A_log','dt_bias'),map(float,gates[11:])));token['zero_qkv']=False
            observed=dict(conv=stages[:,:128].reshape(-1).tolist(),activation=stages[:,128:256].reshape(-1).tolist(),
                activation_exp=stages[:,256:].reshape(-1).tolist(),conv_bits=casts[:,:128].reshape(-1).astype(int).tolist(),
                activation_bits=casts[:,128:].reshape(-1).astype(int).tolist(),q=query.tolist(),q_normalized=conv['unscaled'][channels[0]].tolist(),
                k=key.tolist(),v=value.tolist(),q_stats=conv['stats'][channels[0]].tolist(),k_stats=conv['stats'][channels[1]].tolist(),
                gates=gates.tolist(),beta_bits=int(at('head_beta_bits',xy)[0]))
            pre=math['preprocess'](token,decode(conv_weights[channels]).reshape(-1).tolist(),decode(histories[channels]).reshape(-1).tolist(),observed)
            require(pre['passed'],'All conv/preprocess/gate operator bounds')
            shards=[s['owner'] for s in head['state_shards']]
            for j,s in enumerate(shards):
                equal(at('recurrent_query',s),query,'Full query delivered to every value shard')
                equal(at('recurrent_key',s),key,'Full key delivered to every value shard')
                equal(at('recurrent_value',s),value[32*j:32*(j+1)],'Exact recurrent value shard')
            rec_token=dict(q=query.tolist(),k=key.tolist(),v=value.tolist(),beta=float(gates[10]),decay=float(gates[9]))
            actual=dict(state=np.concatenate([at('recurrent_matrix',s).reshape(128,32) for s in shards],axis=1).reshape(-1),
                **{name:owners('recurrent_'+symbol,shards).reshape(-1) for name,symbol in [('prediction','prediction'),('delta','delta'),('output','output')]})
            conditional=math['recurrent_step'](ideal_states[index],state_bounds[index],rec_token)
            exact=math['recurrent_replay'](exact_states[index],rec_token);rec_checks={}
            for name in ('prediction','delta','state','output'):
                checked=math['recurrent_check'](actual[name].tolist(),conditional[name],conditional[name+'_bounds'])
                require(checked['passed'],'Full128 recurrent conditional enclosure: '+name)
                wanted=np.asarray(exact[name],dtype='<f4')
                require(np.array_equal(actual[name],wanted),'Exact integer numeric full128 recurrent order: '+name)
                signs=int(np.count_nonzero((actual[name]==0)&(wanted==0)&(actual[name].view('<u4')!=wanted.view('<u4'))))
                sign_differences+=signs;rec_checks[name]=dict(passed=True,zero_sign_differences=signs)
            rounded=owners('recurrent_output_bf16',shards).reshape(-1)
            equal(rounded,encode(actual['output']),'Every recurrent output BF16 boundary')
            equal(at('head_recurrent_output',xy),rounded,'All four recurrent shards joined by head')
            hs=at('head_stages',xy).reshape(5,128);hc=at('head_casts',xy).reshape(3,128)
            gated=math['gated'](dict(x=decode(rounded).tolist(),w=decode(head_gains[index]).tolist(),z=decode(z).tolist()),
                dict(stats=at('head_stats',xy).tolist(),norm=hs[0].tolist(),norm_bits=hc[0].astype(int).tolist(),
                    gain_pre=hs[1].tolist(),gain_bits=hc[1].astype(int).tolist(),exp=hs[2].tolist(),gate=hs[3].tolist(),
                    precast=hs[4].tolist(),output_bits=hc[2].astype(int).tolist()))
            require(gated['passed'],'Complete original gated RMS head')
            ideal_states[index]=conditional['state'];state_bounds[index]=conditional['state_bounds'];exact_states[index]=exact['state']
            heads.append(dict(head=index,preprocess=pre,gated=gated,recurrent=rec_checks))
        prexy,postxy=plan['support']['input_norm'],plan['support']['post_norm']
        saved=[at('norm_saved',xy) for xy in (prexy,postxy)];hidden=[at('hidden',xy) for xy in (prexy,postxy)]
        original=prepared.input_row(step['input_row']).reshape(-1);equal(evidence.input(serial),original,'Saved original H2D input')
        equal(saved[0],original,'Actual input norm preserves the original input')
        residual=encode(np.add(decode(saved[0]),decode(projection('linear_attn.out_proj')),dtype='<f4'))
        equal(hidden[0],residual,'Full first residual');equal(saved[1],residual,'First residual feeds actual post norm')
        final=encode(np.add(decode(saved[1]),decode(projection('mlp.down_proj')),dtype='<f4'));equal(hidden[1],final,'Full final residual')
        norms=[math['norm_gate'](saved[i],gains[i],at('norm_stats',xy),vectors[2*i]) for i,xy in enumerate((prexy,postxy))]
        peer={name:owners(name,mlp_points) for name in ('mlp_gate','mlp_up','mlp_product','mlp_silu','mlp_exponential','mlp_sigmoid','mlp_product_fp32')}
        peer['mlp_activation']=owners('mlp_activation_fp32',mlp_points)
        equal(projection('mlp.gate_proj'),peer['mlp_gate'].reshape(-1),'All17408 actual gate values')
        equal(projection('mlp.up_proj'),peer['mlp_up'].reshape(-1),'All17408 actual up values')
        nonlinear=math['nonlinear_gate'](peer)
        if serial==1:first_output=final.copy()
        elif serial==3:equal(final,first_output,'Exact reset first-position final output replay')
        reference=nominal_output[step['input_row']];difference=np.abs(decode(final).astype(float)-decode(reference).astype(float))
        reports.append(dict(serial=serial,position=position,heads=heads,norms=norms,MLP=nonlinear,
            recurrent_zero_sign_differences=sign_differences,nominal_BF16_mismatches=int(np.count_nonzero(final!=reference)),
            nominal_max_absolute_error=float(np.max(difference)),actual_hidden_sha256=hashlib.sha256(final.tobytes()).hexdigest()))
    require(rows==3*31548*128 and samples==3*31548,'Every original matrix PE/row/serial checked')
    return dict(status='all_operator_gates_passed',layer=plan['layer'],serials=3,positions=2,results=reports,
        all_conditional_matrix_rows=rows,exact_FMA_samples=samples,exact_sample_FMA_slots=samples*96,
        exact_sample_formula='((actual_y*37)+(actual_x*17)+(serial*13))%valid_rows',
        full128_recurrent_numeric_exact=True,zero_sign_differences_reported_separately=True,
        recurrent_beta_decay_evidence='Validated head source values plus complete actual prediction/delta/state/output replay; shard scalars are not exported or claimed read back.',
        original_projection_and_neural_handoffs=True,nominal_original_outputs_compared=True,
        source_propagated_whole_layer_enclosure=False,full_model=False,seconds=time.monotonic()-started)
