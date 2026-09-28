"""Fresh full capture verification; no device access or candidate feedback."""
import numpy as np
from reference.frontend_oracle import expand,gated_interval
from reference.gdn_columns_oracle import step


def verify(actual,expected):
    total_state=total_core=total_frontend=0;ratio=relative=0.
    for replay,length in [(0,6),(1,2)]:
        state=np.zeros((16,3,128,128),np.float64);bound=np.zeros_like(state)
        for position in range(length):
            values={n:actual[f'{n}_{replay}_{position}'] for n in ['packet','history','core','output','state']}
            packet=values['packet'];observed=values['state'];core=values['core'];output=values['output']
            if packet.shape!=(16,3,387) or packet.dtype!=np.uint32:raise ValueError('Original frontend packet extent')
            if observed.shape!=(16,3,128,128) or observed.dtype!=np.float32:raise ValueError('Original recurrent state extent')
            if any(a.shape!=(16,3,128) or a.dtype!=np.uint16 for a in [core,output]):raise ValueError('Original BF16 output extent')
            if values['history'].shape!=(16,640,3) or values['history'].dtype!=np.uint16:raise ValueError('Original convolution history extent')
            np.testing.assert_array_equal(packet[:,:,0],position+1)
            p=packet[:,:,1:].copy().view(np.float32)
            old=np.concatenate((p[:,:,258:],p[:,:,130:258],p[:,:,2:130],p[:,:,:2]),axis=2)
            if not np.isfinite(old).all() or not np.all(np.abs(old.astype(np.float64)-expected[f'packet_{position}'])<=expected[f'bound_{position}']):raise ValueError('Original frontend interval')
            np.testing.assert_array_equal(packet[:,0,131:],packet[:,1,131:]);np.testing.assert_array_equal(packet[:,0,131:],packet[:,2,131:])
            np.testing.assert_array_equal(values['history'],expected[f'history_{position}'])
            for g in range(16):
                for h in range(3):
                    state[g,h],bound[g,h],lo,hi,_=step(state[g,h],bound[g,h],p[g,h])
                    diff=np.abs(observed[g,h].astype(np.float64)-state[g,h])
                    if not np.isfinite(observed[g,h]).all() or not np.all(diff<=bound[g,h]):raise ValueError('Propagated recurrent state interval')
                    ratio=max(ratio,float(np.max(diff/np.maximum(bound[g,h],1e-300))))
                    relative=max(relative,float(np.linalg.norm(diff)/max(np.linalg.norm(state[g,h]),1e-30)))
                    c=expand(core[g,h]);o=expand(output[g,h])
                    if not np.isfinite(c).all() or not np.all((c>=lo)&(c<=hi)):raise ValueError('Recurrent BF16 output interval')
                    lo,hi=gated_interval(core[g,h],expected[f'z_{position}'][g,128*h:128*(h+1)],expected['gain'][g])
                    if not np.isfinite(o).all() or not np.all((o>=lo)&(o<=hi)):raise ValueError('Gated BF16 output interval')
            if replay:
                for name,value in values.items():
                    np.testing.assert_array_equal(value.view(np.uint8),actual[f'{name}_0_{position}'].view(np.uint8))
            total_state+=observed.size;total_core+=core.size;total_frontend+=old.size
    return dict(passed=True,state_elements_verified=total_state,core_bf16_verified=total_core,gated_bf16_verified=total_core,
                frontend_fp32_verified=total_frontend,bitwise_reset_replay=True,exact_history=True,exact_shared_qk=True,
                maximum_state_error_bound_ratio=ratio,maximum_state_relative_l2=relative,
                host_injected_recurrent_results=False,full_model_speed_target_achieved=False)
