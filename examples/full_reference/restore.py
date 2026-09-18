"""Restore all official cache buffers from saved stage evidence and compare bits."""
import hashlib


def restore(namespace,layer_types,evidence,step,old_cache,position,check):
    import numpy as np
    import torch
    from pinned import original_cache
    restored=original_cache(namespace,layer_types);count=0;size=0
    for i,kind in enumerate(layer_types):
        check();name=f'step{step}-layer{i}-state.npz';path=evidence.root/name
        spec=evidence.files[name]
        if path.stat().st_size!=spec['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest()!=spec['sha256']:
            raise ValueError('Saved reference state identity changed')
        with np.load(path,allow_pickle=False) as archive:
            values={}
            for key in archive.files:
                array=archive[key]
                if spec['dtypes'][key]=='torch.bfloat16':
                    if array.dtype!=np.dtype('<u2'):raise ValueError('BF16 state encoding')
                    value=torch.from_numpy(array.view('<i2').copy()).view(torch.bfloat16)
                elif spec['dtypes'][key]=='torch.float32':
                    if array.dtype!=np.dtype('<f4'):raise ValueError('FP32 state encoding')
                    value=torch.from_numpy(array.copy())
                else:raise ValueError('Unsupported state type')
                values[key]=value;count+=1;size+=array.nbytes
        original=old_cache.layers[i]
        if kind=='full_attention':
            if set(values)!={'keys','values'}:raise ValueError('KV state fields')
            restored.update(values['keys'],values['values'],i)
            pairs=[(restored.layers[i].keys,original.keys),(restored.layers[i].values,original.values)]
        else:
            if set(values)!={'conv','recurrent'}:raise ValueError('Recurrent state fields')
            restored.update_conv_state(values['conv'],i,conv_kernel_size=4)
            restored.update_recurrent_state(values['recurrent'],i)
            pairs=[(restored.layers[i].conv_states[0],original.conv_states[0]),
                (restored.layers[i].recurrent_states[0],original.recurrent_states[0])]
            if restored.has_previous_state(i,0)!=old_cache.has_previous_state(i,0):raise ValueError('Restored recurrence valid flag')
        for a,b in pairs:
            if a.dtype!=b.dtype or a.shape!=b.shape or not torch.equal(a.view(torch.uint8),b.view(torch.uint8)):
                raise ValueError('Reference state restore bit mismatch')
    if restored.get_seq_length()!=position or count!=2*len(layer_types):raise ValueError('Restored position/coverage')
    return restored,dict(buffers=count,bytes=size,bitwise_equal_to_live_cache=True,position=position)
