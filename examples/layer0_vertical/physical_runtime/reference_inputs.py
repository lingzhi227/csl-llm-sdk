"""Prepare the first two original Layer0 input rows remotely, without a forward.

Step0 contains five prompt positions. Step1 starts at position5 and cannot be
used as position1 with reset-zero recurrent/convolution state. The current
device schedule computes positions0,1,reset0 and evolves its own state.
"""
from pathlib import Path
import hashlib,io

NAME='step0-layer0-input.npz'
PIN=dict(bytes=51466,sha256='c3b204b303a45e9981c655ecac2d836e3b7d88d1660fb6b8a4f4ff88920bf5c9')
SCHEDULE=[dict(serial=1,request=1,reset_generation=1,position=0,input_row=0),
          dict(serial=2,request=1,reset_generation=1,position=1,input_row=1),
          dict(serial=3,request=1,reset_generation=2,position=0,input_row=0)]

def contract(plan):
    if plan['schedule']!=SCHEDULE:raise ValueError('Only actual first positions0,1 then reset0 are bound')
    if {k:plan['reference_inputs'][NAME][k] for k in PIN}!=PIN:
        raise ValueError('Exact accepted original Layer0 embedding-input archive')
    return dict(archive=NAME,pin=PIN,archive_shape=[1,5,5120],archive_dtype='<u2',
        selected_rows=[0,1],prepared_shape=[2,5120],schedule=SCHEDULE,
        initial_state='Device initializes convolution/recurrent state to zero at actual position0',
        position1_state='Retain actual position0 device state; no reference state injection',
        reset_state='Device resets actual state after quiescence, then replays original position0',
        prefix_state_upload_required_for_this_schedule=False,
        excluded_reference_steps=[1,2,3],
        reference_state_after_prompt_is_not_an_initial_state=True,
        reference_neural_intermediates_used_as_inputs=False,
        embedding_boundary_only=True,full_model_embedding_implemented=False,
        CPU_forward=False,SDK_invocations=0)

def load(reference_root,plan):
    """Read a hash-pinned small NPZ and return exactly two original BF16 rows.

    The caller owns remote resource and immutable-output guards. Importing this
    module on the Mac never imports NumPy or reads a reference array.
    """
    import numpy as np
    description=contract(plan);path=Path(reference_root)/NAME
    if path.is_symlink() or not path.is_file() or path.stat().st_size!=PIN['bytes']:
        raise ValueError('Original reference file identity')
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=PIN['sha256']:raise ValueError('Original reference bytes changed')
    with np.load(io.BytesIO(raw),allow_pickle=False) as archive:
        if archive.files!=['hidden']:raise ValueError('Exact original hidden archive entry')
        value=archive['hidden']
        if value.dtype!=np.dtype('<u2') or value.shape!=(1,5,5120):
            raise ValueError('Actual five-position prompt embedding boundary')
        result=np.ascontiguousarray(value[0,:2])
    if result.nbytes!=20480 or np.array_equal(result[0],result[1]):
        raise ValueError('Two distinct original input positions')
    description['selected_BF16_bytes']=result.nbytes
    description['selected_BF16_sha256']=hashlib.sha256(result.tobytes()).hexdigest()
    return result,description
