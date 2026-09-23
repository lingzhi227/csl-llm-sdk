"""Bounded complete head uploads, retained arithmetic and21 native logit reads."""
from stage_contract import require, MAX_COPY_HOST_BYTES
from weight_upload import BANDS


def role(region,x,y):
    local=x-region['x']
    require(0<=local<95 and 0<=y<1160,'Head role coordinate')
    if local==0 and y==3:return 2,0,0
    if 1<=local<=93 and y>=4:
        row,ordinal=divmod(y-4,55);group=(local-1)*21+row
        if row<21 and ordinal<54 and group<1940:return 1,ordinal,group
    return 0,0,0


def owns(symbol,declared):
    r,ordinal,_=declared
    if symbol in ('state','head_transport','checkpoint_control'):return True
    if symbol in ('weights','descriptor','head_partial','head_result','head_active_input'):return r==1
    if symbol=='head_logits':return r==1 and ordinal==0
    if symbol in ('head_normalized','head_norm_input','head_norm_weights','head_norm_stats','handoff_status'):return r==2
    return False


def build(region):
    x=region['x']
    out={k:[] for k in ('uploads','state_copies','transport_copies','diagnostic_copies','checkpoint_copies','handoff_copies','head_logits')}
    def item(symbol,dtype,px,y,width,height,count,**extra):
        value=dict(symbol=symbol,dtype=dtype,bits=16 if dtype=='u16' else 32,x=px,y=y,
                   width=width,height=height,count=count,family='head',layer_id=64,**extra)
        require(width*height*count*4<=MAX_COPY_HOST_BYTES,'Existing16MiB copy bound includes every head transfer')
        return value
    def upload(symbol,dtype,px,y,width,height,count,name,category,**description):
        value=item(symbol,dtype,px,y,width,height,count,category=category)
        native_bytes=width*height*count*value['bits']//8
        source=dict(file=name,symbol=symbol,dtype=dtype,x=px,y=y,width=width,height=height,count=count,
                    order='ROW_MAJOR',host_shape=[height,width,count],host_bytes=width*height*count*4,
                    native_bytes=native_bytes,source_layer=64,source_namespace='original-final-head',**description)
        value['prepared']=source;out['uploads'].append(value)
    for row in range(21):
        width=93 if row<8 else 92;start=4+55*row
        for band_start,band_stop in BANDS:
            first=max(start,band_start);stop=min(start+54,band_stop)
            if first>=stop:continue
            cap=MAX_COPY_HOST_BYTES//(width*6144*4)
            for y in range(first,stop,cap):
                height=min(cap,stop-y)
                upload('weights','u32',x+1,y,width,height,6144,f'head-weights-r{row:02d}-y{y:04d}.npy',
                       'matrix_uploads',owner_row=row,ordinal_begin=y-start,original_tensor='lm_head.weight')
        upload('descriptor','u16',x+1,start,width,54,6,f'head-descriptor-r{row:02d}.npy','config_uploads',owner_row=row)
        for symbol,count in [('head_partial',128),('head_result',128),('head_active_input',96)]:
            out['diagnostic_copies'].append(item(symbol,'f32',x+1,start,width,54,count))
    upload('head_norm_weights','u16',x,3,1,1,5120,'head-final-norm.npy','parameter_uploads',original_tensor='model.language_model.norm.weight')
    out['state_copies'].append(item('state','u32',x,0,95,1160,16))
    out['transport_copies'].append(item('head_transport','u32',x,0,95,1160,32))
    for name in ('head_norm_input','head_normalized'):
        value=item(name,'u16',x,3,1,1,5120)
        out['diagnostic_copies'].append(value)
        out['checkpoint_copies'].append(dict(value))
    out['diagnostic_copies'].append(item('head_norm_stats','f32',x,3,1,1,4))
    out['handoff_copies'].append(item('handoff_status','u32',x,3,1,1,12))
    for roi in region['readout']:
        out['head_logits'].append(item('head_logits','u16',roi['x'],roi['y'],roi['width'],1,128,owner_row=roi['owner_row']))
    weights=[p for p in out['uploads'] if p['symbol']=='weights']
    require(sum(p['prepared']['native_bytes'] for p in weights)==region['packed_matrix_weight_bytes'],
            'Full padded original head matrix exactly once')
    return out
