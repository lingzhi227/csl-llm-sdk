"""Exact expected connected state/frame vectors; no SDK or original input reads."""
GUARDS=(0xa55aa55a,0x5aa55aa5)
MAGIC_DATA=0x4d4c5044
MAGIC_ACK=0x4d4c5041
VERSION=2
INPUT_INDICES=(0,1,3)
COMPLETION_MASKS=(7,7,31,25)
RETAINED_MASKS=(1,2,15,24)
COUNTS=((1,1,0,0),(1,1,0,0),(1,1,2,2),(0,0,1,1))
FIRST_EDGES=(0,1,0,2)

def require(condition,message):
    if not condition:raise ValueError(message)

def identity(generation,*,initialize=False):
    require(type(generation) is int and (generation in (1,2,3) or initialize and generation==0),'Exact connected generation')
    return (0,0xffffffff,0) if generation==0 else (generation,INPUT_INDICES[generation-1],generation)

def scope(generation,*,initialize=False):return (*identity(generation,initialize=initialize),0,112,0,128,0,128)

def key(generation,input_index,contraction_id,phase,name,pe,tile=None):
    require(type(generation) is int and generation in (0,1,2,3),'Evidence generation')
    expected=(0,None,0) if generation==0 else identity(generation)
    require((generation,input_index,contraction_id)==expected,'Evidence generation/input/contraction binding')
    return 'g'+str(generation)+'_i'+str(input_index)+'_c'+str(contraction_id)+'__'+phase+'__'+name+'__pe'+str(pe)+('__tile'+str(tile) if tile is not None else '')

def descriptor(generation,edge):
    require(type(edge) is int and edge in (0,1,2),'Exact connected edge')
    return (*identity(generation),edge,edge,0,128,0,128,128,0,112)

def words32(words):
    require(all(type(x) is int and 0<=x<=0xffffffff for x in words),'Exact unsigned32 words')

def packet(generation,edge,body):
    words32(body);require(len(body)==128,'Exactly128 data payload words')
    require(all(x>>16==0 and x&0x7f80!=0x7f80 for x in body),'Finite low16 BF16 payload')
    return (GUARDS[0],MAGIC_DATA,VERSION,*descriptor(generation,edge),*body,GUARDS[1])

def acknowledgement(generation,edge,error=0):
    require(type(error) is int and 0<=error<=0xffffffff,'ACK status word')
    return (GUARDS[0],MAGIC_ACK,VERSION,*descriptor(generation,edge),error,GUARDS[1])

def verify_packet(generation,edge,words):
    words32(words);require(len(words)==144,'Exact version2 data frame length')
    require(tuple(words)==packet(generation,edge,words[15:143]),'Full current version2 frame identity/guards')

def verify_ack(generation,edge,words):
    words32(words);require(tuple(words)==acknowledgement(generation,edge),'Full current successful ACK before release')

def numeric_state(generation,pe,*,phase=0,tiles=0,consumed=0,committed=False,released=False):
    require(pe in (0,1,2,3),'Numeric PE role')
    g,i,c=identity(generation,initialize=True); total=112 if pe<2 else 128
    last=(112 if pe<2 else 128 if pe==2 else 64) if tiles else 0
    active=tiles>0 or phase==1
    return [phase,g,i,c,total,tiles,consumed,g if committed else 0,0,last,
        tiles,int(active or committed),int(committed),int(released),pe,0,0,112,128,0]

def final_numeric_state(generation,pe,*,released=False):
    return numeric_state(generation,pe,phase=4 if released else 2,tiles=2 if pe==3 else 1,
        consumed=112 if pe<2 else 128,committed=True,released=released)

def mlp_state(generation,pe,*,incoming=0,computed=False,tiles=0,released=False):
    g,i,c=identity(generation,initialize=True)
    return [2 if computed else 1 if incoming else 0,g,i,c,incoming,
        128 if computed and pe==2 else 0,128 if computed and pe==2 else 0,
        128 if incoming and pe==3 else 0,tiles if pe==3 else 0,64*tiles if pe==3 else 0,
        128 if computed else 0,g if computed else 0,0,int(released),pe,0,0,112,0,128]

def initial_handoff_state(generation):
    g,i,c=identity(generation,initialize=True)
    a=[0]*34;a[1:4]=[g,i,c];a[25:29]=[0,128,0,128];a[32:34]=[0,112]
    return a

def completed_handoff_state(generation,edge,pe,*,receipt_first=False,command_before_ack=False,released=False):
    require(pe in (edge,2 if edge<2 else 3),'Participating PE only')
    sender=pe==edge; a=initial_handoff_state(generation)
    a[0]=5 if released else 4;a[4]=edge;a[6]=1;a[11]=1;a[13:15]=[142,15]
    a[24:29]=[edge,0,128,0,128];a[29]=int(released);a[31]=1
    base=(6*edge if pe==2 else 0)
    a[17]=base+1
    if sender:
        a[5]=3 if edge==2 else 0;a[15]=2 if edge==2 else 0
        a[9]=a[10]=1;a[18]=base+2;a[22]=base+3;a[23]=base+4
    else:
        a[5]=(1,3,4)[edge];a[15]=2 if edge==1 else 1;a[7]=a[8]=1
        if receipt_first:
            a[19]=base+2;a[20]=base+3
            # The receive/commit callback is atomic; the command can occur before
            # or after the later asynchronous ACK-send completion callback.
            a[18]=base+(4 if command_before_ack else 5)
            a[21]=base+(5 if command_before_ack else 4)
        else:
            a[18]=base+2;a[19]=base+3;a[20]=base+4;a[21]=base+5
        a[23]=base+6;a[30]=int(receipt_first)
    a[16]=a[23]
    return a

def tile_snapshot(generation,pe,tile):
    require(pe in (0,1,3) and type(tile) is int and tile in ((0,) if pe<2 else (0,1)),'Exact connected numeric tile')
    g,i,c=identity(generation);width,total=(112,112) if pe<2 else (64,128)
    return [0xa55a,0x5aa5,0xa55a,0x5aa5,g,tile,tile*width,width,tile+1,(tile+1)*width,1,1,i,c,total,0]

def weight_snapshot(generation,pe,*,reset=False):
    require(pe in (0,1,3),'Active numeric weight snapshot')
    g,i,c=identity(generation);total=112 if pe<2 else 128
    return [0xa55a,0x5aa5,g,i,c,total,0,128,0 if reset else 2,
        0 if reset else 1 if pe<2 else 2,0 if reset else total,0 if reset else g,
        2*g-1 if reset else 2*g,1,pe,0]

def lifecycle_state(generation,pe,*,stage):
    require(pe in (0,1,2,3) and stage in ('initialize','reset','retention','released'),'Exact lifecycle observation')
    g,i,c=identity(generation,initialize=True);a=[0]*32;a[1:10]=scope(g,initialize=True);a[31]=pe
    if stage=='initialize':
        require(g==0,'Epoch0 initialization only');a[11]=1;a[15]=FIRST_EDGES[pe];return a
    require(g>0,'Live lifecycle epoch')
    done=stage in ('retention','released');released=stage=='released'
    a[0]=3 if released else 2 if done else 1
    a[10]=g-1;a[11]=g+1;a[12]=a[13]=g;a[14]=g if released else g-1
    a[15]=3 if done else FIRST_EDGES[pe];a[16]=8191;a[17]=g
    a[18]=(4*g if done else 4*g-3) if pe<2 else 0
    a[22:26]=[v*(g if done else g-1) for v in COUNTS[pe]]
    a[26]=RETAINED_MASKS[pe] if done else 0;a[27]=g if released else g-1
    a[28]=COMPLETION_MASKS[pe] if done else 0;a[29]=2*g-1;a[30]=2*g if released else 2*g-2
    return a
