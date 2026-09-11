"""One concrete resident MLP fragment, not a full-model temporal scheduler."""
from dataclasses import dataclass

@dataclass(frozen=True)
class Tile:
    rank:int
    role:str
    x:int
    y:int
    columns:int=0
    shard:int=0
    @property
    def matrix(self):return self.columns>0
    @property
    def root(self):return self.matrix and self.shard==0

TILES=(Tile(0,'gate',0,0,96),Tile(1,'gate',1,0,96,1),
       Tile(2,'up',2,0,96),Tile(3,'up',3,0,96,1),
       Tile(4,'ingress',0,1),Tile(5,'nonlinear',1,1),
       Tile(6,'down',2,1,64),Tile(7,'down',3,1,64,1))
BEGIN,READY,DATA,DONE,RELEASE,RELEASED=range(1,7)
PAYLOAD_WORDS=27
PARTICIPANT_MASK=0xef  # Every rank except ingress rank4.

def arrays(tile):
    """Exported arrays, including explicit endpoint guards; count, scalar type."""
    m=tile.matrix;n=tile.columns;non=tile.role=='nonlinear';ing=tile.role=='ingress'
    return dict(weights=(128*n+2 if m else 1,'u16'),
        request=(194 if ing else 1,'u16'),output=(130 if ing else 1,'u16'),
        input=(n+2 if m else 1,'f32'),partial=(130 if m else 1,'f32'),
        reduced=(130 if m else 1,'f32'),rounded=(130 if tile.root else 1,'u16'),
        broadcast=(192 if ing else 128 if non else 1,'f32'),
        gate=(130 if non else 1,'u16'),up=(130 if non else 1,'u16'),
        silu=(130 if non else 1,'u16'),product=(130 if non else 1,'u16'),
        exponential=(130 if non else 1,'f32'),sigmoid=(130 if non else 1,'f32'),
        activation_fp32=(130 if non else 1,'f32'),product_fp32=(130 if non else 1,'f32'),
        state=(32,'u32'),timing=(6,'u16'))

def ledger():
    sizes={'u16':2,'f32':4,'u32':4};tiles=[]
    for t in TILES:
        exported=sum(n*sizes[k] for n,k in arrays(t).values())
        # Non-exported tx/rx packet arrays, timestamps and finite scalar/DSD
        # allowance are charged separately. Actual ordinary ELF end is mandatory.
        hidden=248+12+512+(512 if t.matrix else 0)
        code=14336;stack=4096;total=exported+hidden+code+stack
        if total>48128:raise ValueError('Proposed resident SRAM envelope')
        tiles.append(dict(**t.__dict__,arrays=arrays(t),exported_bytes=exported,
            hidden_data_allowance=hidden,code_SDK_allowance=code,stack_bytes=stack,
            proposed_upper_bytes=total,actual_ELF_measured=False))
    return dict(status='source_only_uncompiled',shape=[192,128,128],application=[4,2],
        simulated_fabric=[11,4],offset=[4,1],tiles=tiles,
        initial_weight_bytes=131072,steady_token_weight_H2D_bytes=0,
        host_stage_launches_per_input=1,host_neural_intermediate_copies=0,
        stage_input_BF16_bytes=384,stage_output_BF16_bytes=256,
        native16_host_input_output_bytes=1280,
        device_broadcast_FP32_words=320,
        device_pair_reduction_FP32_words=384,
        device_pair_result_multicast_FP32_words=384,
        device_BF16_payload_words=192,device_data_packets=9,
        device_control_packets=35,device_application_packet_words=368,
        packet_network_headers_excluded=True,
        per_projection_local_FMA_depth=96,per_down_local_FMA_depth=64,
        FP32_pair_add_levels=1,three_pair_collectives=True,
        full_model_tree_routing_qualified=False,
        resident_assumption='parallel resident shards, device joins and consumer ownership across repeated inputs')

def layout_source():
    lines=['// Resident spatial fragment. Source only: no compiled-fit claim.',
           'const memcpy=@import_module("<memcpy/get_params>",.{.width=4,.height=2});',
           'layout{@set_rectangle(4,2);']
    for t in TILES:
        lines.append(f'@set_tile_code({t.x},{t.y},"tile.csl",.{{.memcpy_params=memcpy.get_params({t.x}),.rank={t.rank}}});')
        if t.rank==4:route='.rx=.{RAMP},.tx=.{NORTH}'
        elif t.y==0:route='.rx=.{'+('SOUTH' if t.x==0 else 'WEST')+'},.tx=.{RAMP'+(',EAST' if t.x<3 else '')+'}'
        else:route=None
        if route:
            filt=(f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter={(192-t.shard*96)%192},.max_counter=95,.limit1=191}}' if t.y==0 else '')
            lines.append(f'@set_color_config({t.x},{t.y},@get_color(3),.{{.routes=.{{{route}}}{filt}}});')
        if t.rank in (5,6,7):
            route='.rx=.{RAMP},.tx=.{EAST}' if t.rank==5 else '.rx=.{WEST},.tx=.{RAMP'+(',EAST' if t.rank==6 else '')+'}'
            filt=(f',.filter=.{{.kind=.{{.counter=true}},.count_data=true,.init_counter={(128-t.shard*64)%128},.max_counter=63,.limit1=127}}' if t.rank!=5 else '')
            lines.append(f'@set_color_config({t.x},{t.y},@get_color(4),.{{.routes=.{{{route}}}{filt}}});')
    for name,(_,kind) in arrays(TILES[0]).items():lines.append(f'@export_name("{name}",[*]{kind},false);')
    for name in ('initialize','stage'):lines.append(f'@export_name("{name}",fn()void);')
    return '\n'.join(lines+['}'])+'\n'
