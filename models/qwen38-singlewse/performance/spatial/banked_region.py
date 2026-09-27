"""Resident-bank composition of the qualified restricted regional topology.

Capacity knobs account for communication roles. Actual ELF admission is mandatory;
this component does not place the complete model or implement BF16 computation.
"""
from dataclasses import dataclass
import re
from spatial.regional import RegionalPlan

@dataclass(frozen=True)
class BankedRegionalPlan:
    width:int=2
    height:int=3
    body_fp8_tiles:int=112
    boundary_fp8_tiles:int=110
    root_fp8_tiles:int=108
    bf16_tiles:int=12

    def base(self):return RegionalPlan(width=self.width,height=self.height)
    def capacity(self,x,y):
        if not (0<=x<self.width and 0<=y<self.height):raise ValueError('PE outside region')
        return self.root_fp8_tiles if x==y==0 else self.boundary_fp8_tiles if x==0 or y==0 else self.body_fp8_tiles
    def validate(self):
        r=self.base().validate()
        r.pop('live_bytes_estimate');r.pop('code_allowance')
        if not (4<=self.root_fp8_tiles<=self.boundary_fp8_tiles<=self.body_fp8_tiles<=112 and self.bf16_tiles==12):raise ValueError('Bank profile')
        r.update(full_model=False,compiled_sram_admitted=False,bank_counts=[[self.capacity(x,y) for x in range(self.width)] for y in range(self.height)],bf16_tiles=self.bf16_tiles,
            bank_bytes=[[self.capacity(x,y)*264+12*516 for x in range(self.width)] for y in range(self.height)],
            limitation='BF16 and descriptor-sized arrays are retained capacity buffers; no BF16 execution or complete matrix scheduler',
            admission='Combined ELF plus unchanged4096-byte stack must be <=48128; do not add separate fits')
        return r
    def document(self):
        d=self.base().document();d['schema']='wse-banked-regional-gemv-v1';d['resources']=self.validate();d['bank_counts']=d['resources']['bank_counts'];return d
    def emit_layout(self):
        self.validate();out=[]
        for line in self.base().emit_layout().splitlines():
            m=re.search(r'@set_tile_code\((\d+),(\d+),',line)
            if m:
                x,y=map(int,m.groups());needle=f'.memcpy_params=memcpy.get_params({x}),'
                line=line.replace(needle,needle+f'.bank_fp8_tiles={self.capacity(x,y)},')
            if line=='}':
                out.extend(['  @export_name("slot",[*]u16,false);','  @export_name("bf16_reserve",[*]u16,false);','  @export_name("metadata",[*]u32,false);'])
            out.append(line)
        return '\n'.join(out)+'\n'
