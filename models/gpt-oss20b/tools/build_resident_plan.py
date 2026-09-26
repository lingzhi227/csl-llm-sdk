"""Complete tensor-to-region storage plan; routing/code generation remain separate."""
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(context):
    tensors = json.loads((ROOT/"configs/tensors.json").read_text())
    if context < 128 or context > 8192:
        raise ValueError("Initial resident design supports context 128..8192")
    assignments = {}
    layers = []
    regions = []
    def region(name, x, width):
        r = dict(name=name, x=x, y=0, width=width, height=1160)
        regions.append(r)
        return r
    region("west_ingress", 0, 4)
    embedding = region("embedding", 4, 42)
    for layer in range(24):
        r = region(f"layer_{layer}", 46+26*layer, 26)
        prefix = f"model.layers.{layer}."
        cursor = 0
        for name, t in sorted(tensors.items()):
            if not name.startswith(prefix) or ".experts." in name:
                continue
            shape = t["shape"]
            if len(shape)==2:
                rows, columns = shape
                count = math.ceil(rows/128)*math.ceil(columns/96)
                spec = dict(role="bf16_matrix", logical_tile=[128,96], padded_weight_bytes_per_pe=24576)
            else:
                count = math.ceil(math.prod(shape)*2/24576)
                spec = dict(role="bf16_vector", tensor_bytes=math.prod(shape)*2)
            assignments[name] = dict(spec, region=r["name"], support_linear_begin=cursor,
                                     pes=count, coordinate_formula="(region.x + i//1160, i%1160)")
            cursor += count
        dense_and_vector_pes = cursor
        window = context if layer % 2 else 128
        kv_count = 8*math.ceil(window/96)
        kv = dict(role="bf16_kv", support_linear_begin=cursor, pes=kv_count,
                  slots_per_pe=96, bytes_per_pe=24576, head_dimension=64,
                  context=window, ring=layer%2==0)
        cursor += kv_count
        # Reserve actual compute/control PEs for normalization, QK/softmax,
        # token/controller state, router join, and shared scalar operations.
        attention_control = dict(support_linear_begin=cursor, pes=128)
        cursor += 128
        if cursor > 4*1160:
            raise ValueError("Attention support region capacity exceeded")
        for expert_part, row_tiles, start_column in (("gate_up_proj",36,4),("down_proj",18,15)):
            base = prefix+"mlp.experts."+expert_part
            block = tensors[base+"_blocks"]
            assert block["shape"] == [32,row_tiles*160,90,16]
            assert tensors[base+"_scales"]["shape"] == [32,row_tiles*160,90]
            assert tensors[base+"_bias"]["shape"] == [32,row_tiles*160]
            spec = dict(role="mxfp4_matrix", region=r["name"], experts=32,
                        logical_tile=[160,288], rows_per_expert=row_tiles, contraction_tiles=10,
                        pes=32*row_tiles*10, packed_bytes_per_pe=24480,
                        x_offset=start_column, expert_row_stride=36,
                        coordinate_formula="(region.x + x_offset + k_tile, 36*expert + row_tile)")
            assignments[base+"_blocks"] = dict(spec, component="packed_nibbles")
            assignments[base+"_scales"] = dict(spec, component="packed_scale_bytes", shares_pes_with=base+"_blocks")
            assignments[base+"_bias"] = dict(role="expert_bias", region=r["name"],
                shares_pes_with=base+"_blocks", contraction_tile=9, bytes_per_owner_pe=320,
                pes=32*row_tiles, owners="one original bias slice at the final contraction PE")
        layers.append(dict(layer=layer, region=r, support_pes_used=cursor,
             dense_and_vector_pes=dense_and_vector_pes, kv=kv, attention_control=attention_control,
             experts=dict(count=32, selected=4, stripe_height=36, gate_x_offset=4,
                          activation_x_offset=14, down_x_offset=15, join_x_offset=25),
             remaining_support_pes=4640-cursor))
    head = region("lm_head", 670, 42)
    region("east_control", 712, 4)
    region("east_egress_and_route_reserve", 716, 34)
    for name, r in (("model.embed_tokens.weight",embedding),("lm_head.weight",head)):
        shape=tensors[name]["shape"]
        count=math.ceil(shape[0]/128)*math.ceil(shape[1]/96)
        assert count <= r["width"]*1160
        assignments[name]=dict(role="bf16_matrix",region=r["name"],support_linear_begin=0,
            pes=count,logical_tile=[128,96],padded_weight_bytes_per_pe=24576,
            coordinate_formula="(region.x + i//1160, i%1160)")
    assignments["model.norm.weight"]=dict(role="bf16_vector",region="east_control",pes=1,tensor_bytes=5760)
    assert set(assignments)==set(tensors) and len(tensors)==459
    for i, a in enumerate(regions):
        assert a["x"]+a["width"]<=750
        for b in regions[i+1:]:
            assert a["x"]+a["width"]<=b["x"] or b["x"]+b["width"]<=a["x"]
    unique_weight_pes=sum(v["pes"] for v in assignments.values() if "shares_pes_with" not in v)
    kv_pes=sum(l["kv"]["pes"] for l in layers)
    return dict(status="complete storage assignment; not a compiled or routed full-model implementation",
        model="openai/gpt-oss-20b",revision="6cee5e81ee83917806bbde320786a8fb61efebee",
        application=[750,1160],physical_fabric=[762,1172],offset=[4,1],context=context,batch=1,
        tensor_count=len(tensors),original_weight_payload_bytes=sum(t["data_offsets"][1]-t["data_offsets"][0] for t in tensors.values()),
        assigned_unique_weight_pes=unique_weight_pes,kv_pes=kv_pes,
        scalar_kernel_actual_sram_with_stack=35200,
        scalar_kernel_scope="one MXFP4 160x288 tile; full expert communication code is not included",
        regions=regions,layers=layers,assignments=assignments,
        runtime_contract=dict(weight_uploads="once at initialization, all 32 experts in all 24 layers",
            request="token ID and position enter west; all neural operands remain on wafer",
            expert_dispatch="broadcast activation plus top-4 mask; only selected experts compute",
            expert_join="four distinct selected expert results per output tile and request epoch",
            output="east-side vocabulary projection and greedy token selection; D2H at east edge",
            persistent_state="all parameters plus per-layer KV caches survive request release"),
        remaining_gates=["actual whole-layout compiler task/queue/color ownership",
                        "all role ELF sections and dynamic buffer/stack bounds",
                        "complete attention and expert transpose/reduction routes",
                        "on-wafer all-layer continuous generation and numerical comparison"])


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--context",type=int,default=8192)
    p.add_argument("--output",type=Path,default=ROOT/"configs/resident-plan.json")
    args=p.parse_args()
    result=build(args.context)
    args.output.write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k not in ("regions","layers","assignments","runtime_contract","remaining_gates")}))
