"""Account for every text tensor and state; do not claim a compiled layout."""
from collections import Counter
import argparse
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--bf16-rows',type=int,choices=[144,140],default=144);args=p.parse_args()
    d = json.loads((ROOT/"configs/tensors.json").read_text())
    c = json.loads((ROOT/"configs/config.json").read_text())["text_config"]
    selected = {k:v for k,v in d["tensors"].items() if not k.startswith(("model.visual.","mtp."))}
    counts = Counter(); assignments = {}; cursor = 0
    for name,s in selected.items():
        if name.endswith("weight_scale_inv"):
            continue
        shape = s["shape"]
        if s["dtype"]=="F8_E4M3":
            assert len(shape)==2 and shape[1]%128==0
            rows,columns=272,128
            grid=[math.ceil(shape[0]/rows),math.ceil(shape[1]/columns)]
            n=math.prod(grid); role="fp8_matrix"
            scale_name=name+"_scale_inv";scale=selected[scale_name]
            assert scale["shape"]==[math.ceil(shape[0]/128),math.ceil(shape[1]/128)]
            assert scale["dtype"]=="BF16"
            # Each tile covers at most three original 128-row scale blocks.
            # The row phase is (tile_row*272)%128 and MUST reach the kernel.
            assignments[scale_name]=dict(shared_with=name,original_bytes=scale["bytes"],
                replicated_fp32_scale_slots_per_pe=3,scale_index="[(tile_row*272+local_row)//128, tile_column]")
            stored=272*128+3*4
        elif len(shape)==2:
            rows,columns=args.bf16_rows,128;grid=[math.ceil(shape[0]/rows),math.ceil(shape[1]/columns)]
            n=math.prod(grid);role="bf16_matrix";stored=rows*128*2
        else:
            rows,columns=None,None;grid=None;n=math.ceil(s["bytes"]/32768);role="other";stored=32768
        counts[role]+=n
        assignments[name]=dict(role=role,shape=shape,original_bytes=s["bytes"],
            tile_rows=rows,tile_columns=columns,tile_grid=grid,pe_interval=[cursor,cursor+n],
            reserved_weight_bytes_per_pe=stored)
        cursor+=n
    assert set(assignments)==set(selected)
    assert sum(s["bytes"] for s in selected.values())==d["role_bytes"]["text"]
    linear=c["layer_types"].count("linear_attention"); full=c["layer_types"].count("full_attention")
    recurrent=linear*c["linear_num_value_heads"]*c["linear_key_head_dim"]*c["linear_value_head_dim"]*4
    conv=linear*(2*c["linear_num_key_heads"]*c["linear_key_head_dim"]+c["linear_num_value_heads"]*c["linear_value_head_dim"])*c["linear_conv_kernel_dim"]*2
    state=[]
    for tokens in [96,2048,8192]:
        kv=full*2*c["num_key_value_heads"]*c["head_dim"]*tokens*2
        state_pes=math.ceil(recurrent/32768)+math.ceil(conv/16384)+math.ceil(kv/24576)
        state.append(dict(tokens=tokens,recurrent_fp32_bytes=recurrent,conv_bf16_bytes=conv,kv_bf16_bytes=kv,
            logical_total_bytes=d["role_bytes"]["text"]+recurrent+conv+kv,
            minimum_state_pes_for_declared_budgets=state_pes,
            unassigned_pes_after_weight_and_state_budget=750*1160-cursor-state_pes))
    result=dict(repository=d["repository"],revision=d["revision"],scope="64 language layers and full vocabulary; text input; ordinary decoding",
        text_payload_bytes=d["role_bytes"]["text"],all_checkpoint_payload_bytes=d["total_payload_bytes"],
        excluded_from_text_execution={k:v for k,v in d["role_bytes"].items() if k!="text"},
        application_geometry=[750,1160],weight_pes=cursor,weight_pe_roles=dict(counts),contexts=state,
        assignments=assignments,
        status="Storage accounting only; PE intervals are abstract identifiers, not fabric coordinates",
        unresolved=["Actual BF16-role SRAM audit", "FP8 row-phase and padded tile qualification",
                    "Dynamic activation quantization and independent official reference",
                    "Physical placement and routing", "Communication buffers and code on weight PEs",
                    "All recurrent/attention/operator control programs", "Full-model numerical and physical acceptance"])
    result['bf16_tile_rows']=args.bf16_rows
    filename='capacity.json' if args.bf16_rows==144 else f'capacity-bf16-{args.bf16_rows}.json'
    (ROOT/'configs'/filename).write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps({k:v for k,v in result.items() if k!="assignments"},indent=2))


if __name__=="__main__":main()
