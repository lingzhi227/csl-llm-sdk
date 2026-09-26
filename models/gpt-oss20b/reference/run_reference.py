"""Run all 24 layers using unmodified OpenAI forward methods and lazy HF weights.

The original attention/MLP equations and BF16 operation boundaries are retained.
Only checkpoint name mapping and lazy indexed expert materialization are adapted.
This is a CPU oracle, not a CSL inference implementation.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import torch
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "core"), str(ROOT / "reference/upstream")]
from checkpoint import Checkpoint, decode_mxfp4
from gpt_oss.torch.model import ModelConfig, AttentionBlock, MLPBlock, RMSNorm


def from_bits(values):
    return torch.from_numpy(np.array(values, dtype=np.uint16, copy=True).view(np.int16)).view(torch.bfloat16)


def parameter(values):
    return torch.nn.Parameter(values, requires_grad=False)


def dense(checkpoint, name):
    raw = checkpoint.tensor(name)
    if raw.ndim < 2 or raw.nbytes < 64 << 20:
        return from_bits(raw)
    out = torch.empty(raw.shape, dtype=torch.bfloat16)
    for start in range(0, raw.shape[0], 4096):
        out[start:start+4096] = from_bits(raw[start:start+4096])
    return out


class IndexedExperts:
    """Implement the checkpoint tensor's indexing without resident decoded weights."""
    def __init__(self, checkpoint, prefix):
        self.c, self.prefix = checkpoint, prefix

    def __getitem__(self, index):
        if isinstance(index, tuple):
            indices, rest = index[0], index[1:]
            if rest != (Ellipsis,):
                raise ValueError("Only original expert indexing is supported")
        else:
            indices = index
        selected = indices.detach().cpu().numpy()
        blocks = self.c.tensor(self.prefix + "_blocks")
        scales = self.c.tensor(self.prefix + "_scales")
        _, rows, groups, _ = blocks.shape
        result = torch.empty((*selected.shape, rows, groups*32), dtype=torch.bfloat16)
        flat = result.reshape(-1, rows, groups*32)
        for target, expert in zip(flat, selected.reshape(-1)):
            for start in range(0, rows, 160):
                b = np.array(blocks[int(expert), start:start+160], copy=True)
                s = np.array(scales[int(expert), start:start+160], copy=True)
                decoded = decode_mxfp4(b, s).reshape(len(b), groups*32)
                target[start:start+160] = torch.from_numpy(decoded).to(torch.bfloat16)
        return result


def attention(c, config, layer):
    prefix = f"model.layers.{layer}."
    module = AttentionBlock(config, layer, device="meta")
    module.norm.scale = parameter(dense(c, prefix+"input_layernorm.weight").float())
    attn = prefix+"self_attn."
    module.sinks = parameter(dense(c, attn+"sinks"))
    module.qkv.weight = parameter(torch.cat([dense(c, attn+p+"_proj.weight") for p in ("q","k","v")], dim=0))
    module.qkv.bias = parameter(torch.cat([dense(c, attn+p+"_proj.bias") for p in ("q","k","v")], dim=0))
    module.out.weight = parameter(dense(c, attn+"o_proj.weight"))
    module.out.bias = parameter(dense(c, attn+"o_proj.bias"))
    module.rope.device = torch.device("cpu")
    return module


def mlp(c, config, layer):
    prefix = f"model.layers.{layer}."
    module = MLPBlock(config, device="meta")
    module.norm.scale = parameter(dense(c, prefix+"post_attention_layernorm.weight").float())
    module.gate.weight = parameter(dense(c, prefix+"mlp.router.weight"))
    module.gate.bias = parameter(dense(c, prefix+"mlp.router.bias"))
    for original, checkpoint in (("mlp1", "gate_up_proj"), ("mlp2", "down_proj")):
        del module._parameters[original+"_weight"]
        setattr(module, original+"_weight", IndexedExperts(c, prefix+"mlp.experts."+checkpoint))
        setattr(module, original+"_bias", parameter(dense(c, prefix+"mlp.experts."+checkpoint+"_bias")))
    return module


def save_tensor(path, tensor):
    raw = tensor.detach().contiguous().view(torch.int16).numpy().astype("<u2", copy=False)
    with path.open("xb") as out:
        np.save(out, raw)
    return dict(file=path.name, shape=list(tensor.shape), dtype="BF16",
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--text", default="Hello")
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--threads", type=int, default=4)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    complete = json.loads((args.model/"COMPLETE.json").read_text())
    if complete["revision"] != "6cee5e81ee83917806bbde320786a8fb61efebee" or complete["tensors"] != 459:
        raise ValueError("Complete pinned checkpoint is required")
    args.output.mkdir(parents=True, exist_ok=False)
    c = Checkpoint(args.model)
    config = ModelConfig(num_hidden_layers=24, num_experts=32)
    tokenizer = Tokenizer.from_file(str(args.model/"tokenizer.json"))
    tokens = tokenizer.encode(args.text).ids
    if not 1 <= len(tokens) <= 16 or not 1 <= args.steps <= 4:
        raise ValueError("This bounded reference run supports short prompts and up to four steps")
    records = []
    started = time.monotonic()
    norm = RMSNorm(2880, device="meta")
    norm.scale = parameter(dense(c, "model.norm.weight").float())
    for step in range(args.steps):
        x = from_bits(c.tensor("model.embed_tokens.weight")[tokens])
        for layer in range(24):
            tick = time.monotonic()
            attn = attention(c, config, layer)
            x = attn(x)
            attn_record = save_tensor(args.output/f"step{step:02}-layer{layer:02}-attention.npy", x)
            del attn
            expert = mlp(c, config, layer)
            routing = []
            def trace_gate(module, operands, logits):
                chosen = torch.topk(logits, 4, dim=-1, sorted=True)
                routing.append(dict(experts=chosen.indices.tolist(),
                                    logits=chosen.values.float().tolist(),
                                    probabilities=torch.softmax(chosen.values, dim=1).float().tolist()))
            hook = expert.gate.register_forward_hook(trace_gate)
            x = expert(x)
            hook.remove()
            del expert
            if not torch.isfinite(x).all():
                raise ValueError("Nonfinite reference hidden state")
            record = dict(step=step, layer=layer, input_tokens=tokens.copy(), routing=routing,
                          attention=attn_record, output=save_tensor(args.output/f"step{step:02}-layer{layer:02}-output.npy", x),
                          seconds=time.monotonic()-tick)
            records.append(record)
            with (args.output/"layers.jsonl").open("a") as out:
                out.write(json.dumps(record)+"\n")
                out.flush()
            print(json.dumps(dict(step=step, layer=layer, seconds=record["seconds"])), flush=True)
        hidden = norm(x[-1:])
        head = dense(c, "lm_head.weight")
        logits = torch.nn.functional.linear(hidden, head)
        del head
        save_tensor(args.output/f"step{step:02}-logits.npy", logits)
        token = int(logits.argmax(dim=-1)[0])
        tokens.append(token)
        report = dict(cpu_reference=True, physical_inference=False, model=complete["model"],
                      revision=complete["revision"], upstream_commit="7b583341fe16729127f6d5b94a7b09ccae97e1a1",
                      torch=torch.__version__, tokens=tokens, decoded=tokenizer.decode(tokens),
                      completed_steps=step+1, layers_per_step=24, elapsed_seconds=time.monotonic()-started,
                      note="Plain text prompt; complete-prefix recomputation, no incremental KV optimization")
        (args.output/"progress.json").write_text(json.dumps(report, indent=2)+"\n")
        print(json.dumps(report), flush=True)
    (args.output/"COMPLETE.json").write_text(json.dumps(report, indent=2)+"\n")


if __name__ == "__main__":
    main()
