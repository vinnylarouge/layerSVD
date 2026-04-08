"""Emit ``data/architecture.json`` from the GGUF metadata.

The architecture viewer in ``docs/index.html`` consumes this. The shape
is intentionally compact — every layer references the same submodule
template, so we serialize the template once and the per-layer fields
that vary (KV head count, SWA flag, head_dim, tensor types).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from layersvd import gguf_reader as gr


def _tensor_meta_for(g: gr.GGUF, name: str) -> dict | None:
    try:
        t = g.tensor(name)
    except KeyError:
        return None
    n = 1
    for d in t.shape:
        n *= int(d)
    return {
        "name": t.name,
        "shape": list(int(d) for d in t.shape),
        "type": t.type_name,
        "params": n,
    }


def build(gguf_path: str) -> dict:
    g = gr.read(gguf_path)
    md = g.metadata
    arch = md.get("general.architecture", "?")
    n_layer = int(md["gemma4.block_count"])

    sliding_pattern = list(md["gemma4.attention.sliding_window_pattern"])
    head_count = int(md["gemma4.attention.head_count"])
    head_count_kv_arr = list(md["gemma4.attention.head_count_kv"])
    n_embd = int(md["gemma4.embedding_length"])
    n_ff_dense = int(md["gemma4.feed_forward_length"])
    n_ff_exp = int(md["gemma4.expert_feed_forward_length"])
    n_expert = int(md["gemma4.expert_count"])
    n_expert_used = int(md["gemma4.expert_used_count"])
    key_length = int(md["gemma4.attention.key_length"])
    value_length = int(md["gemma4.attention.value_length"])
    key_length_swa = int(md["gemma4.attention.key_length_swa"])
    value_length_swa = int(md["gemma4.attention.value_length_swa"])
    rope_base = float(md["gemma4.rope.freq_base"])
    rope_base_swa = float(md["gemma4.rope.freq_base_swa"])
    sliding_window = int(md["gemma4.attention.sliding_window"])
    final_softcap = float(md["gemma4.final_logit_softcapping"])
    rms_eps = float(md["gemma4.attention.layer_norm_rms_epsilon"])

    # Tabulate per-layer
    layers = []
    for i in range(n_layer):
        # In gemma4-iswa.cpp, is_swa(i) is True for SWA, the global layers
        # have sliding_window_pattern[i] == False.
        is_swa = bool(sliding_pattern[i])
        head_count_kv = int(head_count_kv_arr[i])
        head_dim = key_length_swa if is_swa else key_length
        head_dim_v = value_length_swa if is_swa else value_length
        rope = rope_base_swa if is_swa else rope_base

        attn = {
            "q": _tensor_meta_for(g, f"blk.{i}.attn_q.weight"),
            "k": _tensor_meta_for(g, f"blk.{i}.attn_k.weight"),
            "v": _tensor_meta_for(g, f"blk.{i}.attn_v.weight"),
            "o": _tensor_meta_for(g, f"blk.{i}.attn_output.weight"),
            "q_norm": _tensor_meta_for(g, f"blk.{i}.attn_q_norm.weight"),
            "k_norm": _tensor_meta_for(g, f"blk.{i}.attn_k_norm.weight"),
            "norm": _tensor_meta_for(g, f"blk.{i}.attn_norm.weight"),
            "post_norm": _tensor_meta_for(g, f"blk.{i}.post_attention_norm.weight"),
        }
        ffn_dense = {
            "gate": _tensor_meta_for(g, f"blk.{i}.ffn_gate.weight"),
            "up": _tensor_meta_for(g, f"blk.{i}.ffn_up.weight"),
            "down": _tensor_meta_for(g, f"blk.{i}.ffn_down.weight"),
            "norm": _tensor_meta_for(g, f"blk.{i}.ffn_norm.weight"),
            "post_norm_1": _tensor_meta_for(g, f"blk.{i}.post_ffw_norm_1.weight"),
        }
        ffn_moe = {
            "router": _tensor_meta_for(g, f"blk.{i}.ffn_gate_inp.weight"),
            "router_scale": _tensor_meta_for(g, f"blk.{i}.ffn_gate_inp.scale"),
            "gate_up_exps": _tensor_meta_for(g, f"blk.{i}.ffn_gate_up_exps.weight"),
            "down_exps": _tensor_meta_for(g, f"blk.{i}.ffn_down_exps.weight"),
            "down_exps_scale": _tensor_meta_for(g, f"blk.{i}.ffn_down_exps.scale"),
            "pre_norm": _tensor_meta_for(g, f"blk.{i}.pre_ffw_norm_2.weight"),
            "post_norm": _tensor_meta_for(g, f"blk.{i}.post_ffw_norm_2.weight"),
        }
        is_moe = ffn_moe["router"] is not None

        block_post = _tensor_meta_for(g, f"blk.{i}.post_ffw_norm.weight")
        out_scale = _tensor_meta_for(g, f"blk.{i}.layer_output_scale.weight")

        layers.append({
            "index": i,
            "is_swa": is_swa,
            "is_moe": is_moe,
            "n_head": head_count,
            "n_head_kv": head_count_kv,
            "head_dim": head_dim,
            "head_dim_v": head_dim_v,
            "rope_base": rope,
            "attn": attn,
            "ffn_dense": ffn_dense,
            "ffn_moe": ffn_moe,
            "post_norm": block_post,
            "out_scale": out_scale,
        })

    # Global tensors
    globals_ = {
        "token_embd": _tensor_meta_for(g, "token_embd.weight"),
        "output_norm": _tensor_meta_for(g, "output_norm.weight"),
        "rope_freqs": _tensor_meta_for(g, "rope_freqs.weight"),
    }

    total_params = 0
    for t in g.tensors:
        n = 1
        for d in t.shape:
            n *= int(d)
        total_params += n

    out = {
        "version": 1,
        "model": {
            "architecture": arch,
            "name": md.get("general.name", "?"),
            "size_label": md.get("general.size_label", "?"),
            "n_layer": n_layer,
            "n_embd": n_embd,
            "vocab_size": int(g.tensor("token_embd.weight").shape[1]),
            "n_head": head_count,
            "n_ff_dense": n_ff_dense,
            "n_ff_exp": n_ff_exp,
            "n_expert": n_expert,
            "n_expert_used": n_expert_used,
            "key_length": key_length,
            "value_length": value_length,
            "key_length_swa": key_length_swa,
            "value_length_swa": value_length_swa,
            "rope_base": rope_base,
            "rope_base_swa": rope_base_swa,
            "sliding_window": sliding_window,
            "final_logit_softcap": final_softcap,
            "rms_norm_eps": rms_eps,
            "context_length": int(md["gemma4.context_length"]),
            "total_params": total_params,
            "n_tensors": len(g.tensors),
        },
        "globals": globals_,
        "layers": layers,
    }
    return out


def main() -> None:
    if len(sys.argv) not in (2, 3):
        print("usage: python -m layersvd.build_arch <gguf> [<out.json>]")
        raise SystemExit(2)
    gguf_path = sys.argv[1]
    out_path = Path(sys.argv[2] if len(sys.argv) == 3 else "data/architecture.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arch = build(gguf_path)
    with open(out_path, "w") as f:
        json.dump(arch, f, indent=1)
    print(f"wrote {out_path} ({out_path.stat().st_size} bytes)")
    print(f"  total_params: {arch['model']['total_params']:,}")
    print(f"  n_layer: {arch['model']['n_layer']}, n_embd: {arch['model']['n_embd']}, vocab: {arch['model']['vocab_size']:,}")


if __name__ == "__main__":
    main()
