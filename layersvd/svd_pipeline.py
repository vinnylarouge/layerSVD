"""Truncated SVD over every linear weight in the GGUF.

For each 2D weight tensor (and per-expert slices of the 3D MoE tensors)
we compute a top-K SVD ``W ≈ U @ diag(s) @ Vt``. Results are streamed
to a compact on-disk format under ``data/svd/`` so the static site can
fetch ranges via HTTP byte requests.

Per-tensor we also compute a few "composed" SVDs that give more
interpretable concept bases:

* attention value→output: ``W_o @ W_v`` (square dim = d_model).
* dense FFN linearization: ``W_down @ W_up`` (square dim = d_model).
* dense FFN gate × up linearization: ``W_down @ diag(softplus(g_norm)) @ W_up``
  is too input-dependent; we use the simpler ``W_down @ W_up`` instead.

K budget (defaults — overridable via env vars):
    K_ATTN = 64
    K_FFN  = 64
    K_EMBD = 32
    K_EXPERT = 8

Output layout under ``data/svd/``:
    manifest.json
        {tensor_id: {shape, K, U_offset, U_bytes, V_offset, V_bytes,
                     S_offset, S_bytes, group, layer, role}}
    spectra.bin     concatenated singular values  (fp32)
    U.f16           concatenated left singular vector matrices (fp16)
    V.f16           concatenated right singular vector matrices (fp16)

We compute SVDs of W oriented as ``(out_dim, in_dim)``. ``U`` then
spans the *output* directions (left singular vectors) and ``V`` spans
the *input* directions (right singular vectors). For interpretability:

* the right singular vectors V[i, :] are "what this matrix listens for";
* the left singular vectors U[:, i] are "what it produces".
"""

from __future__ import annotations

import json
import os
import struct
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
from tqdm import tqdm

from layersvd import dequant as ld
from layersvd import gguf_reader as gr


# K budgets ---------------------------------------------------------------
K_ATTN = int(os.environ.get("LAYERSVD_K_ATTN", "64"))
K_FFN = int(os.environ.get("LAYERSVD_K_FFN", "64"))
K_EMBD = int(os.environ.get("LAYERSVD_K_EMBD", "32"))
K_EXPERT = int(os.environ.get("LAYERSVD_K_EXPERT", "8"))


@dataclass
class SVDEntry:
    tensor_id: str
    shape: tuple[int, int]
    K: int
    group: str  # 'attn' | 'attn_composed' | 'ffn_dense' | 'ffn_dense_composed' |
                # 'router' | 'expert_gate_up' | 'expert_down' | 'embd'
    layer: int  # -1 for non-block tensors (token_embd)
    role: str   # eg 'q', 'k', 'v', 'o', 'wo_wv', 'gate', 'up', 'down', 'down_up', 'embd'
    expert: int = -1  # expert index, -1 if not an expert tensor
    s_offset: int = 0
    s_bytes: int = 0
    u_offset: int = 0
    u_bytes: int = 0
    v_offset: int = 0
    v_bytes: int = 0
    s_max: float = 0.0
    s_min: float = 0.0
    fro_kept: float = 0.0
    fro_total: float = 0.0


def _truncated_svd(W: np.ndarray, K: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-K SVD of a 2D matrix. ``W`` is shape ``(m, n)`` float32.

    Returns ``(U, s, Vt)`` with shapes ``(m, K)``, ``(K,)``, ``(K, n)``.
    Uses dense LAPACK SVD with truncation; faster and more reliable than
    ARPACK for the matrix shapes we have (max dim ~262K).
    """
    m, n = W.shape
    K = min(K, m, n)
    # For very tall/wide matrices (e.g. 2816 x 262144 token_embd) we use the
    # economy SVD which scales with min(m, n).
    U_full, s_full, Vt_full = np.linalg.svd(W, full_matrices=False, compute_uv=True)
    return U_full[:, :K].copy(), s_full[:K].copy(), Vt_full[:K, :].copy()


def _dequant_2d(g: gr.GGUF, view: memoryview, t: gr.TensorDesc) -> np.ndarray:
    """Dequantize a tensor and reshape to 2D (out_dim, in_dim).

    GGUF stores tensors with shape conventions where the first dim is the
    *fastest-varying* (innermost) axis. For 2D weight tensors that means
    shape ``(in_dim, out_dim)`` in our notation; we transpose to
    ``(out_dim, in_dim)`` so left singular vectors are output directions.
    """
    n = 1
    for d in t.shape:
        n *= int(d)
    buf = gr.read_tensor_bytes(g, t, view)
    flat = ld.dequantize(buf, t.ggml_type, n)
    if len(t.shape) == 2:
        in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
        return flat.reshape(out_dim, in_dim)
    raise ValueError(f"expected 2D tensor, got shape {t.shape}")


def _dequant_3d_expert(
    g: gr.GGUF, view: memoryview, t: gr.TensorDesc, expert_idx: int
) -> np.ndarray:
    """Slice one expert out of a 3D ``ffn_*_exps`` tensor and return ``(out, in)``.

    GGUF expert tensors have shape ``(in, out, n_experts)`` (innermost first).
    We slice on the last dim and transpose.
    """
    in_dim, out_dim, n_experts = (int(d) for d in t.shape)
    assert 0 <= expert_idx < n_experts
    n_per_expert = in_dim * out_dim

    # Element-level offset for this expert
    elem_off = expert_idx * n_per_expert
    # We can't easily slice quantized blocks at non-block boundaries; for
    # K-quants this means dequantizing the whole tensor and then slicing.
    # That's fine because we'll loop experts in the outer pipeline and
    # cache the dequantized whole.
    raise NotImplementedError("use _dequant_3d_full")


def _dequant_3d_full(g: gr.GGUF, view: memoryview, t: gr.TensorDesc) -> np.ndarray:
    """Dequantize an entire 3D expert tensor; returns ``(n_experts, out_dim, in_dim)``."""
    in_dim, out_dim, n_experts = (int(d) for d in t.shape)
    n_total = in_dim * out_dim * n_experts
    buf = gr.read_tensor_bytes(g, t, view)
    flat = ld.dequantize(buf, t.ggml_type, n_total)
    # Layout: innermost-first means flat[ ((expert*out + j)*in) + i ] = W[expert, j, i]
    arr = flat.reshape(n_experts, out_dim, in_dim)
    return arr


# ---------------------------------------------------------------------------
# Streaming output helpers
# ---------------------------------------------------------------------------

class _StreamWriter:
    """Append fp16/fp32 chunks to a file and track offsets."""

    def __init__(self, path: Path, dtype: np.dtype) -> None:
        self.path = path
        self.f = open(path, "wb")
        self.offset = 0
        self.dtype = dtype

    def write(self, arr: np.ndarray) -> tuple[int, int]:
        a = np.ascontiguousarray(arr, dtype=self.dtype)
        b = a.tobytes()
        off = self.offset
        self.f.write(b)
        self.offset += len(b)
        return off, len(b)

    def close(self) -> None:
        self.f.close()


# ---------------------------------------------------------------------------
# Tensor classification
# ---------------------------------------------------------------------------

def _classify_block_tensor(name: str) -> tuple[str | None, str | None, int | None]:
    """Return (group, role, layer_idx) for the tensors we care about."""
    if not name.startswith("blk."):
        return None, None, None
    rest = name[4:]
    layer_str, _, suffix = rest.partition(".")
    try:
        layer = int(layer_str)
    except ValueError:
        return None, None, None

    table = {
        "attn_q.weight": ("attn", "q"),
        "attn_k.weight": ("attn", "k"),
        "attn_v.weight": ("attn", "v"),
        "attn_output.weight": ("attn", "o"),
        "ffn_gate.weight": ("ffn_dense", "gate"),
        "ffn_up.weight": ("ffn_dense", "up"),
        "ffn_down.weight": ("ffn_dense", "down"),
        "ffn_gate_inp.weight": ("router", "router"),
        "ffn_gate_up_exps.weight": ("expert_gate_up", "expert_gate_up"),
        "ffn_down_exps.weight": ("expert_down", "expert_down"),
    }
    if suffix in table:
        g, r = table[suffix]
        return g, r, layer
    return None, None, None


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run(
    gguf_path: str,
    out_dir: str = "data/svd",
    skip_experts: bool = False,
) -> list[SVDEntry]:
    """Compute and serialize all SVDs."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    g = gr.read(gguf_path)
    mm, view = gr.open_data(gguf_path)

    s_writer = _StreamWriter(out / "spectra.bin", np.float32)
    u_writer = _StreamWriter(out / "U.f16", np.float16)
    v_writer = _StreamWriter(out / "V.f16", np.float16)

    entries: list[SVDEntry] = []

    def _record(W: np.ndarray, K: int, *, tensor_id: str, group: str,
                layer: int, role: str, expert: int = -1) -> SVDEntry:
        K_eff = min(K, *W.shape)
        U, s, Vt = _truncated_svd(W, K_eff)
        s_off, s_b = s_writer.write(s)
        u_off, u_b = u_writer.write(U)
        v_off, v_b = v_writer.write(Vt)
        e = SVDEntry(
            tensor_id=tensor_id,
            shape=(int(W.shape[0]), int(W.shape[1])),
            K=K_eff,
            group=group,
            layer=layer,
            role=role,
            expert=expert,
            s_offset=s_off, s_bytes=s_b,
            u_offset=u_off, u_bytes=u_b,
            v_offset=v_off, v_bytes=v_b,
            s_max=float(s.max()) if s.size else 0.0,
            s_min=float(s.min()) if s.size else 0.0,
            fro_kept=float(np.sqrt((s ** 2).sum())),
            fro_total=float(np.linalg.norm(W)),
        )
        entries.append(e)
        return e

    try:
        # ---- Token embedding (single non-block tensor) ----
        tok = g.tensor("token_embd.weight")
        print(f"[svd] dequantizing token_embd {tok.shape} ...")
        W_emb = _dequant_2d(g, view, tok)
        _record(W_emb, K_EMBD, tensor_id="token_embd", group="embd", layer=-1, role="embd")
        del W_emb

        # ---- Per-block: attn + dense FFN + router ----
        block_count = int(g.metadata["gemma4.block_count"])
        # Group tensors by layer, classified
        per_layer: dict[int, dict[str, gr.TensorDesc]] = {i: {} for i in range(block_count)}
        for t in g.tensors:
            grp, role, layer = _classify_block_tensor(t.name)
            if grp is None:
                continue
            per_layer[layer][role] = t

        for layer in tqdm(range(block_count), desc="layers"):
            tens = per_layer[layer]

            # Attention Q/K/V/O — record raw SVD per matrix.
            for role in ("q", "k", "v", "o"):
                if role not in tens:
                    continue
                W = _dequant_2d(g, view, tens[role])
                _record(W, K_ATTN,
                        tensor_id=f"blk.{layer}.attn_{role}",
                        group="attn", layer=layer, role=role)

            # Attention composed value→output (skip global layers without V tensor)
            if "o" in tens and "v" in tens:
                W_o = _dequant_2d(g, view, tens["o"])  # (d_model, n_head_q*head_dim_v)
                W_v = _dequant_2d(g, view, tens["v"])  # (n_head_kv*head_dim_v, d_model)
                # GQA: sum each Q-head's projection through its shared KV head
                d_model = W_o.shape[0]
                n_q_total = W_o.shape[1]
                n_kv_total = W_v.shape[0]
                if n_q_total % n_kv_total == 0:
                    head_dim_v = n_kv_total // int(g.metadata["gemma4.attention.head_count_kv"][layer])
                    n_head_kv = n_kv_total // head_dim_v
                    n_head_q = n_q_total // head_dim_v
                    q_per_kv = n_head_q // n_head_kv
                    W_compose = np.zeros((d_model, d_model), dtype=np.float32)
                    for ih in range(n_head_q):
                        kv_h = ih // q_per_kv
                        Wo_h = W_o[:, ih * head_dim_v : (ih + 1) * head_dim_v]   # (d_model, hd)
                        Wv_h = W_v[kv_h * head_dim_v : (kv_h + 1) * head_dim_v, :]  # (hd, d_model)
                        W_compose += Wo_h @ Wv_h
                    _record(W_compose, K_ATTN,
                            tensor_id=f"blk.{layer}.attn_wo_wv",
                            group="attn_composed", layer=layer, role="wo_wv")
                    del W_compose
                del W_o, W_v

            # Dense FFN gate/up/down
            for role in ("gate", "up", "down"):
                if role in tens:
                    W = _dequant_2d(g, view, tens[role])
                    _record(W, K_FFN,
                            tensor_id=f"blk.{layer}.ffn_{role}",
                            group="ffn_dense", layer=layer, role=role)

            # Composed dense FFN: W_down @ W_up
            if "up" in tens and "down" in tens:
                W_up = _dequant_2d(g, view, tens["up"])     # (d_ffn, d_model)
                W_down = _dequant_2d(g, view, tens["down"]) # (d_model, d_ffn)
                W_compose = W_down @ W_up                   # (d_model, d_model)
                _record(W_compose, K_FFN,
                        tensor_id=f"blk.{layer}.ffn_down_up",
                        group="ffn_dense_composed", layer=layer, role="down_up")
                del W_up, W_down, W_compose

            # MoE router
            if "router" in tens:
                W_r = _dequant_2d(g, view, tens["router"])
                _record(W_r, min(K_FFN, W_r.shape[0], W_r.shape[1]),
                        tensor_id=f"blk.{layer}.ffn_gate_inp",
                        group="router", layer=layer, role="router")

            # Per-expert SVDs
            if not skip_experts:
                for role_name, tensor_role in (
                    ("expert_gate_up", "expert_gate_up"),
                    ("expert_down", "expert_down"),
                ):
                    if tensor_role not in tens:
                        continue
                    arr = _dequant_3d_full(g, view, tens[tensor_role])
                    n_experts = arr.shape[0]
                    for ei in range(n_experts):
                        W = arr[ei]
                        _record(W, K_EXPERT,
                                tensor_id=f"blk.{layer}.{tensor_role}.e{ei}",
                                group=role_name, layer=layer, role=tensor_role, expert=ei)
                    del arr

    finally:
        s_writer.close()
        u_writer.close()
        v_writer.close()
        view.release()
        mm.close()

    # Write manifest
    manifest = {
        "version": 1,
        "k_attn": K_ATTN,
        "k_ffn": K_FFN,
        "k_embd": K_EMBD,
        "k_expert": K_EXPERT,
        "totals": {
            "spectra_bytes": s_writer.offset,
            "u_bytes": u_writer.offset,
            "v_bytes": v_writer.offset,
            "n_entries": len(entries),
        },
        "entries": [asdict(e) for e in entries],
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return entries


def main() -> None:
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("gguf", help="Path to the GGUF file")
    p.add_argument("--out", default="data/svd", help="Output directory")
    p.add_argument("--skip-experts", action="store_true",
                   help="Skip per-expert SVDs (much faster, smaller payload)")
    args = p.parse_args()
    entries = run(args.gguf, args.out, skip_experts=args.skip_experts)
    print(f"\n{len(entries)} SVD entries written to {args.out}/")
    sizes = sum(e.s_bytes + e.u_bytes + e.v_bytes for e in entries)
    print(f"total bytes (S+U+V): {sizes:,}  ({sizes / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
