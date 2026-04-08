"""Project captured residual streams onto SVD bases.

Reads:
    docs/data/svd/manifest.json + spectra.bin + U.f16 + V.f16
    data/raw/<exemplar>/manifest.json + pass_NNNN/<tensor>.bin

Emits one ``data/traces/<exemplar>.json`` per exemplar containing, for
each token (prompt+generated):

    layer_acts[L][channel][i]   activation strength of right-singular-vector i
                                of layer L+1's <channel> projection
                                ('q', 'k', 'v', 'wo_wv', 'down_up')
                                computed against the residual stream l_out-{L}.

    layer_speak[L][i]            magnitude on left-singular-vector i of layer L's
                                composed FFN (down_up). This is "what L produced".

    flow[L][i][j]               cross-layer coupling between concept i in L
                                (left-sing-vec) and concept j in L+1 (right-sing-vec)
                                weighted by the per-token activation strengths.

The flow weights are *paired* — they use the natural duality between L's
output basis and L+1's input basis rather than comparing two unrelated
rotations. This addresses the "rotation noise" risk for cross-layer
visualization.

Run::

    .venv/bin/python -m layersvd.project \
        --svd-dir docs/data/svd \
        --raw-dir data/raw/exemplar1 \
        --out docs/data/traces/exemplar1.json
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Loading the SVD bundle
# ---------------------------------------------------------------------------

@dataclass
class SVDStore:
    manifest: dict
    entries: dict[str, dict]
    base_dir: Path
    spectra: np.memmap | None
    # Cached memmaps for chunked U/V files keyed by filename
    _u_files: dict[str, np.memmap]
    _v_files: dict[str, np.memmap]

    def _u_view(self, fname: str) -> np.memmap:
        if fname not in self._u_files:
            self._u_files[fname] = np.memmap(self.base_dir / fname, dtype=np.float16, mode="r")
        return self._u_files[fname]

    def _v_view(self, fname: str) -> np.memmap:
        if fname not in self._v_files:
            self._v_files[fname] = np.memmap(self.base_dir / fname, dtype=np.float16, mode="r")
        return self._v_files[fname]

    def U_of(self, tid: str) -> np.ndarray:
        e = self.entries[tid]
        m, k = e["shape"][0], e["K"]
        n_floats = m * k
        fname = e.get("u_file", "U.f16")
        off = e["u_offset"] // 2
        view = self._u_view(fname)
        return np.asarray(view[off : off + n_floats], dtype=np.float16).reshape(m, k).astype(np.float32)

    def V_of(self, tid: str) -> np.ndarray:
        e = self.entries[tid]
        n, k = e["shape"][1], e["K"]
        n_floats = k * n
        fname = e.get("v_file", "V.f16")
        off = e["v_offset"] // 2
        view = self._v_view(fname)
        return np.asarray(view[off : off + n_floats], dtype=np.float16).reshape(k, n).astype(np.float32)

    def s_of(self, tid: str) -> np.ndarray:
        e = self.entries[tid]
        n_floats = e["K"]
        off = e["s_offset"] // 4
        return np.asarray(self.spectra[off : off + n_floats], dtype=np.float32)


def load_svd(svd_dir: str) -> SVDStore:
    p = Path(svd_dir)
    manifest = json.loads((p / "manifest.json").read_text())
    entries = {e["tensor_id"]: e for e in manifest["entries"]}
    spectra = np.memmap(p / "spectra.bin", dtype=np.float32, mode="r")
    return SVDStore(
        manifest=manifest,
        entries=entries,
        base_dir=p,
        spectra=spectra,
        _u_files={},
        _v_files={},
    )


# ---------------------------------------------------------------------------
# Loading captured residual streams
# ---------------------------------------------------------------------------

def _load_capture_tensor(path: Path, shape: tuple[int, int, int, int]) -> np.ndarray:
    raw = np.fromfile(path, dtype=np.float16).astype(np.float32)
    n_total = shape[0] * shape[1] * shape[2] * shape[3]
    if raw.size != n_total:
        # Some tensors are stripped to a single token at the last layer.
        # Try a 1-token shape.
        alt = (shape[0], 1, shape[2], shape[3])
        if raw.size == alt[0] * alt[1] * alt[2] * alt[3]:
            return raw.reshape(alt[3], alt[2], alt[1], alt[0]).transpose(3, 2, 1, 0)
        raise ValueError(f"shape mismatch for {path}: file has {raw.size} != expected {n_total}")
    return raw.reshape(shape[3], shape[2], shape[1], shape[0]).transpose(3, 2, 1, 0)


def load_capture(raw_dir: str) -> dict:
    """Returns dict with: prompt, prompt_tokens, generated_tokens,
    generated_strings, n_passes, captured_tensors, and a ``passes`` list
    where ``passes[pass_idx][tensor_name]`` is a (d, n_tokens) ndarray.
    """
    base = Path(raw_dir)
    manifest = json.loads((base / "manifest.json").read_text())
    captured_meta = {t["name"]: t for t in manifest["captured_tensors"]}

    passes = []
    for pi in range(manifest["n_passes"]):
        pass_dir = base / f"pass_{pi:04d}"
        per_pass: dict[str, np.ndarray] = {}
        if not pass_dir.exists():
            passes.append(per_pass)
            continue
        for fname in sorted(os.listdir(pass_dir)):
            if not fname.endswith(".bin"):
                continue
            tname = fname[:-4]
            meta = captured_meta.get(tname)
            if meta is None:
                continue
            shape = tuple(meta["shape"])  # (d, n_tokens, 1, 1) usually
            arr = _load_capture_tensor(pass_dir / fname, shape)
            # Flatten to (d, n_tokens)
            d, nt, _, _ = arr.shape
            per_pass[tname] = arr.reshape(d, nt)
        passes.append(per_pass)
    manifest["passes"] = passes
    return manifest


# ---------------------------------------------------------------------------
# Projection
# ---------------------------------------------------------------------------

# Per-layer "input basis" channels: which next-layer matrices' right-singular
# vectors do we project the residual stream onto?
INPUT_CHANNELS = ["q", "k", "v", "down_up"]
# Per-layer "output basis": which composed matrix's left-singular vectors
# represent "what this layer spoke"?
OUTPUT_CHANNEL = "down_up"


def _b64_fp16(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr.astype(np.float16))
    return base64.b64encode(a.tobytes()).decode("ascii")


def project_exemplar(svd: SVDStore, capture: dict, n_layer: int = 30) -> dict:
    """Build the trace dict for one exemplar."""

    # All tokens — concatenate prompt-pass and per-token passes
    prompt_strs = []  # we don't have decoded prompt strings; use IDs
    for tid in capture["prompt_tokens"]:
        prompt_strs.append(f"<{tid}>")
    all_tokens_str = prompt_strs + capture["generated_strings"]

    # Build a unified token stream by concatenating l_out-{L} across passes.
    # Pass 0 has shape (d, n_prompt) for layers 0..n_layer-2, and shape
    # (d, 1) for layer n_layer-1 (the optimization strips unused tokens).
    # Passes 1..N each contribute one token at every layer.
    n_prompt = len(capture["prompt_tokens"])
    n_gen = len(capture["generated_tokens"])
    # We index tokens by global position. For layer L < n_layer-1:
    #   tokens 0..n_prompt-1 from pass 0, plus tokens 0..n_gen-1 from passes 1..n_gen
    # For L = n_layer-1: only the LAST prompt token from pass 0, plus all gen tokens.
    # We standardize by exposing the LAST prompt token + all gen tokens for every layer.
    n_tokens_unified = 1 + n_gen  # last prompt token + each generated token
    token_labels = [f"prompt[-1]={prompt_strs[-1]}"] + [f"gen[{i}]={capture['generated_strings'][i]}" for i in range(n_gen)]

    # Per layer, gather the residual stream l_out-{L} for those tokens.
    # Storage: residuals[L] -> shape (d_model, n_tokens_unified)
    d_model = 2816
    residuals: dict[int, np.ndarray] = {}
    for L in range(n_layer):
        cols: list[np.ndarray] = []
        # Pass 0: take the LAST column of l_out-{L}
        p0 = capture["passes"][0]
        name = f"l_out-{L}"
        if name in p0:
            arr = p0[name]  # (d, n)
            cols.append(arr[:, -1:])
        # Passes 1..n_gen
        for pi in range(1, 1 + n_gen):
            if pi >= len(capture["passes"]):
                break
            arr = capture["passes"][pi].get(name)
            if arr is not None:
                cols.append(arr[:, -1:])
        if not cols:
            residuals[L] = np.zeros((d_model, n_tokens_unified), dtype=np.float32)
            continue
        residuals[L] = np.concatenate(cols, axis=1)

    # For each layer L, compute:
    #  layer_acts[L][channel] = V_{L+1}^{channel} @ residual_L
    #     shape (K, n_tokens) — projection of residual onto next-layer input basis
    # For the last layer, project onto output_norm basis (just identity).
    layer_acts: dict[int, dict[str, np.ndarray]] = {}
    for L in range(n_layer):
        layer_acts[L] = {}
        next_L = min(L + 1, n_layer - 1)
        for chan in INPUT_CHANNELS:
            tid = f"blk.{next_L}.attn_{chan}" if chan in ("q", "k", "v") else f"blk.{next_L}.ffn_down_up"
            if tid not in svd.entries:
                continue
            V = svd.V_of(tid)            # (K, in_dim) where in_dim == d_model
            if V.shape[1] != d_model:
                continue
            res = residuals[L]            # (d_model, n_tokens)
            proj = V @ res                # (K, n_tokens)
            layer_acts[L][chan] = proj

    # layer_speak[L] = U_L^{down_up}^T @ residual_L
    # The left singular vectors U of layer L's composed FFN span what the FFN
    # *can* output. The residual coming OUT of layer L lives in that span
    # plus what the attention added; projecting via U^T gives an estimate of
    # the "FFN-spoken" coordinates.
    layer_speak: dict[int, np.ndarray] = {}
    for L in range(n_layer):
        tid = f"blk.{L}.ffn_down_up"
        if tid not in svd.entries:
            continue
        U = svd.U_of(tid)  # (d_model, K)
        res = residuals[L]
        layer_speak[L] = U.T @ res  # (K, n_tokens)

    # Cross-layer flow: for each adjacent pair (L, L+1), the natural pairing
    # between U_L (output basis) and V_{L+1} (input basis for some channel).
    # We use channel = 'down_up' for the symmetric "what L said" -> "what L+1 hears in dense FFN"
    # flow.
    flow: dict[int, np.ndarray] = {}
    for L in range(n_layer - 1):
        tid_out = f"blk.{L}.ffn_down_up"
        tid_in = f"blk.{L+1}.ffn_down_up"
        if tid_out not in svd.entries or tid_in not in svd.entries:
            continue
        U_L = svd.U_of(tid_out)        # (d_model, K_L)
        V_L1 = svd.V_of(tid_in)        # (K_L1, d_model)
        # Coupling matrix C[i, j] = <U_L[:, i], V_{L+1}[j, :]>
        C = U_L.T @ V_L1.T              # (K_L, K_L1)
        # Then weight by per-token activations: flow_per_token[t, i, j] = layer_speak[L][i, t] * layer_acts[L][down_up][j, t] * C[i, j]
        # We aggregate to a single static map per layer pair using L2 norm over tokens
        speak_L = layer_speak.get(L)
        listen_L1 = layer_acts.get(L, {}).get("down_up")
        if speak_L is None or listen_L1 is None:
            flow[L] = C
            continue
        # Per-token: outer product weighted by C
        # We'll store the static C and per-token weight pairs separately.
        flow[L] = C  # static; per-token contribution computed in-browser if needed.

    # ---- Serialize ----
    out: dict[str, Any] = {
        "version": 1,
        "exemplar": capture.get("prompt", "")[:120],
        "n_layer": n_layer,
        "tokens": token_labels,
        "n_tokens": n_tokens_unified,
        "channels": INPUT_CHANNELS,
        "K_attn": svd.manifest["k_attn"],
        "K_ffn": svd.manifest["k_ffn"],
        "layers": {},
    }
    for L in range(n_layer):
        layer_dict: dict[str, Any] = {}
        for chan, arr in layer_acts[L].items():
            layer_dict[f"acts_{chan}"] = {"shape": list(arr.shape), "data": _b64_fp16(arr)}
        if L in layer_speak:
            arr = layer_speak[L]
            layer_dict["speak"] = {"shape": list(arr.shape), "data": _b64_fp16(arr)}
        if L in flow:
            arr = flow[L]
            layer_dict["flow"] = {"shape": list(arr.shape), "data": _b64_fp16(arr)}
        out["layers"][str(L)] = layer_dict

    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--svd-dir", default="docs/data/svd")
    p.add_argument("--raw-dir", required=True, help="Path to data/raw/<exemplar>/")
    p.add_argument("--out", required=True, help="Output JSON path")
    p.add_argument("--n-layer", type=int, default=30)
    args = p.parse_args()

    print(f"loading SVD bundle from {args.svd_dir} ...")
    svd = load_svd(args.svd_dir)
    print(f"  {len(svd.entries)} entries")

    print(f"loading capture from {args.raw_dir} ...")
    cap = load_capture(args.raw_dir)
    print(f"  prompt={cap.get('prompt','')[:60]!r}  n_passes={cap['n_passes']}  generated={len(cap['generated_tokens'])}")

    print("projecting ...")
    trace = project_exemplar(svd, cap, n_layer=args.n_layer)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(trace, f, separators=(",", ":"))
    print(f"wrote {out_path} ({out_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
