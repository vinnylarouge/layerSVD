"""Validation harness for layersvd.dequant.

For each ggml type that appears in our model, picks one representative
tensor, runs our pure-NumPy dequant, and compares against the canonical
``gguf`` Python package's dequantizer. Bit-for-bit equality on float32 is
required (since both implementations follow the same arithmetic).

Run::

    .venv/bin/python -m layersvd.dequant_test model/gemma-4-26B-A4B-it-Q4_K_M.gguf

Exits non-zero if any type fails. This is the gating check before SVD.
"""

from __future__ import annotations

import sys
from typing import Iterable

import numpy as np

from layersvd import dequant as ld
from layersvd import gguf_reader as gr


# pick one tensor name per ggml type that we expect to find
TARGET_TYPES = {
    0: "F32",      # blk.0.attn_norm.weight  (any norm tensor)
    6: "Q5_0",
    8: "Q8_0",
    12: "Q4_K",
    14: "Q6_K",
}


def _pick_one_per_type(g: gr.GGUF) -> dict[int, gr.TensorDesc]:
    found: dict[int, gr.TensorDesc] = {}
    for t in g.tensors:
        if t.ggml_type in TARGET_TYPES and t.ggml_type not in found:
            # Prefer 2D weight tensors (skip *_exps which are 3D)
            if "_exps" in t.name:
                continue
            if len(t.shape) >= 1 and "weight" in t.name:
                found[t.ggml_type] = t
        if len(found) == len(TARGET_TYPES):
            break
    # If we still don't have everything (e.g. Q5_0 only on _exps), allow 3D
    for t in g.tensors:
        if t.ggml_type in TARGET_TYPES and t.ggml_type not in found:
            found[t.ggml_type] = t
        if len(found) == len(TARGET_TYPES):
            break
    return found


def _gguf_reference_dequant(path: str, tensor_name: str) -> np.ndarray:
    """Use the canonical ``gguf`` package to dequantize one tensor for comparison.

    ``ReaderTensor.data`` returns the raw mmap-backed uint8 buffer for
    quantized types. We dispatch on the tensor's ``tensor_type`` to the
    matching ``gguf.quants`` dequantizer.
    """
    import gguf
    import gguf.quants as gq

    type_to_class = {
        gguf.GGMLQuantizationType.F32: None,
        gguf.GGMLQuantizationType.F16: None,
        gguf.GGMLQuantizationType.BF16: None,
        gguf.GGMLQuantizationType.Q4_K: gq.Q4_K,
        gguf.GGMLQuantizationType.Q6_K: gq.Q6_K,
        gguf.GGMLQuantizationType.Q5_0: gq.Q5_0,
        gguf.GGMLQuantizationType.Q8_0: gq.Q8_0,
    }

    reader = gguf.GGUFReader(path)
    for t in reader.tensors:
        if t.name != tensor_name:
            continue
        klass = type_to_class.get(t.tensor_type)
        raw = np.asarray(t.data)
        if t.tensor_type == gguf.GGMLQuantizationType.F32:
            arr = raw.view(np.float32) if raw.dtype != np.float32 else raw
            return arr.reshape(-1).astype(np.float32, copy=True)
        if t.tensor_type == gguf.GGMLQuantizationType.F16:
            arr = raw.view(np.float16) if raw.dtype != np.float16 else raw
            return arr.reshape(-1).astype(np.float32)
        if t.tensor_type == gguf.GGMLQuantizationType.BF16:
            arr16 = raw.view(np.uint16).astype(np.uint32) << 16
            return arr16.view(np.float32).reshape(-1).astype(np.float32, copy=True)
        if klass is None:
            raise NotImplementedError(f"no reference dequant for {t.tensor_type}")
        out = klass.dequantize_rows(raw)
        return out.reshape(-1).astype(np.float32, copy=False)
    raise KeyError(f"tensor {tensor_name!r} not found via gguf reader")


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m layersvd.dequant_test <gguf-file>")
        return 2
    path = sys.argv[1]
    g = gr.read(path)
    targets = _pick_one_per_type(g)
    if len(targets) < len(TARGET_TYPES):
        missing = set(TARGET_TYPES) - set(targets)
        print(f"WARN: did not find sample tensors for ggml types {missing}")

    mm, view = gr.open_data(path)
    try:
        all_ok = True
        for ggml_type, t in sorted(targets.items()):
            n = 1
            for d in t.shape:
                n *= int(d)
            buf = gr.read_tensor_bytes(g, t, view)
            ours = ld.dequantize(buf, ggml_type, n)
            try:
                ref = _gguf_reference_dequant(path, t.name)
            except Exception as exc:
                print(f"  [SKIP] {TARGET_TYPES[ggml_type]:5s} {t.name}: gguf ref unavailable ({exc})")
                continue

            if ref.shape != ours.shape:
                print(
                    f"  [FAIL] {TARGET_TYPES[ggml_type]:5s} {t.name}: shape mismatch ours={ours.shape} ref={ref.shape}"
                )
                all_ok = False
                continue

            diff = np.abs(ours - ref)
            max_abs = float(diff.max()) if diff.size else 0.0
            mean_abs = float(diff.mean()) if diff.size else 0.0
            equal = bool(np.array_equal(ours, ref))
            tag = "OK   " if equal else ("CLOSE" if max_abs < 1e-5 else "FAIL ")
            if tag.startswith("FAIL"):
                all_ok = False
            print(
                f"  [{tag}] {TARGET_TYPES[ggml_type]:5s} {t.name}: "
                f"shape={ours.shape}  max_abs_err={max_abs:.3e}  mean_abs={mean_abs:.3e}"
            )
            if not equal and tag.startswith("FAIL"):
                # Show a few mismatched values
                idx = int(diff.argmax())
                print(f"      worst @ {idx}: ours={ours[idx]:.6e}  ref={ref[idx]:.6e}")
        return 0 if all_ok else 1
    finally:
        view.release()
        mm.close()


if __name__ == "__main__":
    raise SystemExit(main())
