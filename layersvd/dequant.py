"""Pure-NumPy dequantizers for the GGML formats present in our Gemma 4 GGUF.

Reference: ``llama.cpp/ggml/src/ggml-quants.c``. Each function takes the raw
packed bytes for a tensor and returns a flat ``np.ndarray[float32]`` with
``n_elements`` entries. Vectorized over the block dimension.

Supported types: F32, F16, BF16, Q4_K, Q6_K, Q5_0, Q8_0.

The block sizes match ``llama.cpp/ggml/src/ggml-common.h``::

    Q4_K: super-block of 256 elements, 144 bytes
    Q6_K: super-block of 256 elements, 210 bytes
    Q5_0: block of 32 elements, 22 bytes
    Q8_0: block of 32 elements, 34 bytes
"""

from __future__ import annotations

import numpy as np

QK_K = 256
K_SCALE_SIZE = 12  # bytes used to pack 8x6-bit scales + 8x6-bit mins

Q4K_BLOCK_BYTES = 144
Q6K_BLOCK_BYTES = 210
Q5_0_BLOCK_BYTES = 22
Q8_0_BLOCK_BYTES = 34


# ---------------------------------------------------------------------------
# Trivial passthroughs
# ---------------------------------------------------------------------------

def dequant_f32(buf: bytes, n_elements: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype="<f4", count=n_elements)
    return arr.astype(np.float32, copy=True)


def dequant_f16(buf: bytes, n_elements: int) -> np.ndarray:
    arr = np.frombuffer(buf, dtype="<f2", count=n_elements)
    return arr.astype(np.float32)


def dequant_bf16(buf: bytes, n_elements: int) -> np.ndarray:
    raw = np.frombuffer(buf, dtype="<u2", count=n_elements).astype(np.uint32)
    raw <<= 16
    return raw.view(np.float32).astype(np.float32, copy=True)


# ---------------------------------------------------------------------------
# Q8_0
# ---------------------------------------------------------------------------

def dequant_q8_0(buf: bytes, n_elements: int) -> np.ndarray:
    assert n_elements % 32 == 0
    nb = n_elements // 32
    raw = np.frombuffer(buf, dtype=np.uint8, count=nb * Q8_0_BLOCK_BYTES).reshape(nb, Q8_0_BLOCK_BYTES)
    d = raw[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)  # (nb,)
    qs = raw[:, 2:].view(np.int8).astype(np.float32)  # (nb, 32)
    return (qs * d[:, None]).reshape(-1)


# ---------------------------------------------------------------------------
# Q5_0
# ---------------------------------------------------------------------------

def dequant_q5_0(buf: bytes, n_elements: int) -> np.ndarray:
    assert n_elements % 32 == 0
    nb = n_elements // 32
    raw = np.frombuffer(buf, dtype=np.uint8, count=nb * Q5_0_BLOCK_BYTES).reshape(nb, Q5_0_BLOCK_BYTES)
    d = raw[:, :2].copy().view(np.float16).reshape(nb).astype(np.float32)  # (nb,)
    qh = raw[:, 2:6].copy().view("<u4").reshape(nb).astype(np.uint32)       # (nb,) packed 32 high bits
    qs = raw[:, 6:22]                                                       # (nb, 16) low nibbles+high nibbles

    out = np.empty((nb, 32), dtype=np.float32)
    js = np.arange(16, dtype=np.uint32)  # (16,)

    # ((qh >> j) << 4) & 0x10  for j in 0..15  -> "5th bit" for the low half
    xh_lo = ((qh[:, None] >> js[None, :]) << 4) & 0x10  # (nb, 16) uint32
    # ((qh >> (j + 12)) & 0x10  for j in 0..15  -> "5th bit" for the high half
    xh_hi = (qh[:, None] >> (js[None, :] + 12)) & 0x10  # (nb, 16)

    x0 = ((qs.astype(np.int32) & 0x0F) | xh_lo.astype(np.int32)) - 16  # (nb, 16)
    x1 = ((qs.astype(np.int32) >> 4) | xh_hi.astype(np.int32)) - 16    # (nb, 16)

    out[:, :16] = x0.astype(np.float32) * d[:, None]
    out[:, 16:] = x1.astype(np.float32) * d[:, None]
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Q4_K (K-quant, super-block of 256, 144 bytes)
# ---------------------------------------------------------------------------

def _unpack_q4k_scales_mins(scales12: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unpack the 12-byte scales array of every super-block.

    ``scales12`` has shape ``(nb, 12)`` (uint8). Returns ``(scales[nb,8],
    mins[nb,8])`` as float32 (the integer 6-bit values, before scaling by
    ``d`` and ``dmin``).
    """
    nb = scales12.shape[0]
    scales = np.empty((nb, 8), dtype=np.uint8)
    mins = np.empty((nb, 8), dtype=np.uint8)
    q = scales12  # alias

    # Sub-blocks 0..3: low 6 bits of bytes 0..3 (scales) and 4..7 (mins)
    scales[:, 0:4] = q[:, 0:4] & 0x3F
    mins[:, 0:4] = q[:, 4:8] & 0x3F

    # Sub-blocks 4..7: from bytes 8..11 plus the top 2 bits of bytes 0..7
    # *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4)
    # *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4)
    # where j runs 4..7, so q[j+4] is q[8..11], q[j-4] is q[0..3], q[j] is q[4..7]
    scales[:, 4:8] = (q[:, 8:12] & 0x0F) | ((q[:, 0:4] >> 6) << 4)
    mins[:, 4:8] = (q[:, 8:12] >> 4) | ((q[:, 4:8] >> 6) << 4)

    return scales.astype(np.float32), mins.astype(np.float32)


def dequant_q4_K(buf: bytes, n_elements: int) -> np.ndarray:
    assert n_elements % QK_K == 0
    nb = n_elements // QK_K
    raw = np.frombuffer(buf, dtype=np.uint8, count=nb * Q4K_BLOCK_BYTES).reshape(nb, Q4K_BLOCK_BYTES)

    d = raw[:, 0:2].copy().view(np.float16).reshape(nb).astype(np.float32)
    dmin = raw[:, 2:4].copy().view(np.float16).reshape(nb).astype(np.float32)
    scales12 = raw[:, 4:16]  # (nb, 12)
    qs = raw[:, 16:144]  # (nb, 128)

    sc, mn = _unpack_q4k_scales_mins(scales12)  # each (nb, 8)
    d_sub = d[:, None] * sc      # (nb, 8)
    m_sub = dmin[:, None] * mn   # (nb, 8)

    # qs is laid out in groups of 32 bytes per "j-stride" of 64 elements:
    #   for j in 0..QK_K step 64:
    #       low nibble of next 32 bytes -> 32 elements (sub-block is = j/32)
    #       high nibble of same 32 bytes -> 32 elements (sub-block is + 1)
    # i.e. 4 strides x 32 bytes = 128 bytes (matches qs size)
    qs_g = qs.reshape(nb, 4, 32)  # 4 strides
    low = qs_g & 0x0F  # (nb, 4, 32) -> sub-blocks 0,2,4,6
    high = qs_g >> 4   # (nb, 4, 32) -> sub-blocks 1,3,5,7

    out = np.empty((nb, 8, 32), dtype=np.float32)
    out[:, 0::2, :] = low.astype(np.float32) * d_sub[:, 0::2, None] - m_sub[:, 0::2, None]
    out[:, 1::2, :] = high.astype(np.float32) * d_sub[:, 1::2, None] - m_sub[:, 1::2, None]
    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Q6_K (super-block of 256, 210 bytes)
# ---------------------------------------------------------------------------

def dequant_q6_K(buf: bytes, n_elements: int) -> np.ndarray:
    assert n_elements % QK_K == 0
    nb = n_elements // QK_K
    raw = np.frombuffer(buf, dtype=np.uint8, count=nb * Q6K_BLOCK_BYTES).reshape(nb, Q6K_BLOCK_BYTES)

    # Layout: ql[128] qh[64] scales[16] d(2)
    ql = raw[:, 0:128]
    qh = raw[:, 128:192]
    sc = raw[:, 192:208].view(np.int8).astype(np.float32)  # (nb, 16) int8 sub-block scales
    d = raw[:, 208:210].copy().view(np.float16).reshape(nb).astype(np.float32)

    out = np.empty((nb, 256), dtype=np.float32)

    # Reference loop (per 128-element half of the super-block):
    #   for n in 0..QK_K step 128:
    #     for l in 0..32:
    #       q1 = ((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32
    #       q2 = ((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32
    #       q3 = ((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32
    #       q4 = ((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32
    #       y[l +  0]  = d * sc[is + 0] * q1   (is = l/16, => 0 or 1)
    #       y[l + 32]  = d * sc[is + 2] * q2
    #       y[l + 64]  = d * sc[is + 4] * q3
    #       y[l + 96]  = d * sc[is + 6] * q4
    # Then advance: y += 128, ql += 64, qh += 32, sc += 8.

    for half in range(2):  # two halves of 128 elements each
        ql_h = ql[:, half * 64 : half * 64 + 64]      # (nb, 64)
        qh_h = qh[:, half * 32 : half * 32 + 32]      # (nb, 32)
        sc_h = sc[:, half * 8 : half * 8 + 8]         # (nb, 8)

        ql0 = ql_h[:, 0:32].astype(np.int32)
        ql1 = ql_h[:, 32:64].astype(np.int32)
        qh0 = qh_h.astype(np.int32)

        q1 = ((ql0 & 0x0F) | ((qh0 >> 0 & 0x3) << 4)) - 32  # (nb, 32)
        q2 = ((ql1 & 0x0F) | ((qh0 >> 2 & 0x3) << 4)) - 32
        q3 = ((ql0 >> 4)   | ((qh0 >> 4 & 0x3) << 4)) - 32
        q4 = ((ql1 >> 4)   | ((qh0 >> 6 & 0x3) << 4)) - 32

        # is = l/16 over l in 0..32 -> 0 for l<16, 1 for l>=16
        is_idx = (np.arange(32) // 16).astype(np.int64)  # (32,)

        sc_for_q1 = sc_h[:, 0:2][:, is_idx]  # (nb, 32) — sc[is+0] for is in {0,1}
        sc_for_q2 = sc_h[:, 2:4][:, is_idx]  # sc[is+2]
        sc_for_q3 = sc_h[:, 4:6][:, is_idx]  # sc[is+4]
        sc_for_q4 = sc_h[:, 6:8][:, is_idx]  # sc[is+6]

        d_b = d[:, None]
        out_h = out[:, half * 128 : (half + 1) * 128]
        out_h[:, 0:32]  = d_b * sc_for_q1 * q1.astype(np.float32)
        out_h[:, 32:64] = d_b * sc_for_q2 * q2.astype(np.float32)
        out_h[:, 64:96] = d_b * sc_for_q3 * q3.astype(np.float32)
        out_h[:, 96:128] = d_b * sc_for_q4 * q4.astype(np.float32)

    return out.reshape(-1)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

# ggml type id -> dequant function
DEQUANT: dict[int, callable] = {
    0: dequant_f32,
    1: dequant_f16,
    6: dequant_q5_0,
    8: dequant_q8_0,
    12: dequant_q4_K,
    14: dequant_q6_K,
    30: dequant_bf16,
}


def dequantize(buf: bytes, ggml_type: int, n_elements: int) -> np.ndarray:
    fn = DEQUANT.get(ggml_type)
    if fn is None:
        raise NotImplementedError(f"no dequantizer for ggml type {ggml_type}")
    out = fn(buf, n_elements)
    if out.size != n_elements:
        raise ValueError(
            f"dequant produced {out.size} elements, expected {n_elements} (type {ggml_type})"
        )
    return out
