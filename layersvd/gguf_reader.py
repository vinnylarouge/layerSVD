"""GGUF v3 reader (metadata + tensor index).

Stdlib only. Decodes the header and tensor descriptors of a GGUF file
without loading any tensor data into memory. Tensor data is read on
demand via :func:`read_tensor_bytes` (mmap-backed).

The GGUF spec we follow:
https://github.com/ggerganov/ggml/blob/master/docs/gguf.md
"""

from __future__ import annotations

import mmap
import os
import struct
from dataclasses import dataclass
from typing import Any, Iterable

# ggml type id -> (name, block_size_elements, block_bytes)
# Only the types that appear in our Gemma 4 GGUF file are populated
# precisely; others are name-only.
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    30: ("BF16", 1, 2),
}

# GGUF metadata value types
GGUF_TYPE_UINT8 = 0
GGUF_TYPE_INT8 = 1
GGUF_TYPE_UINT16 = 2
GGUF_TYPE_INT16 = 3
GGUF_TYPE_UINT32 = 4
GGUF_TYPE_INT32 = 5
GGUF_TYPE_FLOAT32 = 6
GGUF_TYPE_BOOL = 7
GGUF_TYPE_STRING = 8
GGUF_TYPE_ARRAY = 9
GGUF_TYPE_UINT64 = 10
GGUF_TYPE_INT64 = 11
GGUF_TYPE_FLOAT64 = 12


@dataclass
class TensorDesc:
    name: str
    shape: tuple[int, ...]
    ggml_type: int
    type_name: str
    offset: int  # offset relative to data section start
    nbytes: int  # raw stored bytes (after dequant->fp32 this multiplies)


@dataclass
class GGUF:
    path: str
    version: int
    metadata: dict[str, Any]
    tensors: list[TensorDesc]
    data_offset: int  # absolute file offset where tensor data begins

    def tensor(self, name: str) -> TensorDesc:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)

    def tensors_matching(self, predicate) -> list[TensorDesc]:
        return [t for t in self.tensors if predicate(t)]


def _read_string(buf: memoryview, off: int) -> tuple[str, int]:
    n = struct.unpack_from("<Q", buf, off)[0]
    off += 8
    s = bytes(buf[off : off + n]).decode("utf-8", "replace")
    return s, off + n


def _read_value(buf: memoryview, off: int, vtype: int) -> tuple[Any, int]:
    if vtype == GGUF_TYPE_UINT8:
        return struct.unpack_from("<B", buf, off)[0], off + 1
    if vtype == GGUF_TYPE_INT8:
        return struct.unpack_from("<b", buf, off)[0], off + 1
    if vtype == GGUF_TYPE_UINT16:
        return struct.unpack_from("<H", buf, off)[0], off + 2
    if vtype == GGUF_TYPE_INT16:
        return struct.unpack_from("<h", buf, off)[0], off + 2
    if vtype == GGUF_TYPE_UINT32:
        return struct.unpack_from("<I", buf, off)[0], off + 4
    if vtype == GGUF_TYPE_INT32:
        return struct.unpack_from("<i", buf, off)[0], off + 4
    if vtype == GGUF_TYPE_FLOAT32:
        return struct.unpack_from("<f", buf, off)[0], off + 4
    if vtype == GGUF_TYPE_BOOL:
        return bool(struct.unpack_from("<B", buf, off)[0]), off + 1
    if vtype == GGUF_TYPE_STRING:
        return _read_string(buf, off)
    if vtype == GGUF_TYPE_UINT64:
        return struct.unpack_from("<Q", buf, off)[0], off + 8
    if vtype == GGUF_TYPE_INT64:
        return struct.unpack_from("<q", buf, off)[0], off + 8
    if vtype == GGUF_TYPE_FLOAT64:
        return struct.unpack_from("<d", buf, off)[0], off + 8
    if vtype == GGUF_TYPE_ARRAY:
        et = struct.unpack_from("<I", buf, off)[0]
        n = struct.unpack_from("<Q", buf, off + 4)[0]
        off += 12
        out: list[Any] = []
        for _ in range(n):
            v, off = _read_value(buf, off, et)
            out.append(v)
        return out, off
    raise ValueError(f"unknown gguf value type {vtype}")


def read(path: str) -> GGUF:
    """Read GGUF metadata + tensor index. Tensor data stays on disk."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    buf = memoryview(mm)

    magic = bytes(buf[0:4])
    if magic != b"GGUF":
        raise ValueError(f"not a GGUF file (magic={magic!r})")
    version = struct.unpack_from("<I", buf, 4)[0]
    if version != 3:
        raise ValueError(f"unsupported GGUF version {version}, only v3 supported")
    n_tensors = struct.unpack_from("<Q", buf, 8)[0]
    n_kv = struct.unpack_from("<Q", buf, 16)[0]

    off = 24
    metadata: dict[str, Any] = {}
    for _ in range(n_kv):
        key, off = _read_string(buf, off)
        vtype = struct.unpack_from("<I", buf, off)[0]
        off += 4
        val, off = _read_value(buf, off, vtype)
        metadata[key] = val

    # Tensor descriptors
    descs: list[TensorDesc] = []
    for _ in range(n_tensors):
        name, off = _read_string(buf, off)
        n_dims = struct.unpack_from("<I", buf, off)[0]
        off += 4
        shape = struct.unpack_from(f"<{n_dims}Q", buf, off)
        off += 8 * n_dims
        gtype = struct.unpack_from("<I", buf, off)[0]
        off += 4
        toff = struct.unpack_from("<Q", buf, off)[0]
        off += 8
        type_name, block_n, block_b = GGML_TYPES.get(gtype, (f"GGML?{gtype}", 1, 0))
        n_elements = 1
        for d in shape:
            n_elements *= int(d)
        if block_n == 0:
            nbytes = 0
        else:
            n_blocks = (n_elements + block_n - 1) // block_n
            nbytes = n_blocks * block_b
        descs.append(
            TensorDesc(
                name=name,
                shape=tuple(int(d) for d in shape),
                ggml_type=gtype,
                type_name=type_name,
                offset=toff,
                nbytes=nbytes,
            )
        )

    # GGUF aligns the data section start to alignment boundary (default 32).
    alignment = int(metadata.get("general.alignment", 32))
    header_end = off
    data_offset = (header_end + alignment - 1) // alignment * alignment

    # Release mmap so we can re-open per call
    buf.release()
    mm.close()

    return GGUF(
        path=path,
        version=version,
        metadata=metadata,
        tensors=descs,
        data_offset=data_offset,
    )


def open_data(path: str) -> tuple[mmap.mmap, memoryview]:
    """Open the GGUF file's data area for random access. Caller closes."""
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        mm = mmap.mmap(fd, size, prot=mmap.PROT_READ)
    finally:
        os.close(fd)
    return mm, memoryview(mm)


def read_tensor_bytes(gguf: GGUF, t: TensorDesc, mm_view: memoryview) -> bytes:
    """Slice the raw bytes for one tensor from a previously-opened mmap view."""
    start = gguf.data_offset + t.offset
    return bytes(mm_view[start : start + t.nbytes])


def summarize(gguf: GGUF) -> dict[str, Any]:
    """Compact summary suitable for jsonification."""
    from collections import Counter

    type_counts: Counter[str] = Counter()
    for t in gguf.tensors:
        type_counts[t.type_name] += 1

    arch = gguf.metadata.get("general.architecture", "?")
    name = gguf.metadata.get("general.name", "?")
    size_label = gguf.metadata.get("general.size_label", "?")
    return {
        "version": gguf.version,
        "architecture": arch,
        "name": name,
        "size_label": size_label,
        "n_tensors": len(gguf.tensors),
        "n_kv": len(gguf.metadata),
        "tensor_type_counts": dict(type_counts),
        "data_offset": gguf.data_offset,
    }


def main() -> None:
    import sys

    if len(sys.argv) != 2:
        print("usage: python -m layersvd.gguf_reader <gguf-file>")
        raise SystemExit(2)
    g = read(sys.argv[1])
    s = summarize(g)
    for k, v in s.items():
        print(f"{k}: {v}")
    print("\n--- arch metadata ---")
    arch = s["architecture"]
    for k, v in g.metadata.items():
        if k.startswith(arch + ".") or k.startswith("general."):
            sv = str(v)
            if len(sv) > 120:
                sv = sv[:120] + "..."
            print(f"  {k} = {sv}")
    print("\n--- first 12 tensors ---")
    for t in g.tensors[:12]:
        print(f"  {t.type_name:6s}  shape={t.shape}  off={t.offset}  bytes={t.nbytes}  {t.name}")


if __name__ == "__main__":
    main()
