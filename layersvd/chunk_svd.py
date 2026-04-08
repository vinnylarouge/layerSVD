"""Split monolithic ``U.f16`` / ``V.f16`` into per-layer chunk files.

GitHub blocks single files larger than 100 MB on the regular blob path.
Our K=64 SVD bundle has each of ``U.f16`` and ``V.f16`` slightly above
that. We split per-layer (layer -1 = the global ``token_embd`` plus
anything else without a block parent) so each chunk stays ~3-15 MB.

Updates the manifest in place: each entry gains ``u_file`` and ``v_file``,
and ``u_offset`` / ``v_offset`` are rewritten to be relative to the
chunked files. The original monolithic files are removed.

Run::

    .venv/bin/python -m layersvd.chunk_svd docs/data/svd
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m layersvd.chunk_svd <svd-dir>")
        raise SystemExit(2)
    base = Path(sys.argv[1])
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    entries = manifest["entries"]

    # Read source files
    u_path = base / "U.f16"
    v_path = base / "V.f16"
    u_data = u_path.read_bytes()
    v_data = v_path.read_bytes()

    # Group entries by (layer); -1 stays as its own bucket.
    by_layer: dict[int, list[dict]] = {}
    for e in entries:
        by_layer.setdefault(e["layer"], []).append(e)

    # Sort each bucket by current u_offset so files are contiguous.
    for L, lst in by_layer.items():
        lst.sort(key=lambda e: e["u_offset"])

    # Write per-layer chunks
    new_u_offsets: dict[str, int] = {}
    new_v_offsets: dict[str, int] = {}
    for L in sorted(by_layer.keys()):
        u_chunk = bytearray()
        v_chunk = bytearray()
        for e in by_layer[L]:
            tid = e["tensor_id"]
            u_chunk += u_data[e["u_offset"] : e["u_offset"] + e["u_bytes"]]
            v_chunk += v_data[e["v_offset"] : e["v_offset"] + e["v_bytes"]]
        u_name = f"U-L{L:+03d}.f16"
        v_name = f"V-L{L:+03d}.f16"
        (base / u_name).write_bytes(u_chunk)
        (base / v_name).write_bytes(v_chunk)
        # Now compute new offsets within the chunk
        cu, cv = 0, 0
        for e in by_layer[L]:
            tid = e["tensor_id"]
            new_u_offsets[tid] = cu
            new_v_offsets[tid] = cv
            cu += e["u_bytes"]
            cv += e["v_bytes"]

    # Update entries
    for e in entries:
        L = e["layer"]
        tid = e["tensor_id"]
        e["u_file"] = f"U-L{L:+03d}.f16"
        e["v_file"] = f"V-L{L:+03d}.f16"
        e["u_offset"] = new_u_offsets[tid]
        e["v_offset"] = new_v_offsets[tid]

    manifest["chunked"] = True
    manifest_path.write_text(json.dumps(manifest, indent=2))

    # Remove monolithic files
    u_path.unlink()
    v_path.unlink()
    print(f"chunked into {len(by_layer)} layer files; removed U.f16/V.f16")
    # Verify all chunks are below 100 MB
    over = []
    for f in sorted(base.iterdir()):
        sz = f.stat().st_size
        if sz > 100 * 1024 * 1024:
            over.append((f.name, sz))
        if f.suffix == ".f16":
            print(f"  {f.name}: {sz/1024/1024:.1f} MB")
    if over:
        print("WARNING: still oversized:", over)


if __name__ == "__main__":
    main()
