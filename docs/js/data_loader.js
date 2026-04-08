// Shared async loaders for the layerSVD static site.
//
// All data lives under docs/data/ and is fetched relative to the page.

const DATA = {
  arch: null,
  svd_manifest: null,
  trace_index: null,
};

async function loadArchitecture() {
  if (DATA.arch) return DATA.arch;
  const r = await fetch("data/architecture.json");
  if (!r.ok) throw new Error("failed to load architecture.json: " + r.status);
  DATA.arch = await r.json();
  return DATA.arch;
}

async function loadSVDManifest() {
  if (DATA.svd_manifest) return DATA.svd_manifest;
  const r = await fetch("data/svd/manifest.json");
  if (!r.ok) throw new Error("failed to load svd manifest: " + r.status);
  DATA.svd_manifest = await r.json();
  return DATA.svd_manifest;
}

// In-memory cache for full SVD files when the server doesn't honor Range.
const SVD_FULL_CACHE = new Map();

async function loadSVDByteRange(file, offset, nbytes) {
  // Try a Range request first (production: GitHub Pages supports them).
  try {
    const r = await fetch("data/svd/" + file, {
      headers: { Range: `bytes=${offset}-${offset + nbytes - 1}` },
    });
    if (r.status === 206) {
      return new Uint8Array(await r.arrayBuffer());
    }
    if (r.ok) {
      // Server returned the full file (e.g. python http.server).
      // Cache it and slice locally.
      let full = SVD_FULL_CACHE.get(file);
      if (!full) {
        full = new Uint8Array(await r.arrayBuffer());
        SVD_FULL_CACHE.set(file, full);
      }
      return full.subarray(offset, offset + nbytes);
    }
    throw new Error("fetch failed: " + r.status);
  } catch (e) {
    // Fallback: full-file load with caching
    if (!SVD_FULL_CACHE.has(file)) {
      const r = await fetch("data/svd/" + file);
      if (!r.ok) throw new Error("fallback fetch failed: " + r.status);
      SVD_FULL_CACHE.set(file, new Uint8Array(await r.arrayBuffer()));
    }
    const full = SVD_FULL_CACHE.get(file);
    return full.subarray(offset, offset + nbytes);
  }
}

function fp16ToFp32(u16) {
  // u16: Uint16Array of half-precision values
  const out = new Float32Array(u16.length);
  for (let i = 0; i < u16.length; i++) {
    const h = u16[i];
    const s = (h & 0x8000) >> 15;
    const e = (h & 0x7c00) >> 10;
    const f = h & 0x03ff;
    let val;
    if (e === 0) {
      val = (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
    } else if (e === 0x1f) {
      val = f ? NaN : (s ? -Infinity : Infinity);
    } else {
      val = (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
    }
    out[i] = val;
  }
  return out;
}

async function fetchFP16Block(file, offset, nbytes) {
  const u8 = await loadSVDByteRange(file, offset, nbytes);
  // Reinterpret as Uint16 little-endian
  const u16 = new Uint16Array(u8.buffer, u8.byteOffset, nbytes / 2);
  return fp16ToFp32(u16);
}

async function fetchFP32Block(file, offset, nbytes) {
  const u8 = await loadSVDByteRange(file, offset, nbytes);
  return new Float32Array(u8.buffer, u8.byteOffset, nbytes / 4);
}

function fmtParams(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(2) + "B";
  if (n >= 1e6) return (n / 1e6).toFixed(1) + "M";
  if (n >= 1e3) return (n / 1e3).toFixed(1) + "K";
  return String(n);
}

function fmtBytes(n) {
  if (n >= 1024 * 1024 * 1024) return (n / 1024 / 1024 / 1024).toFixed(2) + " GiB";
  if (n >= 1024 * 1024) return (n / 1024 / 1024).toFixed(1) + " MiB";
  if (n >= 1024) return (n / 1024).toFixed(1) + " KiB";
  return n + " B";
}
