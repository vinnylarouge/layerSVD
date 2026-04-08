// SVD spectrum browser.
//
// On load: fetches the SVD manifest, builds layer + tensor dropdowns,
// and on change fetches the relevant byte ranges from spectra.bin / U.f16
// / V.f16 to render: spectrum (singular values), top-K right-singular-vector
// heatmap, and a static cross-layer flow heatmap (U_L^T V_{L+1}).

(async function () {
  const manifest = await loadSVDManifest();
  const arch = await loadArchitecture();
  const entries = manifest.entries;
  const byId = new Map(entries.map((e) => [e.tensor_id, e]));

  // Group entries by layer
  const layers = new Map();
  for (const e of entries) {
    if (e.layer < 0) continue;
    if (!layers.has(e.layer)) layers.set(e.layer, []);
    layers.get(e.layer).push(e);
  }

  // Populate layer dropdown
  const layerSel = document.getElementById("layer-sel");
  for (let i = 0; i < arch.model.n_layer; i++) {
    const opt = document.createElement("option");
    opt.value = String(i);
    const tag = arch.layers[i].is_swa ? "SWA" : "GLOBAL";
    opt.textContent = `blk.${i} (${tag})`;
    layerSel.appendChild(opt);
  }

  // Tensor dropdown — filled when a layer is selected
  const tensorSel = document.getElementById("tensor-sel");
  function repopulateTensors() {
    const L = parseInt(layerSel.value, 10);
    tensorSel.innerHTML = "";
    const ents = (layers.get(L) || []).slice().sort((a, b) => a.tensor_id.localeCompare(b.tensor_id));
    for (const e of ents) {
      const opt = document.createElement("option");
      opt.value = e.tensor_id;
      opt.textContent = `${e.role.padEnd(8)}  K=${e.K}  ${e.shape.join("×")}  ${e.group}`;
      tensorSel.appendChild(opt);
    }
  }
  layerSel.addEventListener("change", () => { repopulateTensors(); render(); });
  tensorSel.addEventListener("change", render);

  // Default to a useful starting tensor
  layerSel.value = "0";
  repopulateTensors();
  // Try to default to ffn_down_up if it exists
  for (const opt of tensorSel.options) {
    if (opt.value.endsWith("ffn_down_up")) { tensorSel.value = opt.value; break; }
  }

  await render();

  // ----- rendering -----------------------------------------------------

  async function render() {
    const tid = tensorSel.value;
    const e = byId.get(tid);
    if (!e) return;

    const detail = document.getElementById("detail");
    const fro_kept = e.fro_kept;
    const fro_total = e.fro_total;
    const captured = fro_total > 0 ? (fro_kept / fro_total) * 100 : 0;
    detail.innerHTML = `<strong>${e.tensor_id}</strong> · ${e.shape.join("×")} · K=${e.K} ·
        captured ${captured.toFixed(1)}% Frobenius · σ₀=${e.s_max.toExponential(2)} · σ_K=${e.s_min.toExponential(2)}`;

    const s = await fetchFP32Block("spectra.bin", e.s_offset, e.s_bytes);
    drawSpectrum(s);

    // Right singular vectors heatmap (downsample columns to ~600)
    const V = await fetchFP16Block(e.v_file || "V.f16", e.v_offset, e.v_bytes);
    const K = e.K;
    const inDim = e.shape[1];
    drawHeatmap(V, K, inDim);

    // Cross-layer flow: this layer's left singular vectors (ffn_down_up if available)
    // vs next layer's right singular vectors of the same.
    drawFlow(e);
  }

  function drawSpectrum(s) {
    const c = document.getElementById("spectrum");
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);
    const PAD = 28;
    const w = W - PAD * 2;
    const h = H - PAD * 2;
    const n = s.length;
    const max = Math.max(...s);
    const min = Math.max(1e-12, Math.min(...s.filter((x) => x > 0)));
    const xs = (i) => PAD + (i / (n - 1)) * w;
    const ys = (v) => PAD + h - (Math.log10(v / min) / Math.log10(max / min)) * h;
    // axis
    ctx.strokeStyle = "#2a3344"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(PAD, PAD); ctx.lineTo(PAD, PAD + h); ctx.lineTo(PAD + w, PAD + h); ctx.stroke();
    // bars
    ctx.fillStyle = "#6ab0ff";
    for (let i = 0; i < n; i++) {
      const x = xs(i);
      const y = ys(Math.max(s[i], 1e-12));
      ctx.fillRect(x - 1, y, 2, PAD + h - y);
    }
    // labels
    ctx.fillStyle = "#8a96aa"; ctx.font = "10px ui-monospace,Menlo,monospace";
    ctx.textAlign = "left";
    ctx.fillText(`σ₀ = ${max.toExponential(2)}`, PAD + 4, PAD + 12);
    ctx.fillText(`σ_${n - 1} = ${s[n - 1].toExponential(2)}`, PAD + 4, PAD + 24);
    ctx.textAlign = "center";
    ctx.fillText("singular value index (log y)", PAD + w / 2, PAD + h + 18);
  }

  function drawHeatmap(V, K, inDim) {
    const c = document.getElementById("heatmap");
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);

    // Downsample to a manageable image: rows = K, cols = min(W, inDim)
    const cols = Math.min(W, 512);
    const stride = inDim / cols;
    // V is (K, inDim) flat -- compute amplitude per cell
    const img = ctx.createImageData(cols, K);
    let maxAbs = 1e-12;
    for (let i = 0; i < K * inDim; i++) {
      const v = Math.abs(V[i]);
      if (v > maxAbs) maxAbs = v;
    }
    for (let i = 0; i < K; i++) {
      for (let j = 0; j < cols; j++) {
        const j_in = Math.floor(j * stride);
        const v = V[i * inDim + j_in];
        const t = Math.max(-1, Math.min(1, v / maxAbs));
        // Diverging colormap: blue (negative) -> black (0) -> orange (positive)
        let r, g, b;
        if (t >= 0) {
          r = Math.floor(255 * t);
          g = Math.floor(160 * t);
          b = Math.floor(80 * t);
        } else {
          r = Math.floor(80 * -t);
          g = Math.floor(160 * -t);
          b = Math.floor(255 * -t);
        }
        const idx = ((K - 1 - i) * cols + j) * 4;  // flip y
        img.data[idx] = r;
        img.data[idx + 1] = g;
        img.data[idx + 2] = b;
        img.data[idx + 3] = 255;
      }
    }
    // Draw scaled to canvas
    const off = document.createElement("canvas");
    off.width = cols; off.height = K;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, H);
  }

  async function drawFlow(e) {
    const c = document.getElementById("flow");
    const ctx = c.getContext("2d");
    const W = c.width, H = c.height;
    ctx.clearRect(0, 0, W, H);

    // Find ffn_down_up entries for this layer and the next.
    const L = e.layer;
    const tidL = `blk.${L}.ffn_down_up`;
    const tidL1 = `blk.${L + 1}.ffn_down_up`;
    const eL = byId.get(tidL);
    const eL1 = byId.get(tidL1);
    if (!eL || !eL1) {
      ctx.fillStyle = "#8a96aa";
      ctx.font = "12px ui-monospace,Menlo,monospace";
      ctx.fillText("(no adjacent ffn_down_up SVD pair available)", 12, 24);
      return;
    }

    const U_L = await fetchFP16Block(eL.u_file || "U.f16", eL.u_offset, eL.u_bytes);
    const V_L1 = await fetchFP16Block(eL1.v_file || "V.f16", eL1.v_offset, eL1.v_bytes);
    const KL = eL.K;
    const KL1 = eL1.K;
    const dim = eL.shape[0]; // d_model — same on both

    // U_L is stored as (m, K) row-major flat: m=dim, K=KL
    // V_L1 is stored as (K, n) row-major flat: K=KL1, n=dim
    // We want C[i, j] = <U_L[:, i], V_L1[j, :]> = sum_d U_L[d, i] * V_L1[j, d]
    const C = new Float32Array(KL * KL1);
    for (let i = 0; i < KL; i++) {
      for (let j = 0; j < KL1; j++) {
        let s = 0;
        for (let d = 0; d < dim; d++) {
          s += U_L[d * KL + i] * V_L1[j * dim + d];
        }
        C[i * KL1 + j] = s;
      }
    }

    // Render absolute values as a heatmap.
    let maxAbs = 1e-12;
    for (let i = 0; i < KL * KL1; i++) {
      const v = Math.abs(C[i]);
      if (v > maxAbs) maxAbs = v;
    }
    const img = ctx.createImageData(KL1, KL);
    for (let i = 0; i < KL; i++) {
      for (let j = 0; j < KL1; j++) {
        const t = Math.abs(C[i * KL1 + j]) / maxAbs;
        const v = Math.floor(255 * Math.pow(t, 0.5));
        const idx = (i * KL1 + j) * 4;
        img.data[idx] = v;
        img.data[idx + 1] = Math.floor(v * 0.6);
        img.data[idx + 2] = Math.floor(v * 0.2);
        img.data[idx + 3] = 255;
      }
    }
    const off = document.createElement("canvas");
    off.width = KL1; off.height = KL;
    off.getContext("2d").putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(off, 0, 0, W, H);

    ctx.fillStyle = "#8a96aa";
    ctx.font = "11px ui-monospace,Menlo,monospace";
    ctx.fillText(`L=${L} ffn_down_up   →   L=${L + 1} ffn_down_up    max |cos|=${maxAbs.toExponential(2)}`, 8, 14);
  }
})().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<pre style="color:#ff6f6f;padding:20px;">${err.stack || err.message}</pre>`
  );
});
