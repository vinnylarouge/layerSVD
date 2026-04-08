// Architecture page — flowchart (layout A) + small multiples ribbon (layout B).
//
// Tufte ground rules in effect:
//   - sparklines, no stat tables (stats live in the hover tooltip)
//   - one global log-σ y-axis shared across every sparkline on the page
//   - color carries one bit only: SWA (cool) vs global (warm)
//   - residual rail is one thin line; tee marks are one stroke each

const SVG_NS = "http://www.w3.org/2000/svg";
function el(tag, attrs = {}, parent = null) {
  const node = document.createElementNS(SVG_NS, tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (v !== undefined && v !== null) node.setAttribute(k, String(v));
  }
  if (parent) parent.appendChild(node);
  return node;
}

(async function main() {
  const [arch, manifest] = await Promise.all([loadArchitecture(), loadSVDManifest()]);
  // Pull spectra.bin once and slice locally — it's only ~75 KB.
  const spectraU8 = await loadSVDByteRange("spectra.bin", 0, manifest.totals.spectra_bytes);
  const spectraF32 = new Float32Array(spectraU8.buffer, spectraU8.byteOffset, spectraU8.byteLength / 4);

  const byId = new Map(manifest.entries.map((e) => [e.tensor_id, e]));

  // Pick the most informative tensor per layer per role.
  // Roles we plot: 'attn' (composed wo_wv if present, else attn_q),
  //                'ffn'  (composed ffn_down_up),
  //                'router' (ffn_gate_inp).
  function pickAttnTid(L) {
    const composed = `blk.${L}.attn_wo_wv`;
    if (byId.has(composed)) return composed;
    return `blk.${L}.attn_q`; // fall back for global layers
  }
  function pickFfnTid(L)   { return `blk.${L}.ffn_down_up`; }
  function pickRouterTid(L){ return `blk.${L}.ffn_gate_inp`; }

  // Spectra fetcher: returns Float32Array view (logged) of K values for a tensor.
  function spectraOf(tid) {
    const e = byId.get(tid);
    if (!e) return null;
    const start = e.s_offset / 4;
    return spectraF32.subarray(start, start + e.K);
  }

  // ---- compute global log-sigma range across everything we will plot ----
  // We include attn, ffn, router for every layer plus the token embedding.
  let logMin = +Infinity, logMax = -Infinity;
  function eatRange(s) {
    if (!s) return;
    for (let i = 0; i < s.length; i++) {
      const v = s[i];
      if (v <= 0) continue;
      const lv = Math.log10(v);
      if (lv < logMin) logMin = lv;
      if (lv > logMax) logMax = lv;
    }
  }
  for (let L = 0; L < arch.model.n_layer; L++) {
    eatRange(spectraOf(pickAttnTid(L)));
    eatRange(spectraOf(pickFfnTid(L)));
    eatRange(spectraOf(pickRouterTid(L)));
  }
  eatRange(spectraOf("token_embd"));
  // Pad the range a tiny bit so values at the extremes aren't on the edge.
  const logSpan = logMax - logMin;
  logMin -= logSpan * 0.05;
  logMax += logSpan * 0.05;

  // Sparkline drawing primitive.
  function makeSparkline(parent, x, y, w, h, spectrum, label) {
    if (!spectrum || spectrum.length === 0) {
      el("text", { x: x + w / 2, y: y + h / 2 + 3, "text-anchor": "middle",
                   class: "spark-label" }, parent).textContent = "—";
      return;
    }
    // Background dashed baseline (top + bottom of log range)
    el("line", { x1: x, y1: y, x2: x + w, y2: y, class: "spark-base" }, parent);
    el("line", { x1: x, y1: y + h, x2: x + w, y2: y + h, class: "spark-base" }, parent);

    const n = spectrum.length;
    const xs = (i) => x + (i / (n - 1)) * w;
    const ys = (v) => {
      if (v <= 0) return y + h;
      const lv = Math.log10(v);
      const t = (lv - logMin) / (logMax - logMin);
      return y + h - t * h;
    };

    let d = "";
    for (let i = 0; i < n; i++) {
      d += (i === 0 ? "M" : "L") + xs(i).toFixed(1) + " " + ys(spectrum[i]).toFixed(1);
    }
    el("path", { d, class: "spark-line" }, parent);

    if (label) {
      el("text", { x, y: y - 3, class: "spark-label" }, parent).textContent = label;
    }
  }

  // -------- one-line model summary --------
  renderModelLine(arch);

  // -------- Layout A: flowchart with residual rail --------
  renderFlow(arch, byId, spectraOf, makeSparkline);

  // -------- Layout B: small multiples ribbon --------
  renderRibbon(arch, byId, spectraOf, makeSparkline);

  // -------- tooltip wiring --------
  setupTooltip(arch);
})().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<pre style="color:#ff6f6f;padding:20px;">${err.stack || err.message}</pre>`
  );
});


// ---------- model line --------------------------------------------------

function renderModelLine(arch) {
  const m = arch.model;
  const root = document.getElementById("model-line");
  const sep = '<span class="sep">·</span>';
  root.innerHTML = [
    `<strong>${m.name}</strong>`,
    `${m.n_layer} blocks`,
    `d_model <strong>${m.n_embd}</strong>`,
    `q×<strong>${m.n_head}</strong>`,
    `kv × {8,2} ISWA`,
    `${m.n_expert} experts top-${m.n_expert_used}`,
    `vocab <strong>${m.vocab_size.toLocaleString()}</strong>`,
    `ctx <strong>${m.context_length.toLocaleString()}</strong>`,
    `softcap <strong>${m.final_logit_softcap}</strong>`,
    `${(m.total_params / 1e9).toFixed(2)}B params`,
  ].join(sep);
}


// ---------- Layout A: flowchart with residual rail ----------------------

function renderFlow(arch, byId, spectraOf, makeSparkline) {
  const svg = document.getElementById("flow");
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const W = 1200;
  const HEAD_H = 92;
  const FOOT_H = 80;
  const ROW_H = 64;
  const N = arch.model.n_layer;
  const H = HEAD_H + N * ROW_H + FOOT_H;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");

  // Geometry
  const RAIL_X = 78;          // residual rail center
  const BLOCK_X = 130;
  const BLOCK_W = W - BLOCK_X - 24;
  const SP_W = 200;            // sparkline width
  const SP_H = 28;             // sparkline height
  const SP_GAP = 28;
  const SP_Y_OFF = 22;         // sparkline top offset within a block

  // Header — token embedding panel + initial scaling note
  {
    const g = el("g", { transform: `translate(${BLOCK_X},10)` }, svg);
    el("text", { x: 0, y: 14, class: "panel-title" }, g).textContent =
      `token_embd  ${arch.globals.token_embd.shape.join("×")}  ${arch.globals.token_embd.type}`;
    el("text", { x: 0, y: 30, class: "panel-sub" }, g).textContent =
      `× √d_model · tied with lm_head · vocab ${arch.model.vocab_size.toLocaleString()}`;
    // Vocab spectrum sparkline (token_embd has K=32)
    const sx = BLOCK_W - SP_W - 16;
    const sy = 12;
    el("text", { x: sx, y: sy + 6, class: "spark-label" }, g).textContent = "vocab spectrum";
    makeSparkline(g, sx, sy + 12, SP_W, SP_H, spectraOf("token_embd"), null);
  }

  // Vertical residual rail
  el("line", {
    x1: RAIL_X, y1: HEAD_H - 8,
    x2: RAIL_X, y2: HEAD_H + N * ROW_H + 6,
    class: "rail",
  }, svg);
  // Tiny "× √d" arrow into the rail
  el("line", { x1: BLOCK_X + 30, y1: HEAD_H - 12, x2: RAIL_X, y2: HEAD_H - 4, class: "tee" }, svg);

  // Each block
  for (let L = 0; L < N; L++) {
    const layer = arch.layers[L];
    const yTop = HEAD_H + L * ROW_H;
    const yMid = yTop + ROW_H / 2;
    const isSwa = layer.is_swa;

    // Two tee strokes from the rail into the block:
    //   upper tee = attention residual add
    //   lower tee = FFN residual add
    el("line", {
      x1: RAIL_X, y1: yTop + 18, x2: BLOCK_X, y2: yTop + 18, class: "tee",
    }, svg);
    el("line", {
      x1: RAIL_X, y1: yTop + ROW_H - 14, x2: BLOCK_X, y2: yTop + ROW_H - 14, class: "tee",
    }, svg);

    // Block group (whole row, hoverable)
    const g = el("g", { class: "block", "data-layer": String(L) }, svg);

    // Background (transparent by default; CSS hover brings it in)
    el("rect", {
      x: BLOCK_X, y: yTop + 4, width: BLOCK_W, height: ROW_H - 8,
      class: "block-bg", rx: 2,
    }, g);

    // Color marker on the left edge of the block (one bit: SWA vs global)
    el("rect", {
      x: BLOCK_X, y: yTop + 4, width: 3, height: ROW_H - 8,
      class: `block-mark ${isSwa ? "swa" : "global"}`,
    }, g);

    // Block label (left side)
    const labelX = BLOCK_X + 12;
    el("text", { x: labelX, y: yTop + 18, class: "block-label" }, g).textContent =
      `blk.${L.toString().padStart(2, "0")}`;
    el("text", { x: labelX, y: yTop + 32, class: "block-sub" }, g).textContent =
      isSwa ? "SWA" : "GLOBAL";
    el("text", { x: labelX, y: yTop + 44, class: "block-sub" }, g).textContent =
      `q${layer.n_head} kv${layer.n_head_kv} hd${layer.head_dim}`;

    // Three sparklines, right-aligned
    const sp1x = BLOCK_X + 100;
    const sp2x = sp1x + SP_W + SP_GAP;
    const sp3x = sp2x + SP_W + SP_GAP;
    const spY = yTop + SP_Y_OFF;

    const attnTid = byId.has(`blk.${L}.attn_wo_wv`) ? `blk.${L}.attn_wo_wv` : `blk.${L}.attn_q`;
    const attnLabel = byId.has(`blk.${L}.attn_wo_wv`) ? "attn (W_o·W_v)" : "attn (W_q)";
    makeSparkline(g, sp1x, spY, SP_W, SP_H, spectraOf(attnTid), attnLabel);

    makeSparkline(g, sp2x, spY, SP_W, SP_H, spectraOf(`blk.${L}.ffn_down_up`),
                  "ffn dense (W_down·W_up)");

    // Router spectrum is shorter (K up to 64 but the matrix is only 128 wide)
    makeSparkline(g, sp3x, spY, SP_W * 0.6, SP_H, spectraOf(`blk.${L}.ffn_gate_inp`),
                  "moe router");
  }

  // Footer — output norm + lm_head + softcap
  {
    const yTop = HEAD_H + N * ROW_H + 14;
    const g = el("g", { transform: `translate(${BLOCK_X},${yTop})` }, svg);
    el("text", { x: 0, y: 14, class: "panel-title" }, g).textContent =
      "RMSNorm  ·  lm_head (tied)  ·  softcap = 30 · tanh(x/30)";
    el("text", { x: 0, y: 30, class: "panel-sub" }, g).textContent =
      `${arch.model.vocab_size.toLocaleString()} logits per token`;
    // continuation of rail
    el("line", {
      x1: RAIL_X - BLOCK_X, y1: -16,
      x2: RAIL_X - BLOCK_X, y2: 0,
      class: "tee",
    }, g);
  }
}


// ---------- Layout B: small multiples ribbon ----------------------------

function renderRibbon(arch, byId, spectraOf, makeSparkline) {
  const svg = document.getElementById("ribbon");
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const N = arch.model.n_layer;
  const W = 1200;

  // Geometry
  const ROW_H = 28;
  const HEAD_H = 28;
  const LEFT_W = 110;          // label column
  const COL_GAP = 18;
  const COL_W = (W - LEFT_W - COL_GAP * 3 - 16) / 3;
  const SP_H = ROW_H - 6;

  const H = HEAD_H + N * ROW_H + 12;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");

  // Column headers
  const colXs = [
    LEFT_W,
    LEFT_W + COL_W + COL_GAP,
    LEFT_W + (COL_W + COL_GAP) * 2,
  ];
  const colTitles = ["attention", "ffn dense", "moe router"];
  for (let c = 0; c < 3; c++) {
    el("text", { x: colXs[c], y: HEAD_H - 10, class: "spark-label" }, svg)
      .textContent = colTitles[c];
  }
  // y-axis hint at far right
  el("text", { x: W - 8, y: HEAD_H - 10, "text-anchor": "end", class: "spark-label" }, svg)
    .textContent = "log σ (shared)";

  // One row per layer
  for (let L = 0; L < N; L++) {
    const layer = arch.layers[L];
    const yTop = HEAD_H + L * ROW_H;
    const yMid = yTop + ROW_H / 2;
    const isSwa = layer.is_swa;

    const g = el("g", { class: "block", "data-layer": String(L) }, svg);

    // Hover background, full row
    el("rect", {
      x: 0, y: yTop, width: W, height: ROW_H,
      class: "block-bg", rx: 0,
    }, g);

    // Color tag at far left
    el("rect", {
      x: 0, y: yTop + 3, width: 3, height: ROW_H - 6,
      class: `block-mark ${isSwa ? "swa" : "global"}`,
    }, g);

    // Layer label
    el("text", { x: 10, y: yMid + 4, class: "block-label" }, g).textContent =
      `blk.${L.toString().padStart(2, "0")}`;
    el("text", { x: 56, y: yMid + 4, class: "block-sub" }, g).textContent =
      isSwa ? "SWA" : "GLBL";

    // Three sparklines, no per-cell labels — column header is enough
    const attnTid = byId.has(`blk.${L}.attn_wo_wv`) ? `blk.${L}.attn_wo_wv` : `blk.${L}.attn_q`;
    makeSparkline(g, colXs[0], yTop + 3, COL_W, SP_H, spectraOf(attnTid), null);
    makeSparkline(g, colXs[1], yTop + 3, COL_W, SP_H, spectraOf(`blk.${L}.ffn_down_up`), null);
    makeSparkline(g, colXs[2], yTop + 3, COL_W * 0.6, SP_H, spectraOf(`blk.${L}.ffn_gate_inp`), null);
  }
}


// ---------- tooltip -----------------------------------------------------

function setupTooltip(arch) {
  const tt = document.getElementById("tooltip");

  function show(L, evt) {
    const layer = arch.layers[L];
    const tensors = collectTensors(layer);
    const totalParams = tensors.reduce((s, [, t]) => s + (t ? t.params : 0), 0);

    let html = `<div class="tt-head">blk.${L}  ${layer.is_swa ? "SWA" : "GLOBAL"}</div>`;
    html += `<div class="tt-line">heads <strong>${layer.n_head}</strong> · kv heads <strong>${layer.n_head_kv}</strong> · head_dim <strong>${layer.head_dim}</strong></div>`;
    html += `<div class="tt-line">RoPE base <strong>${layer.rope_base.toExponential(0)}</strong></div>`;
    html += `<div class="tt-line">params <strong>${fmtParams(totalParams)}</strong></div>`;
    html += `<table class="tt-tensors">`;
    for (const [role, t] of tensors) {
      if (!t) continue;
      html += `<tr><td class="role">${role}</td><td class="shape">${t.shape.join("×")}</td><td class="type">${t.type}</td><td class="params">${fmtParams(t.params)}</td></tr>`;
    }
    html += `</table>`;
    tt.innerHTML = html;
    tt.hidden = false;
    place(evt);
  }

  function place(evt) {
    const PAD = 14;
    const r = tt.getBoundingClientRect();
    let x = evt.clientX + PAD;
    let y = evt.clientY + PAD;
    if (x + r.width > window.innerWidth - 8) x = evt.clientX - r.width - PAD;
    if (y + r.height > window.innerHeight - 8) y = evt.clientY - r.height - PAD;
    tt.style.left = x + "px";
    tt.style.top  = y + "px";
  }

  function hide() { tt.hidden = true; }

  function bind(svgId) {
    const svg = document.getElementById(svgId);
    svg.addEventListener("mousemove", (evt) => {
      const block = evt.target.closest("g.block");
      if (!block) { hide(); return; }
      const L = parseInt(block.dataset.layer, 10);
      show(L, evt);
    });
    svg.addEventListener("mouseleave", hide);
  }
  bind("flow");
  bind("ribbon");
}

function collectTensors(layer) {
  // Flatten the architecture.json layer record into [role, tensor] pairs
  // for the tooltip table. Skip null entries.
  const out = [];
  const groups = [
    ["attn", layer.attn],
    ["ffn_dense", layer.ffn_dense],
    ["ffn_moe", layer.ffn_moe],
  ];
  for (const [prefix, sub] of groups) {
    if (!sub) continue;
    for (const [role, t] of Object.entries(sub)) {
      if (t) out.push([`${prefix}.${role}`, t]);
    }
  }
  if (layer.post_norm) out.push(["block.post_norm", layer.post_norm]);
  if (layer.out_scale) out.push(["block.out_scale", layer.out_scale]);
  return out;
}
