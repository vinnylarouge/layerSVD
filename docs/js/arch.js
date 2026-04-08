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

  // Spectra fetcher: returns the raw singular values for a tensor.
  function spectraOf(tid) {
    const e = byId.get(tid);
    if (!e) return null;
    const start = e.s_offset / 4;
    return spectraF32.subarray(start, start + e.K);
  }

  // Cumulative variance: cv[i] = (sum_{j<=i} sigma_j^2) / fro_total^2.
  // The end of the curve is fro_kept^2 / fro_total^2 = fraction of the
  // total Frobenius energy captured by the top-K SVD. Both axes share
  // the same [0, 1] range across every tensor on the page, so the
  // *shape* of the rise is what reads first: a curve that hits 0.9 in
  // the first few indices is rank-collapsed, a curve that climbs slowly
  // is diffuse.
  const cumvarCache = new Map();
  function cumvarOf(tid) {
    if (cumvarCache.has(tid)) return cumvarCache.get(tid);
    const e = byId.get(tid);
    if (!e) return null;
    const sig = spectraOf(tid);
    const total = e.fro_total;
    if (!sig || !total || total <= 0) return null;
    const t2 = total * total;
    const out = new Float64Array(sig.length);
    let acc = 0;
    for (let i = 0; i < sig.length; i++) {
      acc += sig[i] * sig[i];
      out[i] = acc / t2;
    }
    cumvarCache.set(tid, out);
    return out;
  }

  // Sparkline drawing primitive.
  // ``cv`` is a [0..1] cumulative-variance curve. ``isGlobal`` selects the
  // colour family. ``label`` (optional) is rendered above the curve.
  function makeSparkline(parent, x, y, w, h, cv, label, isGlobal) {
    if (!cv || cv.length === 0) {
      el("text", { x: x + w / 2, y: y + h / 2 + 3, "text-anchor": "middle",
                   class: "spark-label" }, parent).textContent = "—";
      return;
    }
    // Top reference line (1.0 = 100% captured). Solid faint, the
    // baseline (0%) is implied by the bottom of the cell.
    el("line", { x1: x, y1: y, x2: x + w, y2: y, class: "spark-base dotted" }, parent);
    el("line", { x1: x, y1: y + h, x2: x + w, y2: y + h, class: "spark-base" }, parent);

    const n = cv.length;
    const xs = (i) => x + (i / (n - 1)) * w;
    const ys = (v) => y + h - Math.max(0, Math.min(1, v)) * h;

    // Build the line path
    let d = "";
    for (let i = 0; i < n; i++) {
      d += (i === 0 ? "M" : "L") + xs(i).toFixed(1) + " " + ys(cv[i]).toFixed(1);
    }
    // Build a closed area path for the fill underneath
    let dFill = "M" + x.toFixed(1) + " " + (y + h).toFixed(1);
    for (let i = 0; i < n; i++) {
      dFill += "L" + xs(i).toFixed(1) + " " + ys(cv[i]).toFixed(1);
    }
    dFill += "L" + (x + w).toFixed(1) + " " + (y + h).toFixed(1) + "Z";

    const cls = isGlobal ? "global" : "";
    el("path", { d: dFill, class: `spark-fill ${cls}` }, parent);
    el("path", { d, class: `spark-line ${cls}` }, parent);

    // Final-value tick + numeric annotation
    const finalY = ys(cv[n - 1]);
    el("circle", { cx: x + w, cy: finalY, r: 1.5, class: "spark-final-dot" }, parent);
    el("text", { x: x + w + 4, y: finalY + 3, class: "spark-final-text" }, parent)
      .textContent = (cv[n - 1] * 100).toFixed(0) + "%";

    if (label) {
      el("text", { x, y: y - 3, class: "spark-label" }, parent).textContent = label;
    }
  }

  // -------- one-line model summary --------
  renderModelLine(arch);

  // -------- Layout A: flowchart with residual rail --------
  renderFlow(arch, byId, cumvarOf, makeSparkline);

  // -------- Layout B: small multiples ribbon --------
  renderRibbon(arch, byId, cumvarOf, makeSparkline);

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

function renderFlow(arch, byId, cumvarOf, makeSparkline) {
  const svg = document.getElementById("flow");
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const W = 1200;
  const HEAD_H = 130;
  const FOOT_H = 100;
  const ROW_H = 102;
  const N = arch.model.n_layer;
  const H = HEAD_H + N * ROW_H + FOOT_H;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  // Lock aspect ratio so the SVG always renders at the right intrinsic
  // height regardless of how the host browser interprets viewBox+height:auto.
  svg.style.aspectRatio = `${W} / ${H}`;

  // Geometry
  const RAIL_X = 84;           // residual rail center
  const BLOCK_X = 140;
  const BLOCK_W = W - BLOCK_X - 24;
  const SP_W = 220;            // sparkline width
  const SP_H = 60;             // sparkline height (was 28)
  const SP_GAP = 38;
  const SP_Y_OFF = 28;         // sparkline top offset within a block

  // Header — token embedding panel + initial scaling note
  {
    const g = el("g", { transform: `translate(${BLOCK_X},14)` }, svg);
    el("text", { x: 0, y: 14, class: "panel-title" }, g).textContent =
      `token_embd  ${arch.globals.token_embd.shape.join("×")}  ${arch.globals.token_embd.type}`;
    el("text", { x: 0, y: 30, class: "panel-sub" }, g).textContent =
      `× √d_model · tied with lm_head · vocab ${arch.model.vocab_size.toLocaleString()}`;
    // Vocab spectrum sparkline (token_embd has K=32)
    const sx = BLOCK_W - SP_W - 56;
    const sy = 14;
    el("text", { x: sx, y: sy + 6, class: "spark-label" }, g).textContent = "vocab cumvar";
    makeSparkline(g, sx, sy + 14, SP_W, SP_H, cumvarOf("token_embd"), null, false);
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
    const isSwa = layer.is_swa;

    // Two tee strokes from the rail into the block:
    //   upper tee = attention residual add (top half of the block)
    //   lower tee = FFN residual add (bottom half of the block)
    el("line", {
      x1: RAIL_X, y1: yTop + 24, x2: BLOCK_X, y2: yTop + 24, class: "tee",
    }, svg);
    el("line", {
      x1: RAIL_X, y1: yTop + ROW_H - 24, x2: BLOCK_X, y2: yTop + ROW_H - 24, class: "tee",
    }, svg);

    // Block group (whole row, hoverable)
    const g = el("g", { class: "block", "data-layer": String(L) }, svg);

    // Background (transparent by default; CSS hover brings it in)
    el("rect", {
      x: BLOCK_X, y: yTop + 6, width: BLOCK_W, height: ROW_H - 12,
      class: "block-bg", rx: 2,
    }, g);

    // Color marker on the left edge of the block (one bit: SWA vs global)
    el("rect", {
      x: BLOCK_X, y: yTop + 6, width: 4, height: ROW_H - 12,
      class: `block-mark ${isSwa ? "swa" : "global"}`,
    }, g);

    // Block label (left side)
    const labelX = BLOCK_X + 14;
    el("text", { x: labelX, y: yTop + 22, class: "block-label" }, g).textContent =
      `blk.${L.toString().padStart(2, "0")}`;
    el("text", { x: labelX, y: yTop + 38, class: "block-sub" }, g).textContent =
      isSwa ? "SWA" : "GLOBAL";
    el("text", { x: labelX, y: yTop + 52, class: "block-sub" }, g).textContent =
      `q${layer.n_head} kv${layer.n_head_kv}`;
    el("text", { x: labelX, y: yTop + 66, class: "block-sub" }, g).textContent =
      `hd${layer.head_dim}`;

    // Three sparklines, right of the label
    const sp1x = BLOCK_X + 120;
    const sp2x = sp1x + SP_W + SP_GAP + 36;  // extra room for the % annotation
    const sp3x = sp2x + SP_W + SP_GAP + 36;
    const spY = yTop + SP_Y_OFF;

    const attnTid = byId.has(`blk.${L}.attn_wo_wv`) ? `blk.${L}.attn_wo_wv` : `blk.${L}.attn_q`;
    const attnLabel = byId.has(`blk.${L}.attn_wo_wv`) ? "attn (W_o·W_v)" : "attn (W_q)";
    makeSparkline(g, sp1x, spY, SP_W, SP_H, cumvarOf(attnTid), attnLabel, !isSwa);

    makeSparkline(g, sp2x, spY, SP_W, SP_H, cumvarOf(`blk.${L}.ffn_down_up`),
                  "ffn dense (W_down·W_up)", !isSwa);

    // Router has K capped at 128 -> shorter sparkline
    makeSparkline(g, sp3x, spY, SP_W * 0.55, SP_H, cumvarOf(`blk.${L}.ffn_gate_inp`),
                  "moe router (W_router)", !isSwa);
  }

  // Footer — output norm + lm_head + softcap
  {
    const yTop = HEAD_H + N * ROW_H + 24;
    const g = el("g", { transform: `translate(${BLOCK_X},${yTop})` }, svg);
    el("text", { x: 0, y: 14, class: "panel-title" }, g).textContent =
      "RMSNorm  ·  lm_head (tied)  ·  softcap = 30 · tanh(x/30)";
    el("text", { x: 0, y: 30, class: "panel-sub" }, g).textContent =
      `${arch.model.vocab_size.toLocaleString()} logits per token`;
  }
}


// ---------- Layout B: small multiples ribbon ----------------------------

function renderRibbon(arch, byId, cumvarOf, makeSparkline) {
  const svg = document.getElementById("ribbon");
  while (svg.firstChild) svg.removeChild(svg.firstChild);

  const N = arch.model.n_layer;
  const W = 1200;

  // Geometry
  const ROW_H = 50;
  const HEAD_H = 38;
  const LEFT_W = 110;          // label column
  const COL_GAP = 60;          // generous gap so % annotations fit
  const RIGHT_PAD = 16;
  const SP_H = ROW_H - 10;
  const COL_W = (W - LEFT_W - COL_GAP * 2 - RIGHT_PAD - 48) / 3;

  const H = HEAD_H + N * ROW_H + 18;
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);
  svg.setAttribute("preserveAspectRatio", "xMidYMin meet");
  svg.style.aspectRatio = `${W} / ${H}`;

  // Column headers
  const colXs = [
    LEFT_W,
    LEFT_W + COL_W + COL_GAP,
    LEFT_W + (COL_W + COL_GAP) * 2,
  ];
  const colTitles = ["attention (W_o·W_v / W_q)", "ffn dense (W_down·W_up)", "moe router"];
  for (let c = 0; c < 3; c++) {
    el("text", { x: colXs[c], y: HEAD_H - 14, class: "spark-label" }, svg)
      .textContent = colTitles[c];
  }
  // y-axis hint at far right
  el("text", { x: W - 8, y: HEAD_H - 14, "text-anchor": "end", class: "spark-label" }, svg)
    .textContent = "y: cumulative variance, 0→1 (shared)";

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
      x: 0, y: yTop + 4, width: 4, height: ROW_H - 8,
      class: `block-mark ${isSwa ? "swa" : "global"}`,
    }, g);

    // Layer label
    el("text", { x: 14, y: yMid + 4, class: "block-label" }, g).textContent =
      `blk.${L.toString().padStart(2, "0")}`;
    el("text", { x: 60, y: yMid + 4, class: "block-sub" }, g).textContent =
      isSwa ? "SWA" : "GLBL";

    // Three sparklines, no per-cell labels — column header is enough
    const attnTid = byId.has(`blk.${L}.attn_wo_wv`) ? `blk.${L}.attn_wo_wv` : `blk.${L}.attn_q`;
    makeSparkline(g, colXs[0], yTop + 5, COL_W, SP_H, cumvarOf(attnTid), null, !isSwa);
    makeSparkline(g, colXs[1], yTop + 5, COL_W, SP_H, cumvarOf(`blk.${L}.ffn_down_up`), null, !isSwa);
    makeSparkline(g, colXs[2], yTop + 5, COL_W * 0.55, SP_H, cumvarOf(`blk.${L}.ffn_gate_inp`), null, !isSwa);
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
