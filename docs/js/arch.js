// Architecture map view.
//
// Renders a card per transformer block, plus a summary panel and a
// "global tensors" section. Each card expands to show the per-tensor
// shape/dtype/param count.

(async function () {
  const data = await loadArchitecture();

  // -- model summary cards ----------------------------------------------
  const m = data.model;
  const summaryCards = [
    ["arch", m.architecture + " " + m.size_label],
    ["layers", m.n_layer + ""],
    ["d_model", m.n_embd + ""],
    ["q heads", m.n_head + ""],
    ["vocab", m.vocab_size.toLocaleString()],
    ["context", m.context_length.toLocaleString()],
    ["experts", `${m.n_expert} (top ${m.n_expert_used})`],
    ["expert d_ffn", m.n_ff_exp + ""],
    ["dense d_ffn", m.n_ff_dense + ""],
    ["RoPE base", m.rope_base.toExponential(0)],
    ["RoPE base SWA", m.rope_base_swa.toExponential(0)],
    ["softcap", m.final_logit_softcap + ""],
    ["params", fmtParams(m.total_params)],
    ["tensors", m.n_tensors + ""],
  ];
  const summaryEl = document.getElementById("model-summary");
  const grid = document.createElement("div");
  grid.className = "summary";
  for (const [label, val] of summaryCards) {
    const c = document.createElement("div");
    c.className = "card";
    const l = document.createElement("div");
    l.className = "label";
    l.textContent = label;
    const v = document.createElement("div");
    v.className = "value";
    v.textContent = val;
    c.appendChild(l);
    c.appendChild(v);
    grid.appendChild(c);
  }
  summaryEl.appendChild(grid);

  // -- per-layer cards ---------------------------------------------------
  const layersEl = document.getElementById("layers");
  for (const layer of data.layers) {
    const card = document.createElement("div");
    card.className = "layer-card " + (layer.is_swa ? "swa" : "global");

    const head = document.createElement("div");
    head.className = "head";
    const num = document.createElement("div");
    num.className = "num";
    num.textContent = `blk.${layer.index}`;
    const tags = document.createElement("div");
    tags.className = "tags";
    tags.innerHTML = `
      <span class="tag ${layer.is_swa ? "swa" : "global"}">${layer.is_swa ? "SWA" : "GLOBAL"}</span>
      ${layer.is_moe ? '<span class="tag moe">MoE</span>' : ""}
    `;
    head.appendChild(num);
    head.appendChild(tags);

    const summary = document.createElement("div");
    summary.className = "summary-row";
    summary.textContent = `q×${layer.n_head}, kv×${layer.n_head_kv}, hd=${layer.head_dim}`;

    const detail = document.createElement("div");
    detail.className = "layer-detail";
    detail.appendChild(buildSubmodule("attention", layer.attn, "dense"));
    detail.appendChild(buildSubmodule("dense FFN", layer.ffn_dense, "dense"));
    if (layer.is_moe) {
      detail.appendChild(buildSubmodule("MoE FFN (parallel branch)", layer.ffn_moe, "moe"));
    }
    if (layer.post_norm || layer.out_scale) {
      const post = {
        post_ffw_norm: layer.post_norm,
        layer_output_scale: layer.out_scale,
      };
      detail.appendChild(buildSubmodule("block residual", post, "dense"));
    }

    card.appendChild(head);
    card.appendChild(summary);
    card.appendChild(detail);

    card.addEventListener("click", (e) => {
      // Don't toggle when clicking inside the detail (so e.g. text is selectable)
      if (e.target.closest(".layer-detail") && card.classList.contains("expanded")) return;
      card.classList.toggle("expanded");
    });

    layersEl.appendChild(card);
  }

  // -- global tensors ---------------------------------------------------
  const globalsEl = document.getElementById("globals");
  const gcard = document.createElement("div");
  gcard.className = "layer-card";
  gcard.classList.add("expanded");
  const ghead = document.createElement("div");
  ghead.className = "head";
  ghead.innerHTML = `<div class="num">model globals</div>`;
  const gdetail = document.createElement("div");
  gdetail.className = "layer-detail";
  gdetail.appendChild(buildSubmodule("globals", data.globals, "dense"));
  gcard.appendChild(ghead);
  gcard.appendChild(gdetail);
  globalsEl.appendChild(gcard);

  function buildSubmodule(title, tensors, kind) {
    const sub = document.createElement("div");
    sub.className = "submodule";
    const h = document.createElement("h3");
    h.innerHTML = `${title} <span class="pill ${kind}">${kind === "moe" ? "moe" : "dense"}</span>`;
    sub.appendChild(h);

    const table = document.createElement("table");
    table.className = "tensor-table";
    const thead = document.createElement("thead");
    thead.innerHTML =
      "<tr><th>role</th><th>tensor</th><th>shape</th><th>type</th><th>params</th></tr>";
    table.appendChild(thead);
    const tbody = document.createElement("tbody");

    for (const [role, t] of Object.entries(tensors)) {
      if (!t) continue;
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="role">${role}</td>
        <td>${t.name}</td>
        <td class="shape">${t.shape.join("×")}</td>
        <td class="type">${t.type}</td>
        <td class="params">${fmtParams(t.params)}</td>
      `;
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    sub.appendChild(table);
    return sub;
  }
})().catch((err) => {
  console.error(err);
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<pre style="color:#ff6f6f;padding:20px;">${err.stack || err.message}</pre>`
  );
});
