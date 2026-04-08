# layerSVD

A static, GitHub-Pages-hosted explorer for the **Gemma 4 26B-A4B (Q4_K_M)**
weights and inference traces.

It lets you:

1. **Navigate the architecture** — every block, every tensor, every shape and
   quant type, with the ISWA / dual-FFN / 128-expert MoE structure exposed.
2. **Browse SVD bases** — for every linear weight (and a few composed maps:
   `W_o @ W_v` per attention block, `W_down @ W_up` per dense FFN), pick a
   layer and a tensor and inspect its singular value spectrum, top-K right
   singular vectors as a heatmap, and a paired cross-layer flow heatmap
   (`U_L^T V_{L+1}` of the composed dense FFN).
3. **Trace concepts during exemplar generations** — for three small prompts
   (arithmetic, narrative, code) we capture the residual stream at every
   layer for every token, project it onto the SVD bases, and let you scrub
   tokens and label concepts.

The published site is at
**<https://vinnylarouge.github.io/layerSVD/>**.

## Why this exists

SVD of weight matrices won't give you ground-truth interpretable features
the way a sparse autoencoder does, but it's a useful first sketch: the
right singular vectors of any projection are "what the next layer listens
for", and the left singular vectors are "what the producing layer can
say". Pairing those across adjacent layers gives an honest, basis-aligned
notion of cross-layer concept flow that doesn't smuggle in rotation noise.

The site is intentionally a labelling surface: every concept slot at every
layer has a textbox, persisted in `localStorage`, and a "download
labels.json" button so you can PR human labels back.

## Architecture (read directly from the GGUF)

Gemma 4 26B-A4B is a 30-block transformer with several unusual design choices:

- **Interleaved sliding-window / global attention.** 5 SWA layers + 1 global,
  repeating × 5. SWA: 16 q-heads × 256 head-dim, 8 KV heads, RoPE base 10K.
  Global: 16 q-heads × 512 head-dim, 2 KV heads, RoPE base 1M.
- **Per-head Q/K RMS norms** and a (non-learnable) RMS norm on V.
- **Each block has BOTH a dense FFN and a 128-expert MoE FFN, run in parallel
  and summed.** The dense MLP and the MoE branch each have their own
  pre/post norms; the MoE router reads the un-normed `attn_out` directly.
  Top-8 of 128 experts; per-expert fused gate+up of width 1408 = 2 × 704.
- **Final logit softcap** of 30 (`tanh(x/30) * 30`).
- **No `tok_embd_per_layer`** in the file we have (the metadata flag exists
  but is set to 0).

This is verified against `llama.cpp/src/models/gemma4-iswa.cpp`'s graph
builder, which is the only existing implementation of this arch.

## Layout

```
layersvd/                         python package (no pip install needed)
  gguf_reader.py                  GGUF v3 metadata + tensor index parser
  dequant.py                      pure-numpy Q4_K / Q6_K / Q5_0 / Q8_0 dequant
  dequant_test.py                 gating: bit-for-bit vs gguf python ref
  build_arch.py                   emits docs/data/architecture.json
  svd_pipeline.py                 truncated SVD over every linear weight
  chunk_svd.py                    splits U.f16/V.f16 into per-layer chunks
                                  (one-time post-process; required because
                                  GitHub blocks single files >100 MB)
  project.py                      projects captured residuals onto SVD bases

tools/eval-capture/               custom llama.cpp tool that runs inference
  eval-capture.cpp                  and dumps named tensors per token
  CMakeLists.txt
  build.sh                        copies into llama.cpp/examples and builds

tools/llama.cpp/                  cloned upstream (gitignored)
                                  build via tools/eval-capture/build.sh
                                  or `cmake --build build`

model/                            16 GB Q4_K_M GGUF (gitignored, kept local)
data/raw/                         captured per-token tensor dumps (gitignored)

docs/                             github pages root (everything below is shipped)
  index.html  svd.html  trace.html
  css/style.css
  js/d3.v7.min.js  js/data_loader.js  js/arch.js  js/svd.js  js/trace.js
  data/architecture.json          ~100 KB
  data/svd/manifest.json          ~150 KB
  data/svd/spectra.bin            ~75 KB
  data/svd/U-L+NN.f16             per-layer left singular vector chunks
  data/svd/V-L+NN.f16             per-layer right singular vector chunks
  data/traces/index.json
  data/traces/{arithmetic,narrative,code}.json
```

Total payload pushed to GitHub Pages: ~210 MB. SVD bases dominate.

## Reproducing locally

You will need:

- macOS (or Linux) with cmake and a C++17 compiler. Apple Silicon strongly
  recommended; the published bundle was generated on an M4 Max with 128 GB.
- A copy of `gemma-4-26B-A4B-it-Q4_K_M.gguf` placed at `model/`.
- Python 3.11+ and [`uv`](https://github.com/astral-sh/uv).

```bash
# 1. python env
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python numpy scipy tqdm jinja2 gguf

# 2. clone llama.cpp (gemma4 needs a recent master with src/models/gemma4-iswa.cpp)
git clone https://github.com/ggerganov/llama.cpp tools/llama.cpp
cd tools/llama.cpp
cmake -B build -DLLAMA_METAL=ON -DGGML_METAL=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j 12 --target llama-cli llama-eval-callback
cd ../..

# 3. validate dequantizers (gating)
.venv/bin/python -m layersvd.dequant_test model/gemma-4-26B-A4B-it-Q4_K_M.gguf
# expect: [OK] for F32, Q5_0, Q8_0, Q4_K, Q6_K

# 4. emit architecture.json
.venv/bin/python -m layersvd.build_arch model/gemma-4-26B-A4B-it-Q4_K_M.gguf

# 5. compute SVDs (~12 minutes on M4 Max, ~210 MB output)
.venv/bin/python -m layersvd.svd_pipeline model/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    --out docs/data/svd --skip-experts

# 6. chunk SVD into per-layer files (required: GitHub blocks files >100 MB)
.venv/bin/python -m layersvd.chunk_svd docs/data/svd

# 7. build the eval-capture tool
tools/eval-capture/build.sh

# 8. capture exemplar generations + project onto SVD bases
for ex in arithmetic narrative code; do
  PROMPT_FILE=prompts/$ex.txt
  tools/llama.cpp/build/bin/eval-capture \
    -m model/gemma-4-26B-A4B-it-Q4_K_M.gguf \
    -p "$(cat $PROMPT_FILE)" -n 32 -ngl 99 \
    --capture-out data/raw/$ex
  .venv/bin/python -m layersvd.project \
    --raw-dir data/raw/$ex \
    --out docs/data/traces/$ex.json
done

# 9. preview locally
.venv/bin/python -m http.server -d docs 8765
# open http://localhost:8765/
```

## Caveats

- **Q4_K_M quantization noise** is baked into every singular vector.
  Top-K SVD on dequantized Q4 is meaningful (the dequant is bit-perfect
  to llama.cpp's reference), but sub-σ-floor structure is dominated by
  quant noise. For sharper bases re-run from the BF16 originals.
- **Per-expert SVDs are not yet shipped.** The 128 experts × 30 layers
  × 2 matrices is ~3 GB even at K=8, which doesn't fit on Pages without
  more aggressive chunking. The router and the per-block dense FFN
  carry the v0 story.
- **The "concept" abstraction is basis-only.** SVD singular vectors are
  not the same as SAE features. The right singular vectors of layer
  L+1's input projections are an honest answer to "what does this layer
  *attend to*" but they don't carve nature at semantic joints.
- **The trace covers 1 last-prompt token + 32 generated tokens per
  exemplar.** The prompt-pass shape optimization in llama.cpp drops
  unused intermediate tokens at the last layer, which is why we
  standardize on the last prompt token only.

## License

Code: MIT.

The Gemma 4 weights themselves are not redistributed here — they live in
`model/` only on the build machine.
