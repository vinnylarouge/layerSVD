// eval-capture: a llama.cpp tool that runs a prompt + greedy generation
// while dumping selected residual-stream tensors to disk per pass.
//
// Built as part of the LayerSVD project. Source lives outside the cloned
// llama.cpp tree; build.sh copies it into examples/eval-capture/ and adds
// it to the cmake build.
//
// Usage:
//   eval-capture -m <gguf> -p <prompt> -n <n_predict> --capture-out <dir>
//                [--capture-name <regex> ...]
//
// Output layout:
//   <dir>/manifest.json     -- prompt, tokens, captured tensor list
//   <dir>/pass_<NNNN>/<sanitized_name>.bin   -- raw fp16 tensor data
//
// Each .bin file is a flat fp16 buffer; the tensor's shape is in
// manifest.json. Pass index 0 = prompt processing pass; subsequent
// passes are one generated token each.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "llama-cpp.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <regex>
#include <string>
#include <sys/stat.h>
#include <vector>

struct capture_data {
    std::string out_dir;
    int pass_idx = 0;
    std::vector<std::regex> name_filters;  // empty -> capture nothing
    std::vector<uint8_t> tmp;              // host scratch buffer
    // For manifest:
    std::vector<std::string> captured_names_in_pass0;
    std::vector<std::vector<int64_t>> captured_shapes_in_pass0;
};

static void mkdirs(const std::string & path) {
    // create directories recursively, ignoring EEXIST
    std::string cur;
    for (size_t i = 0; i < path.size(); i++) {
        cur += path[i];
        if (path[i] == '/' && i > 0) {
            mkdir(cur.c_str(), 0755);
        }
    }
    mkdir(path.c_str(), 0755);
}

static std::string sanitize_name(const std::string & name) {
    std::string out;
    out.reserve(name.size());
    for (char c : name) {
        if (isalnum((unsigned char)c) || c == '_' || c == '-' || c == '.') {
            out += c;
        } else if (c == ' ') {
            out += '_';
        } else {
            out += '_';
        }
    }
    return out;
}

static uint16_t fp32_to_fp16(float f) {
    // IEEE 754 binary16 round-to-nearest. Compact and good enough for capture.
    union { float f; uint32_t u; } v;
    v.f = f;
    uint32_t x = v.u;
    uint32_t sign = (x >> 31) & 0x1;
    int32_t  exp  = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    uint32_t mant = x & 0x7FFFFF;

    uint16_t h_sign = (uint16_t)(sign << 15);
    if (exp <= 0) {
        if (exp < -10) return h_sign;  // too small -> 0
        // subnormal
        mant |= 0x800000;
        uint32_t shift = 14 - exp;
        uint16_t h_mant = (uint16_t)((mant + (1u << (shift - 1))) >> shift);
        return h_sign | h_mant;
    }
    if (exp >= 31) {
        if (mant == 0) return h_sign | 0x7C00;  // ±inf
        return h_sign | 0x7C00 | (uint16_t)(mant >> 13);  // NaN
    }
    uint16_t h_exp = (uint16_t)(exp << 10);
    uint16_t h_mant = (uint16_t)((mant + 0x1000) >> 13);
    if (h_mant & 0x400) {  // mantissa overflow -> bump exponent
        h_mant = 0;
        h_exp = (uint16_t)((exp + 1) << 10);
    }
    return h_sign | h_exp | h_mant;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * cb = (capture_data *) user_data;
    if (ask) {
        // Tell the scheduler we want data; we'll filter on the actual call.
        return true;
    }
    if (cb->name_filters.empty()) return true;

    // Match against any filter
    bool matched = false;
    for (const auto & rx : cb->name_filters) {
        if (std::regex_search(t->name, rx)) { matched = true; break; }
    }
    if (!matched) return true;

    // Quantized intermediates are unusual; skip them
    if (ggml_is_quantized(t->type)) return true;
    if (t->type != GGML_TYPE_F32 && t->type != GGML_TYPE_F16 && t->type != GGML_TYPE_BF16) {
        return true;
    }

    // Pull bytes to host
    const size_t nbytes = ggml_nbytes(t);
    cb->tmp.resize(nbytes);
    const bool host = ggml_backend_buffer_is_host(t->buffer);
    const uint8_t * src;
    if (host) {
        src = (const uint8_t *) t->data;
    } else {
        ggml_backend_tensor_get(t, cb->tmp.data(), 0, nbytes);
        src = cb->tmp.data();
    }

    // Convert to a contiguous fp32 vector first, honoring strides.
    const int64_t ne0 = t->ne[0];
    const int64_t ne1 = t->ne[1];
    const int64_t ne2 = t->ne[2];
    const int64_t ne3 = t->ne[3];
    const size_t total_elems = (size_t) ne0 * ne1 * ne2 * ne3;
    std::vector<float> f32(total_elems);

    auto load = [&](size_t i0, size_t i1, size_t i2, size_t i3) -> float {
        size_t off = i0 * t->nb[0] + i1 * t->nb[1] + i2 * t->nb[2] + i3 * t->nb[3];
        if (t->type == GGML_TYPE_F32) {
            return *(const float *)(src + off);
        } else if (t->type == GGML_TYPE_F16) {
            return ggml_fp16_to_fp32(*(const ggml_fp16_t *)(src + off));
        } else if (t->type == GGML_TYPE_BF16) {
            return ggml_bf16_to_fp32(*(const ggml_bf16_t *)(src + off));
        }
        return 0.0f;
    };

    size_t k = 0;
    for (int64_t i3 = 0; i3 < ne3; i3++)
    for (int64_t i2 = 0; i2 < ne2; i2++)
    for (int64_t i1 = 0; i1 < ne1; i1++)
    for (int64_t i0 = 0; i0 < ne0; i0++) {
        f32[k++] = load((size_t)i0, (size_t)i1, (size_t)i2, (size_t)i3);
    }

    // Convert to fp16 and write
    std::vector<uint16_t> half(total_elems);
    for (size_t i = 0; i < total_elems; i++) half[i] = fp32_to_fp16(f32[i]);

    char pass_dir[64];
    snprintf(pass_dir, sizeof(pass_dir), "%s/pass_%04d", cb->out_dir.c_str(), cb->pass_idx);
    mkdirs(pass_dir);
    std::string sname = sanitize_name(t->name);
    std::string fpath = std::string(pass_dir) + "/" + sname + ".bin";
    std::ofstream f(fpath, std::ios::binary);
    f.write((const char *) half.data(), half.size() * sizeof(uint16_t));
    f.close();

    if (cb->pass_idx == 0) {
        cb->captured_names_in_pass0.push_back(t->name);
        cb->captured_shapes_in_pass0.push_back({ne0, ne1, ne2, ne3});
    }
    return true;
}

static void write_manifest(
    const std::string & out_dir,
    const std::string & prompt,
    const std::vector<llama_token> & prompt_tokens,
    const std::vector<llama_token> & generated_tokens,
    const std::vector<std::string> & sampled_strings,
    int n_passes,
    const std::vector<std::string> & names,
    const std::vector<std::vector<int64_t>> & shapes)
{
    auto esc = [](const std::string & s) {
        std::string out;
        for (char c : s) {
            if (c == '"') { out += "\\\""; }
            else if (c == '\\') { out += "\\\\"; }
            else if (c == '\n') { out += "\\n"; }
            else if (c == '\r') { out += "\\r"; }
            else if (c == '\t') { out += "\\t"; }
            else if ((unsigned char)c < 0x20) {
                char buf[8];
                snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += c;
            }
        }
        return out;
    };

    std::ofstream f(out_dir + "/manifest.json");
    f << "{\n";
    f << "  \"version\": 1,\n";
    f << "  \"prompt\": \"" << esc(prompt) << "\",\n";
    f << "  \"prompt_tokens\": [";
    for (size_t i = 0; i < prompt_tokens.size(); i++) {
        if (i) f << ", ";
        f << prompt_tokens[i];
    }
    f << "],\n";
    f << "  \"generated_tokens\": [";
    for (size_t i = 0; i < generated_tokens.size(); i++) {
        if (i) f << ", ";
        f << generated_tokens[i];
    }
    f << "],\n";
    f << "  \"generated_strings\": [";
    for (size_t i = 0; i < sampled_strings.size(); i++) {
        if (i) f << ", ";
        f << "\"" << esc(sampled_strings[i]) << "\"";
    }
    f << "],\n";
    f << "  \"n_passes\": " << n_passes << ",\n";
    f << "  \"captured_tensors\": [\n";
    for (size_t i = 0; i < names.size(); i++) {
        f << "    {\"name\": \"" << esc(names[i]) << "\", \"sanitized\": \"" << sanitize_name(names[i])
          << "\", \"shape\": [";
        for (size_t j = 0; j < shapes[i].size(); j++) {
            if (j) f << ", ";
            f << shapes[i][j];
        }
        f << "]}";
        if (i + 1 < names.size()) f << ",";
        f << "\n";
    }
    f << "  ]\n";
    f << "}\n";
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    // Custom args (we parse them manually then strip them before delegating
    // to common_params_parse).
    std::string out_dir = "data/raw/exemplar";
    std::vector<std::string> filter_strs = {
        // Default: capture the layer-output residual stream and a few
        // adjacent named tensors that the gemma4 graph builder cb()s.
        "^l_out-",
        "^attn_out-",
        "^ffn_moe_combined-",
        "^ffn_moe_logits-",
        "^inp_scaled$",
        "^result_norm$",
        "^result_output$",
    };

    std::vector<char *> argv2;
    argv2.push_back(argv[0]);
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--capture-out" && i + 1 < argc) {
            out_dir = argv[++i];
        } else if (a == "--capture-name" && i + 1 < argc) {
            // First explicit --capture-name resets defaults
            static bool reset = false;
            if (!reset) { filter_strs.clear(); reset = true; }
            filter_strs.push_back(argv[++i]);
        } else {
            argv2.push_back(argv[i]);
        }
    }

    capture_data cb;
    cb.out_dir = out_dir;
    for (const auto & s : filter_strs) {
        cb.name_filters.emplace_back(s, std::regex::optimize);
    }
    mkdirs(out_dir);

    common_params params;
    common_init();
    if (!common_params_parse((int) argv2.size(), argv2.data(), params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    params.cb_eval = capture_cb;
    params.cb_eval_user_data = &cb;
    params.warmup = false;

    auto llama_init = common_init_from_params(params);
    auto * model = llama_init->model();
    auto * ctx = llama_init->context();
    if (!model || !ctx) {
        LOG_ERR("failed to init model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> prompt_tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (prompt_tokens.empty()) {
        LOG_ERR("no input tokens\n");
        return 1;
    }
    LOG_INF("prompt tokens = %zu\n", prompt_tokens.size());

    // -- Pass 0: process the prompt --
    cb.pass_idx = 0;
    if (llama_decode(ctx, llama_batch_get_one(prompt_tokens.data(), (int32_t) prompt_tokens.size()))) {
        LOG_ERR("decode prompt failed\n");
        return 1;
    }

    // Greedy decode for n_predict tokens
    int n_predict = params.n_predict > 0 ? params.n_predict : 32;
    std::vector<llama_token> generated;
    std::vector<std::string> generated_strs;

    int n_past = (int) prompt_tokens.size();
    for (int step = 0; step < n_predict; step++) {
        // Sample greedy: pick argmax of last token's logits
        const float * logits = llama_get_logits_ith(ctx, -1);
        int n_vocab = llama_vocab_n_tokens(vocab);
        int best = 0;
        float best_v = logits[0];
        for (int i = 1; i < n_vocab; i++) {
            if (logits[i] > best_v) { best_v = logits[i]; best = i; }
        }
        llama_token tok = (llama_token) best;
        if (llama_vocab_is_eog(vocab, tok)) break;

        generated.push_back(tok);
        char buf[256] = {0};
        int n = llama_token_to_piece(vocab, tok, buf, sizeof(buf), 0, false);
        if (n > 0) generated_strs.emplace_back(buf, n); else generated_strs.emplace_back("");

        cb.pass_idx = step + 1;
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            LOG_ERR("decode step failed\n");
            break;
        }
        n_past += 1;
    }

    write_manifest(out_dir, params.prompt, prompt_tokens, generated, generated_strs,
                   1 + (int) generated.size(),
                   cb.captured_names_in_pass0,
                   cb.captured_shapes_in_pass0);

    LOG("\n");
    llama_perf_context_print(ctx);
    llama_backend_free();
    return 0;
}
