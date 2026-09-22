// route-trace: record the native MoE router's expert choices for complete
// prompt-response sequences.
//
// For every request the tool clears the KV cache, prefills the prompt as one
// ubatch, then generates one token per decode call, feeding every sampled token
// back (the final token included, EOG excluded). An eval callback reads
// `ffn_moe_topk-<il>` after it is computed, so the trace is the router's own
// selection, not a re-derivation. During prefill it also sums `ffn_moe_probs-<il>`
// over prompt tokens, which is the only router state a prompt-level predictor may
// use besides the prefill selections themselves.
//
// The tool never alters routing or compute: the callback only reads tensors.
// A decode call that does not deliver exactly one top-k tensor per MoE layer is
// a hard error, not a gap in the trace.
//
// Input: JSONL, one object per line with string fields "id" and "text" (text is
// already formatted, special tokens are parsed). Output: JSONL, one object per
// request with prompt/generated token ids and the route arrays.

#include "llama.h"
#include "ggml.h"
#include "ggml-backend.h"

#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct trace_state {
    int n_layer  = 0;
    int n_expert = 0;
    int k        = 0;
    bool prefill = false;
    // per layer, flattened [token][k]
    std::vector<std::vector<int32_t>> topk;
    // per layer, [n_expert] sum over prefill tokens of the router probabilities
    std::vector<std::vector<double>> prob_sum;
    std::vector<int> topk_calls;   // per layer, top-k tensors seen in the current decode call
    std::vector<int> prob_calls;
    std::vector<int> prob_rows;    // per layer, prefill tokens summed into prob_sum
    std::string error;
};

int layer_of(const char * name, const char * prefix) {
    const size_t n = strlen(prefix);
    if (strncmp(name, prefix, n) != 0) {
        return -1;
    }
    const char * p = name + n;
    // exact "<prefix><digits>" only: views such as "ffn_moe_probs-0 (reshaped)" are other nodes
    if (*p == '\0') {
        return -1;
    }
    for (const char * q = p; *q; ++q) {
        if (*q < '0' || *q > '9') {
            return -1;
        }
    }
    return atoi(p);
}

bool read_tensor_host(const ggml_tensor * t, std::vector<uint8_t> & buf, const uint8_t *& base) {
    if (t->buffer && ggml_backend_buffer_is_host(t->buffer)) {
        base = (const uint8_t *) t->data;
        return true;
    }
    buf.resize(ggml_nbytes(t));
    ggml_backend_tensor_get(t, buf.data(), 0, buf.size());
    base = buf.data();
    return true;
}

bool eval_cb(ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (trace_state *) user_data;
    const int il_topk = layer_of(t->name, "ffn_moe_topk-");
    const int il_prob = st->prefill ? layer_of(t->name, "ffn_moe_probs-") : -1;
    if (ask) {
        return il_topk >= 0 || il_prob >= 0;
    }
    std::vector<uint8_t> tmp;
    const uint8_t * base = nullptr;
    if (il_topk >= 0) {
        if (t->type != GGML_TYPE_I32 || il_topk >= st->n_layer || t->ne[0] != st->k) {
            st->error = std::string("unexpected top-k tensor ") + t->name;
            return false;
        }
        read_tensor_host(t, tmp, base);
        auto & dst = st->topk[il_topk];
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            for (int64_t i = 0; i < t->ne[0]; ++i) {
                const int32_t e = *(const int32_t *) (base + i * t->nb[0] + j * t->nb[1]);
                if (e < 0 || e >= st->n_expert) {
                    st->error = std::string("expert id out of range in ") + t->name;
                    return false;
                }
                dst.push_back(e);
            }
        }
        st->topk_calls[il_topk]++;
    } else if (il_prob >= 0) {
        if (t->type != GGML_TYPE_F32 || il_prob >= st->n_layer || t->ne[0] != st->n_expert) {
            st->error = std::string("unexpected probs tensor ") + t->name;
            return false;
        }
        read_tensor_host(t, tmp, base);
        auto & dst = st->prob_sum[il_prob];
        st->prob_rows[il_prob] += (int) t->ne[1];
        for (int64_t j = 0; j < t->ne[1]; ++j) {
            for (int64_t i = 0; i < t->ne[0]; ++i) {
                dst[i] += *(const float *) (base + i * t->nb[0] + j * t->nb[1]);
            }
        }
        st->prob_calls[il_prob]++;
    }
    return true;
}

// Minimal JSON string field extraction for the flat request objects we write ourselves.
std::string json_get_string(const std::string & line, const std::string & key) {
    const std::string pat = "\"" + key + "\"";
    size_t p = line.find(pat);
    if (p == std::string::npos) {
        throw std::runtime_error("missing key " + key);
    }
    p = line.find(':', p + pat.size());
    p = line.find('"', p);
    std::string out;
    for (size_t i = p + 1; i < line.size(); ++i) {
        char c = line[i];
        if (c == '"') {
            return out;
        }
        if (c == '\\') {
            char n = line[++i];
            switch (n) {
                case 'n': out += '\n'; break;
                case 't': out += '\t'; break;
                case 'r': out += '\r'; break;
                case 'b': out += '\b'; break;
                case 'f': out += '\f'; break;
                case 'u': {
                    unsigned cp = std::stoul(line.substr(i + 1, 4), nullptr, 16);
                    i += 4;
                    if (cp >= 0xD800 && cp <= 0xDBFF && line.compare(i + 1, 2, "\\u") == 0) {
                        unsigned lo = std::stoul(line.substr(i + 3, 4), nullptr, 16);
                        cp = 0x10000 + ((cp - 0xD800) << 10) + (lo - 0xDC00);
                        i += 6;
                    }
                    if (cp < 0x80) { out += (char) cp; }
                    else if (cp < 0x800) { out += (char) (0xC0 | (cp >> 6)); out += (char) (0x80 | (cp & 0x3F)); }
                    else if (cp < 0x10000) { out += (char) (0xE0 | (cp >> 12)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
                    else { out += (char) (0xF0 | (cp >> 18)); out += (char) (0x80 | ((cp >> 12) & 0x3F)); out += (char) (0x80 | ((cp >> 6) & 0x3F)); out += (char) (0x80 | (cp & 0x3F)); }
                    break;
                }
                default: out += n;
            }
        } else {
            out += c;
        }
    }
    throw std::runtime_error("unterminated string for key " + key);
}

std::string json_escape(const std::string & s) {
    std::string o;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n"; break;
            case '\r': o += "\\r"; break;
            case '\t': o += "\\t"; break;
            default:
                if (c < 0x20) { char b[8]; snprintf(b, sizeof(b), "\\u%04x", c); o += b; }
                else { o += (char) c; }
        }
    }
    return o;
}

template <typename T>
void write_int_array(FILE * f, const std::vector<T> & v) {
    fputc('[', f);
    for (size_t i = 0; i < v.size(); ++i) {
        fprintf(f, i ? ",%d" : "%d", (int) v[i]);
    }
    fputc(']', f);
}

void usage(const char * argv0) {
    fprintf(stderr,
        "usage: %s -m model.gguf --requests in.jsonl --out out.jsonl [-n max_gen] [-t threads]\n"
        "          [--ctx n_ctx] [--temp T] [--top-p P] [--seed S]\n"
        "  --temp 0 (default) is greedy decoding\n", argv0);
}

} // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    std::string model_path, req_path, out_path;
    int n_gen = 256, n_threads = 4, n_ctx = 2048;
    float temp = 0.0f, top_p = 1.0f;
    uint32_t seed = 0;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) { usage(argv[0]); exit(1); }
            return argv[++i];
        };
        if (a == "-m") model_path = next();
        else if (a == "--requests") req_path = next();
        else if (a == "--out") out_path = next();
        else if (a == "-n") n_gen = std::stoi(next());
        else if (a == "-t") n_threads = std::stoi(next());
        else if (a == "--ctx") n_ctx = std::stoi(next());
        else if (a == "--temp") temp = std::stof(next());
        else if (a == "--top-p") top_p = std::stof(next());
        else if (a == "--seed") seed = (uint32_t) std::stoul(next());
        else { usage(argv[0]); return 1; }
    }
    if (model_path.empty() || req_path.empty() || out_path.empty()) {
        usage(argv[0]);
        return 1;
    }

    ggml_backend_load_all();

    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), mp);
    if (!model) {
        fprintf(stderr, "error: unable to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(model);

    trace_state st;
    st.n_layer = llama_model_n_layer(model);
    {
        char buf[64];
        const char * arch_keys[] = {"%s.expert_count", "%s.expert_used_count"};
        char arch[64] = {0};
        llama_model_meta_val_str(model, "general.architecture", arch, sizeof(arch));
        int vals[2] = {0, 0};
        for (int j = 0; j < 2; ++j) {
            char key[128];
            snprintf(key, sizeof(key), arch_keys[j], arch);
            if (llama_model_meta_val_str(model, key, buf, sizeof(buf)) < 0) {
                fprintf(stderr, "error: model has no %s; not an MoE\n", key);
                return 1;
            }
            vals[j] = atoi(buf);
        }
        st.n_expert = vals[0];
        st.k        = vals[1];
    }
    fprintf(stderr, "route-trace: n_layer %d n_expert %d k %d\n", st.n_layer, st.n_expert, st.k);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx           = n_ctx;
    cp.n_batch         = n_ctx;
    cp.n_ubatch        = n_ctx;
    cp.n_threads       = n_threads;
    cp.n_threads_batch = n_threads;
    cp.cb_eval           = eval_cb;
    cp.cb_eval_user_data = &st;
    cp.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) {
        fprintf(stderr, "error: failed to create context\n");
        return 1;
    }

    std::ifstream in(req_path);
    if (!in) {
        fprintf(stderr, "error: cannot open %s\n", req_path.c_str());
        return 1;
    }
    FILE * out = fopen(out_path.c_str(), "w");
    if (!out) {
        fprintf(stderr, "error: cannot open %s\n", out_path.c_str());
        return 1;
    }

    std::string line;
    int n_req = 0;
    while (std::getline(in, line)) {
        if (line.empty()) {
            continue;
        }
        const std::string id   = json_get_string(line, "id");
        const std::string text = json_get_string(line, "text");

        const bool add_bos = llama_vocab_get_add_bos(vocab);
        int n_prompt = -llama_tokenize(vocab, text.c_str(), text.size(), nullptr, 0, add_bos, true);
        std::vector<llama_token> prompt(n_prompt);
        if (llama_tokenize(vocab, text.c_str(), text.size(), prompt.data(), prompt.size(), add_bos, true) < 0) {
            fprintf(stderr, "error: tokenize failed for %s\n", id.c_str());
            return 1;
        }
        if (n_prompt + n_gen > n_ctx) {
            fprintf(stderr, "error: request %s needs %d > n_ctx %d\n", id.c_str(), n_prompt + n_gen, n_ctx);
            return 1;
        }

        llama_memory_clear(llama_get_memory(ctx), true);

        llama_sampler * smpl = llama_sampler_chain_init(llama_sampler_chain_default_params());
        if (temp <= 0.0f) {
            llama_sampler_chain_add(smpl, llama_sampler_init_greedy());
        } else {
            llama_sampler_chain_add(smpl, llama_sampler_init_top_p(top_p, 1));
            llama_sampler_chain_add(smpl, llama_sampler_init_temp(temp));
            // per-request seed so a request's sample does not depend on its position in the file
            uint32_t h = seed;
            for (char c : id) { h = h * 16777619u ^ (uint8_t) c; }
            llama_sampler_chain_add(smpl, llama_sampler_init_dist(h));
        }

        st.topk.assign(st.n_layer, {});
        st.prob_sum.assign(st.n_layer, std::vector<double>(st.n_expert, 0.0));
        st.prob_rows.assign(st.n_layer, 0);
        std::vector<std::vector<int32_t>> prefill_topk;

        auto run = [&](llama_batch b, int n_tok) {
            st.topk_calls.assign(st.n_layer, 0);
            st.prob_calls.assign(st.n_layer, 0);
            std::vector<size_t> layer_before(st.n_layer);
            for (int il = 0; il < st.n_layer; ++il) layer_before[il] = st.topk[il].size();
            if (llama_decode(ctx, b) != 0 || !st.error.empty()) {
                fprintf(stderr, "error: decode failed for %s: %s\n", id.c_str(), st.error.c_str());
                exit(1);
            }
            // llama.cpp prunes the last layer to the rows that produce logits (inp_out_ids),
            // so during prefill that layer routes only the final prompt token. That is what
            // executes, so it is what the trace records.
            for (int il = 0; il < st.n_layer; ++il) {
                const size_t rows = (st.topk[il].size() - layer_before[il]) / st.k;
                const bool rows_ok = rows == (size_t) n_tok || (il == st.n_layer - 1 && rows == 1);
                if (st.topk_calls[il] != 1 || !rows_ok ||
                    (st.prefill && st.prob_calls[il] != 1)) {
                    fprintf(stderr, "error: layer %d delivered %d top-k / %d probs tensors for %s; trace incomplete\n",
                            il, st.topk_calls[il], st.prob_calls[il], id.c_str());
                    exit(1);
                }
            }
        };

        st.prefill = true;
        run(llama_batch_get_one(prompt.data(), prompt.size()), n_prompt);
        st.prefill = false;
        prefill_topk.swap(st.topk);
        st.topk.assign(st.n_layer, {});

        std::vector<llama_token> gen;
        std::string stop = "max";
        for (int i = 0; i < n_gen; ++i) {
            llama_token tok = llama_sampler_sample(smpl, ctx, -1);
            if (llama_vocab_is_eog(vocab, tok)) {
                stop = "eog";
                break;
            }
            gen.push_back(tok);
            run(llama_batch_get_one(&gen.back(), 1), 1);
        }
        llama_sampler_free(smpl);

        std::string gen_text;
        for (llama_token t : gen) {
            char buf[256];
            int n = llama_token_to_piece(vocab, t, buf, sizeof(buf), 0, true);
            if (n < 0) {
                fprintf(stderr, "error: detokenize failed\n");
                return 1;
            }
            gen_text.append(buf, n);
        }

        fprintf(out, "{\"id\":\"%s\",\"n_layer\":%d,\"n_expert\":%d,\"k\":%d,\"n_prompt\":%d,\"n_gen\":%zu,\"stop\":\"%s\",",
                json_escape(id).c_str(), st.n_layer, st.n_expert, st.k, n_prompt, gen.size(), stop.c_str());
        fprintf(out, "\"prompt_ids\":");
        write_int_array(out, prompt);
        fprintf(out, ",\"gen_ids\":");
        write_int_array(out, gen);
        fprintf(out, ",\"gen_text\":\"%s\"", json_escape(gen_text).c_str());
        fprintf(out, ",\"prefill_topk\":[");
        for (int il = 0; il < st.n_layer; ++il) {
            if (il) fputc(',', out);
            write_int_array(out, prefill_topk[il]);
        }
        fprintf(out, "],\"decode_topk\":[");
        for (int il = 0; il < st.n_layer; ++il) {
            if (il) fputc(',', out);
            write_int_array(out, st.topk[il]);
        }
        fprintf(out, "],\"prefill_prob_mean\":[");
        for (int il = 0; il < st.n_layer; ++il) {
            if (il) fputc(',', out);
            fputc('[', out);
            for (int e = 0; e < st.n_expert; ++e) {
                fprintf(out, e ? ",%.6g" : "%.6g", st.prob_sum[il][e] / st.prob_rows[il]);
            }
            fputc(']', out);
        }
        fprintf(out, "]}\n");
        fflush(out);
        ++n_req;
        fprintf(stderr, "route-trace: %s prompt %d gen %zu stop %s\n", id.c_str(), n_prompt, gen.size(), stop.c_str());
    }
    fclose(out);
    fprintf(stderr, "route-trace: %d requests\n", n_req);

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}
