// Capture the K and V tensors that are written into the KV cache, in full, for
// the native-packed-KV isolation tests (branch native-packed-kv, PREREG.md).
//
// Read-only: it installs an eval callback, copies Kcur-<il> and Vcur-<il> after
// RoPE, and writes each to a raw f32 file plus a manifest. Routing, compute and
// sampling are untouched.
//
//   llama-kv-capture -m MODEL -p PROMPT -c CTX --kv-out DIR
//
// Output: DIR/<name>.f32 (little-endian float32, ne0 fastest) and DIR/manifest.tsv
// with name, type, ne0..ne3, bytes and sha256 of each file's contents.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

struct capture_state {
    std::string dir;
    FILE *      manifest = nullptr;
    int         written  = 0;
};

static bool is_kv_tensor(const char * name) {
    return strncmp(name, "Kcur-", 5) == 0 || strncmp(name, "Vcur-", 5) == 0;
}

static bool capture_cb(struct ggml_tensor * t, bool ask, void * user_data) {
    auto * st = (capture_state *) user_data;
    if (ask) {
        return is_kv_tensor(t->name);       // only the cache-bound K and V
    }
    if (!is_kv_tensor(t->name) || t->type != GGML_TYPE_F32) {
        return true;
    }
    const size_t n = ggml_nelements(t);
    std::vector<float> buf(n);
    ggml_backend_tensor_get(t, buf.data(), 0, n * sizeof(float));

    // One file per (tensor, occurrence): RoPE rewrites the same name, so the
    // last write for a name in a graph is the one that reaches the cache.
    char path[1024];
    snprintf(path, sizeof(path), "%s/%s.f32", st->dir.c_str(), t->name);
    FILE * f = fopen(path, "wb");
    if (!f) {
        LOG_ERR("kv-capture: cannot write %s\n", path);
        return true;
    }
    fwrite(buf.data(), sizeof(float), n, f);
    fclose(f);
    // Record the producing op and the occurrence index, so "the file holds the
    // last write, which is post-RoPE" is verifiable from the manifest rather
    // than asserted: the last row for a name states its op.
    fprintf(st->manifest, "%s\t%s\t%d\tf32\t%lld\t%lld\t%lld\t%lld\t%zu\n", t->name,
            ggml_op_name(t->op), st->written,
            (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2],
            (long long) t->ne[3], n * sizeof(float));
    fflush(st->manifest);
    st->written++;
    return true;
}

int main(int argc, char ** argv) {
    common_params params;
    std::string out_dir = "kv_capture";
    // --kv-out is consumed here so the common parser does not see it
    for (int i = 1; i < argc - 1; ++i) {
        if (strcmp(argv[i], "--kv-out") == 0) {
            out_dir = argv[i + 1];
            for (int j = i; j + 2 <= argc; ++j) argv[j] = argv[j + 2];
            argc -= 2;
            break;
        }
    }
    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }
    common_init();
    llama_backend_init();
    llama_numa_init(params.numa);

    capture_state st;
    st.dir = out_dir;
    std::string mpath = out_dir + "/manifest.tsv";
    st.manifest = fopen(mpath.c_str(), "w");
    if (!st.manifest) {
        LOG_ERR("kv-capture: cannot write %s (does the directory exist?)\n", mpath.c_str());
        return 1;
    }
    fprintf(st.manifest, "name\top\toccurrence\ttype\tne0\tne1\tne2\tne3\tbytes\n");

    params.cb_eval           = capture_cb;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto init = common_init_from_params(params);
    auto * ip = init.get();
    if (!ip->model() || !ip->context()) {
        LOG_ERR("kv-capture: failed to load model\n");
        return 1;
    }
    const llama_vocab * vocab = llama_model_get_vocab(ip->model());
    std::vector<llama_token> tokens =
        common_tokenize(ip->context(), params.prompt, llama_vocab_get_add_bos(vocab), true);
    if (tokens.empty()) {
        LOG_ERR("kv-capture: no input tokens\n");
        return 1;
    }
    LOG_INF("kv-capture: %zu tokens, writing to %s\n", tokens.size(), out_dir.c_str());
    if (llama_decode(ip->context(), llama_batch_get_one(tokens.data(), tokens.size()))) {
        LOG_ERR("kv-capture: decode failed\n");
        return 1;
    }
    fclose(st.manifest);
    LOG_INF("kv-capture: wrote %d tensors\n", st.written);
    llama_backend_free();
    return 0;
}
