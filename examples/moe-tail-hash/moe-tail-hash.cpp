// moe-tail-hash: hash EVERY tensor of EVERY graph a real model evaluates, so that two
// runs of the same model under different moe-tail switch settings can be compared bit
// for bit rather than only at the generated text.
//
// The graph callback fires once per computed node with the node's own buffer, so the
// hash below covers every intermediate tensor, not just the logits. It is FNV-1a over
// the raw bytes of the node's byte span, which is exactly a bit comparison: any single
// differing byte anywhere in any intermediate changes the printed line.
//
//   moe-tail-hash -m <model> -p <prompt> -t <threads> -ngl 0
//
// Run it twice with the switch under test set differently and diff the two outputs.
// A non-empty diff is a failure; an empty diff is the bit-identity evidence.

#include "arg.h"
#include "common.h"
#include "log.h"
#include "llama.h"
#include "ggml.h"

#include <clocale>
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

struct hash_state {
    std::vector<uint8_t> buf;
    long                 n_nodes = 0;
    long                 n_graphs = 0;
    int                  last_idx = -1;
};

static uint64_t fnv1a(const uint8_t * p, size_t n) {
    uint64_t h = 1469598103934665603ULL;
    for (size_t i = 0; i < n; i++) {
        h ^= (uint64_t) p[i];
        h *= 1099511628211ULL;
    }
    return h;
}

static bool cb_hash(struct ggml_tensor * t, bool ask, void * user_data) {
    hash_state * st = (hash_state *) user_data;

    if (ask) {
        return true; // observe every node
    }

    const size_t n = ggml_nbytes(t);
    st->buf.resize(n);
    ggml_backend_tensor_get(t, st->buf.data(), 0, n);

    printf("%6ld %6ld %-16s %-40s [%lld,%lld,%lld,%lld] %016llx\n",
           st->n_graphs, st->n_nodes, ggml_op_name(t->op), t->name,
           (long long) t->ne[0], (long long) t->ne[1], (long long) t->ne[2], (long long) t->ne[3],
           (unsigned long long) fnv1a(st->buf.data(), n));

    st->n_nodes++;
    return true;
}

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");

    common_params params;
    params.prompt = "The capital of France is";

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_COMMON)) {
        return 1;
    }

    llama_backend_init();
    llama_numa_init(params.numa);

    hash_state st;
    params.cb_eval           = cb_hash;
    params.cb_eval_user_data = &st;
    params.warmup            = false;

    auto llama_init = common_init_from_params(params);

    auto * model = llama_init->model();
    auto * ctx   = llama_init->context();

    if (model == nullptr || ctx == nullptr) {
        fprintf(stderr, "moe-tail-hash: failed to init\n");
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const bool add_bos = llama_vocab_get_add_bos(vocab);

    std::vector<llama_token> tokens = common_tokenize(ctx, params.prompt, add_bos, true);
    if (tokens.empty()) {
        fprintf(stderr, "moe-tail-hash: empty prompt\n");
        return 1;
    }

    // one prompt graph, then a few single-token graphs, so both the batched and the
    // one-token shapes of every op are hashed
    if (llama_decode(ctx, llama_batch_get_one(tokens.data(), tokens.size()))) {
        fprintf(stderr, "moe-tail-hash: decode failed\n");
        return 1;
    }
    st.n_graphs++;

    llama_token tok = tokens.back();
    const int n_dec = params.n_predict > 0 ? params.n_predict : 4;
    for (int i = 0; i < n_dec; i++) {
        if (llama_decode(ctx, llama_batch_get_one(&tok, 1))) {
            fprintf(stderr, "moe-tail-hash: decode failed\n");
            return 1;
        }
        st.n_graphs++;
    }

    printf("moe-tail-hash: graphs=%ld nodes=%ld\n", st.n_graphs, st.n_nodes);

    llama_backend_free();
    return 0;
}
