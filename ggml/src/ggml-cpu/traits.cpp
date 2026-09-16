#include "traits.h"

#include "ggml-backend-impl.h"
#include "ggml-backend.h"
#include "moe-tail.h"

namespace ggml::cpu {
tensor_traits::~tensor_traits() {}

extra_buffer_type::~extra_buffer_type() {}
}  // namespace ggml::cpu

// mm-invoke change 3: EXTRA-BUFFER DISPATCH SHORTCUT.
//
// Every node of every graph enters this function, and on a decode graph most of them are the
// matmuls whose per-invocation cost this branch is about. The walk itself is a call into
// ggml_backend_cpu_get_extra_buffer_types(), which is a function-local static and therefore
// carries a thread-safe-initialisation guard load on EVERY call, followed by a std::vector
// iteration through a pointer that has to be loaded from the heap. The list is fixed for the
// life of the process and is one entry long on this build.
//
// The shortcut is a flat array filled once, before any worker thread exists, by
// ggml_cpu_extra_cache_init() from ggml_graph_compute. It changes no dispatch DECISION: the
// same contexts are visited in the same order and the same tensor_traits is asked the same
// question. GGML_CPU_NO_MM_DISPATCH=1 restores the vector walk.
static ggml::cpu::extra_buffer_type * g_extra_ctx[8];
static int                            g_extra_n = -1;

void ggml_cpu_extra_cache_init(void) {
    if (g_extra_n >= 0) {
        return;
    }
    int n = 0;
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context && n < 8) {
            g_extra_ctx[n++] = (ggml::cpu::extra_buffer_type *) extra->context;
        }
    }
    g_extra_n = n;
}

bool ggml_cpu_extra_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * op) {
    if (g_extra_n >= 0 && ggml_mm_dispatch()) {
        for (int i = 0; i < g_extra_n; i++) {
            auto tensor_traits = g_extra_ctx[i]->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
        return false;
    }
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->compute_forward(params, op)) {
                return true;
            }
        }
    }
    return false;
}

bool ggml_cpu_extra_work_size(int n_threads, const struct ggml_tensor * op, size_t * size) {
    for (auto extra : ggml_backend_cpu_get_extra_buffer_types()) {
        if (extra && extra->context) {
            auto buf_extra     = (ggml::cpu::extra_buffer_type *) extra->context;
            auto tensor_traits = buf_extra->get_tensor_traits(op);
            if (tensor_traits && tensor_traits->work_size(n_threads, op, *size)) {
                return true;
            }
        }
    }
    return false;
}
