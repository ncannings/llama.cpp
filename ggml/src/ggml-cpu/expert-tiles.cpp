// expert-tiles: expert-index to L3-domain placement for GGML_OP_MUL_MAT_ID.
// See expert-tiles.h for what this is, what it deliberately does not touch, and why it
// aborts rather than falling back.

#include "expert-tiles.h"

#include "ggml-impl.h"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <mutex>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#if defined(__linux__)
#    include <sched.h>
#endif


// ------------------------------------------------------------------ cached environment

static int   g_mode     = -1;   // -1 unread, 0 off, 1 pin and place, 2 pin only
static int   g_verbose  = -1;

static const char * sysfs_root() {
    const char * e = getenv("GGML_CPU_SYSFS_ROOT");
    return (e && *e) ? e : "/sys/devices/system/cpu";
}

// ------------------------------------------------------------------------- the plan
//
// A pure function of (n_threads, the process affinity mask, the sysfs topology). It is
// built once on the main thread and then read-only, so the workers need no synchronisation
// beyond the kickoff they already have.

struct ggml_expert_tiles_ctx {
    std::vector<int> cpus;
    int planned_for = -1;                 // n_threads this plan was built for, -1 = none

    int n_domains = 0;
    int thread_domain[GGML_MAX_N_THREADS];   // ith -> domain
    int thread_cpu   [GGML_MAX_N_THREADS];   // ith -> the one cpu it is pinned to
    int thread_rank  [GGML_MAX_N_THREADS];   // ith -> rank among its domain's threads
    int domain_nth   [GGML_EXPERT_TILES_MAX_DOMAINS];
    long domain_l3   [GGML_EXPERT_TILES_MAX_DOMAINS];  // bytes, for the record only
};

namespace {

std::once_flag env_once;
#if defined(__linux__)
thread_local cpu_set_t saved_affinity;
thread_local bool affinity_saved = false;
#endif

// map: a per-expert domain list, applied modulo its length. Empty means "block".
std::vector<int> g_map;
int              g_map_mode = -1;   // -1 unread, 0 block, 1 cyclic, 2 explicit list

// ------------------------------------------------------------------------ sysfs reading

bool read_file(const std::string & path, std::string & out) {
    FILE * f = fopen(path.c_str(), "r");
    if (!f) {
        return false;
    }
    char buf[4096];
    size_t n = fread(buf, 1, sizeof(buf) - 1, f);
    fclose(f);
    buf[n] = '\0';
    out.assign(buf);
    while (!out.empty() && (out.back() == '\n' || out.back() == ' ')) {
        out.pop_back();
    }
    return true;
}

// "0-4,7,9-10" -> the set of cpus, as a sorted vector
std::vector<int> parse_cpu_list(const std::string & s) {
    std::vector<int> out;
    const char * p = s.c_str();
    while (*p) {
        if (*p < '0' || *p > '9') { return {}; }
        char * end = nullptr;
        errno = 0;
        long a = strtol(p, &end, 10), b = a;
        if (errno || a > INT_MAX) { return {}; }
        p = end;
        if (*p == '-') {
            ++p;
            if (*p < '0' || *p > '9') { return {}; }
            errno = 0;
            b = strtol(p, &end, 10);
            if (errno || b > INT_MAX || b < a) { return {}; }
            p = end;
        }
        if (b - a > GGML_MAX_N_THREADS || out.size() + (size_t) (b - a + 1) > GGML_MAX_N_THREADS) { return {}; }
        for (long c = a; c <= b; ++c) { out.push_back((int) c); }
        if (!*p) { break; }
        if (*p++ != ',' || !*p) { return {}; }
    }
    return out;
}

// "8192K", "16384K", "2M" -> bytes. 0 if unreadable.
long parse_size(const std::string & s) {
    if (s.empty() || s[0] < '0' || s[0] > '9') { return 0; }
    char * end = nullptr;
    errno = 0;
    long v = strtol(s.c_str(), &end, 10), factor = 1;
    if (errno || v <= 0) { return 0; }
    if (*end == 'K' || *end == 'k') { factor = 1024; ++end; }
    else if (*end == 'M' || *end == 'm') { factor = 1024*1024; ++end; }
    if (*end || v > LONG_MAX/factor) { return 0; }
    return v*factor;
}

// The last-level (highest level) unified cache shared by this cpu. Returns false if the
// cpu has no cache topology at all, which is the abort condition: we will not guess.
bool cpu_llc(int cpu, std::vector<int> & shared, long & size_bytes) {
    const std::string base = std::string(sysfs_root()) + "/cpu" + std::to_string(cpu) + "/cache";
    int best_level = -1;
    for (int idx = 0; idx < 16; idx++) {
        const std::string d = base + "/index" + std::to_string(idx);
        std::string lvl, type, list, sz;
        if (!read_file(d + "/level", lvl)) {
            continue;
        }
        if (!read_file(d + "/type", type) || (type != "Unified" && type != "Data")) {
            continue;
        }
        const long level = lvl.find_first_not_of("0123456789") == std::string::npos ? parse_size(lvl) : 0;
        if (level < 2 || level > INT_MAX || level <= best_level) {
            continue;
        }
        if (!read_file(d + "/shared_cpu_list", list)) {
            continue;
        }
        if (!read_file(d + "/size", sz)) { continue; }
        const long bytes = parse_size(sz);
        auto cpus = parse_cpu_list(list);
        if (bytes <= 0 || cpus.empty()) { continue; }
        std::sort(cpus.begin(), cpus.end());
        if (std::adjacent_find(cpus.begin(), cpus.end()) != cpus.end() ||
            !std::binary_search(cpus.begin(), cpus.end(), cpu)) { continue; }
        best_level  = (int) level;
        shared      = cpus;
        size_bytes  = bytes;
    }
    return best_level > 0 && !shared.empty();
}

void parse_map(void) {
    const char * e = getenv("GGML_CPU_EXPERT_TILE_MAP");
    if (!e || !*e || strcmp(e, "block") == 0) {
        g_map_mode = 0;
        return;
    }
    if (strcmp(e, "cyclic") == 0) {
        g_map_mode = 1;
        return;
    }
    g_map = parse_cpu_list(e);   // same "a,b,c" / "a-b" grammar, reused
    if (g_map.empty()) {
        GGML_ABORT("expert-tiles: GGML_CPU_EXPERT_TILE_MAP='%s' parsed to an empty map", e);
    }
    g_map_mode = 2;
}

}  // namespace

// ------------------------------------------------------------------------- public API

static void read_mode(void) {
    std::call_once(env_once, [] {
    const char * e = getenv("GGML_CPU_EXPERT_TILES");
    const int v = e ? atoi(e) : 0;
    g_mode = (v == 1 || v == 2) ? v : 0;
    const char * vb = getenv("GGML_CPU_EXPERT_TILES_VERBOSE");
    g_verbose = (vb != NULL && atoi(vb) == 1) ? 1 : 0;
    if (g_mode) { parse_map(); }
    });
}

bool ggml_expert_tiles_enabled(void) {
    read_mode();
    return g_mode != 0;
}

bool ggml_expert_tiles_placed(void) {
    read_mode();
    return g_mode == 1;
}

struct ggml_expert_tiles_ctx * ggml_expert_tiles_new(const bool * mask) {
    if (!ggml_expert_tiles_enabled()) { return nullptr; }
#if !defined(__linux__)
    GGML_UNUSED(mask);
    GGML_ABORT("expert-tiles: placement requires Linux affinity");
#else
    auto * ctx = new ggml_expert_tiles_ctx{};
    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        GGML_ABORT("expert-tiles: sched_getaffinity failed");
    }
    bool explicit_mask = false;
    for (int c = 0; c < GGML_MAX_N_THREADS; ++c) { explicit_mask |= mask[c]; }
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (CPU_ISSET(c, &allowed) && (!explicit_mask || (c < GGML_MAX_N_THREADS && mask[c]))) {
            ctx->cpus.push_back(c);
        }
    }
    if (ctx->cpus.empty()) { GGML_ABORT("expert-tiles: the pool affinity mask is empty"); }
    return ctx;
#endif
}

void ggml_expert_tiles_free(struct ggml_expert_tiles_ctx * ctx) { delete ctx; }

void ggml_expert_tiles_plan(struct ggml_expert_tiles_ctx * ctx, int n_threads) {
    if (!ctx) { return; }
    auto & g_plan = *ctx;
    const auto & cpus = ctx->cpus;
    if (g_plan.planned_for == n_threads) { return; }
    if (n_threads < 1 || n_threads > GGML_MAX_N_THREADS) {
        GGML_ABORT("expert-tiles: n_threads %d out of range", n_threads);
    }
    if ((int) cpus.size() < n_threads) {
        GGML_ABORT("expert-tiles: %d threads but only %zu cpus in the affinity mask", n_threads, cpus.size());
    }
    // 2. group them by last-level cache. Domains are numbered by their lowest cpu, so the
    //    numbering is a property of the machine and not of the iteration order.
    std::vector<std::vector<int>> dom_cpus;
    std::vector<long>             dom_size;
    std::vector<std::vector<int>> dom_key;

    for (int c : cpus) {
        std::vector<int> shared;
        long sz = 0;
        if (!cpu_llc(c, shared, sz)) {
            GGML_ABORT("expert-tiles: no cache topology for cpu%d under %s. "
                       "The placement will not guess a topology.", c, sysfs_root());
        }
        int found = -1;
        for (size_t d = 0; d < dom_key.size(); d++) {
            if (dom_key[d] == shared) {
                found = (int) d;
                break;
            }
        }
        if (found < 0) {
            if ((int) dom_key.size() >= GGML_EXPERT_TILES_MAX_DOMAINS) {
                GGML_ABORT("expert-tiles: more than %d cache domains in the mask", GGML_EXPERT_TILES_MAX_DOMAINS);
            }
            dom_key.push_back(shared);
            dom_cpus.push_back({});
            dom_size.push_back(sz);
            found = (int) dom_key.size() - 1;
        }
        dom_cpus[found].push_back(c);
    }

    int ndom = (int) dom_cpus.size();

    // ONE THREAD IS THE DEGENERATE CASE, not a failure. With a single thread there is no
    // "which thread computes this expert" question to answer: that thread computes every
    // expert whatever the map says, exactly as it does with the placement off. So a
    // one-thread pool is planned as a single domain. Nothing is silently disabled, because
    // a one-thread pool has no partition to disable.
    if (n_threads == 1) {
        ndom = 1;
        dom_cpus.assign(1, { cpus[0] });
        dom_size.assign(1, 0);
    } else if (n_threads < ndom) {
        // More than one thread but not enough to cover every domain: a domain would end up
        // with no threads and its experts would never be computed. Refuse.
        GGML_ABORT("expert-tiles: %d threads across %d cache domains. Every domain needs at least one "
                   "thread or its experts would never be computed.", n_threads, ndom);
    }

    // 3. deal the threads out over the domains, round robin, one distinct cpu each. Round
    //    robin rather than block so that an even pool splits evenly, which is what the
    //    equal-split default expert map is balanced against.
    std::vector<size_t> next(ndom, 0);
    memset(g_plan.domain_nth, 0, sizeof(g_plan.domain_nth));
    for (int t = 0; t < n_threads; t++) {
        int d = -1;
        for (int k = 0; k < ndom; k++) {
            const int cand = (t + k) % ndom;
            if (next[cand] < dom_cpus[cand].size()) {
                d = cand;
                break;
            }
        }
        if (d < 0) {
            GGML_ABORT("expert-tiles: ran out of cpus dealing thread %d", t);
        }
        g_plan.thread_domain[t] = d;
        g_plan.thread_cpu[t]    = dom_cpus[d][next[d]++];
        g_plan.thread_rank[t]   = g_plan.domain_nth[d]++;
    }

    for (int d = 0; d < ndom; d++) {
        if (g_plan.domain_nth[d] == 0) {
            GGML_ABORT("expert-tiles: domain %d got no threads", d);
        }
        g_plan.domain_l3[d] = dom_size[d];
    }

    if (g_map_mode == 2 && n_threads > 1) {
        for (int d : g_map) {
            if (d < 0 || d >= ndom) { GGML_ABORT("expert-tiles: map domain %d outside %d domains", d, ndom); }
        }
    }

    g_plan.n_domains  = ndom;
    g_plan.planned_for = n_threads;


    if (g_verbose == 1) {
        fprintf(stderr, "expert-tiles: mode %d (%s), %d threads over %d domains\n", g_mode,
                g_mode == 1 ? "pin and place" : "pin only, experts dealt pool-wide", n_threads, ndom);
        for (int d = 0; d < ndom; d++) {
            fprintf(stderr, "expert-tiles:   domain %d  llc %ld KB  threads %d  cpus", d, dom_size[d] / 1024,
                    g_plan.domain_nth[d]);
            for (int t = 0; t < n_threads; t++) {
                if (g_plan.thread_domain[t] == d) {
                    fprintf(stderr, " %d(t%d)", g_plan.thread_cpu[t], t);
                }
            }
            fprintf(stderr, "\n");
        }
    }
}

void ggml_expert_tiles_apply(const struct ggml_expert_tiles_ctx * ctx, int ith) {
    if (!ctx) { return; }
    const auto & g_plan = *ctx;
#if defined(__linux__)
    if (g_plan.planned_for < 0) {
        GGML_ABORT("expert-tiles: thread %d reached apply with no plan", ith);
    }
    if (ith < 0 || ith >= g_plan.planned_for) {
        GGML_ABORT("expert-tiles: thread %d is outside the plan for %d threads. The pool grew after "
                   "the plan was built; the placement will not run unpinned.", ith, g_plan.planned_for);
    }
    if (affinity_saved || sched_getaffinity(0, sizeof(saved_affinity), &saved_affinity) != 0) {
        GGML_ABORT("expert-tiles: could not save worker affinity");
    }
    affinity_saved = true;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(g_plan.thread_cpu[ith], &set);
    if (sched_setaffinity(0, sizeof(set), &set) != 0) {
        GGML_ABORT("expert-tiles: could not pin thread %d to cpu %d", ith, g_plan.thread_cpu[ith]);
    }

#else
    GGML_UNUSED(ith);
#endif
}

void ggml_expert_tiles_restore(void) {
#if defined(__linux__)
    if (affinity_saved) {
        if (sched_setaffinity(0, sizeof(saved_affinity), &saved_affinity) != 0) {
            GGML_ABORT("expert-tiles: could not restore worker affinity");
        }
        affinity_saved = false;
    }
#endif
}

int ggml_expert_tiles_planned_for(const struct ggml_expert_tiles_ctx * ctx) {
    const auto & g_plan = *ctx;
    return g_plan.planned_for;
}

int ggml_expert_tiles_n_domains(const struct ggml_expert_tiles_ctx * ctx) {
    const auto & g_plan = *ctx;
    return g_plan.n_domains;
}

int ggml_expert_tiles_thread_domain(const struct ggml_expert_tiles_ctx * ctx, int ith) {
    const auto & g_plan = *ctx;
    if (ith < 0 || ith >= g_plan.planned_for) {
        GGML_ABORT("expert-tiles: thread %d outside the plan for %d", ith, g_plan.planned_for);
    }
    return g_plan.thread_domain[ith];
}

int ggml_expert_tiles_domain_nth(const struct ggml_expert_tiles_ctx * ctx, int dom) {
    const auto & g_plan = *ctx;
    if (dom < 0 || dom >= g_plan.n_domains) {
        GGML_ABORT("expert-tiles: domain %d outside 0..%d", dom, g_plan.n_domains - 1);
    }
    return g_plan.domain_nth[dom];
}

int ggml_expert_tiles_domain_rank(const struct ggml_expert_tiles_ctx * ctx, int ith) {
    const auto & g_plan = *ctx;
    if (ith < 0 || ith >= g_plan.planned_for) {
        GGML_ABORT("expert-tiles: thread %d outside the plan for %d", ith, g_plan.planned_for);
    }
    return g_plan.thread_rank[ith];
}

int ggml_expert_tiles_expert_domain(const struct ggml_expert_tiles_ctx * ctx, int expert, int n_expert) {
    const auto & g_plan = *ctx;
    const int ndom = g_plan.n_domains;
    if (ndom <= 1) {
        return 0;
    }
    switch (g_map_mode) {
        case 1:  // cyclic
            return expert % ndom;
        case 2:  // explicit list, applied modulo its length
            {
                const int d = g_map[(size_t) expert % g_map.size()];
                if (d < 0 || d >= ndom) {
                    GGML_ABORT("expert-tiles: map sends expert %d to domain %d, but there are %d domains",
                               expert, d, ndom);
                }
                return d;
            }
        default: // block: an equal split, matching the equal thread split
            return (int) (((long) expert * ndom) / (n_expert > 0 ? n_expert : 1));
    }
}
