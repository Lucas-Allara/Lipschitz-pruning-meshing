// Evaluation counters, timers and memory measurement.
#pragma once
#include <chrono>
#include <cstdint>
#include <sys/resource.h>

namespace lpm {

struct Counters {
    uint64_t prim_evals = 0;      // point evaluations of a primitive
    uint64_t op_evals = 0;        // point evaluations of a binary operator
    uint64_t prim_ivals = 0;      // interval evaluations of a primitive
    uint64_t op_ivals = 0;        // interval evaluations of a binary operator
    void add(const Counters& o) {
        prim_evals += o.prim_evals; op_evals += o.op_evals;
        prim_ivals += o.prim_ivals; op_ivals += o.op_ivals;
    }
    uint64_t node_evals() const { return prim_evals + op_evals + prim_ivals + op_ivals; }
};

struct Timer {
    std::chrono::steady_clock::time_point t0;
    Timer() : t0(std::chrono::steady_clock::now()) {}
    double seconds() const {
        return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    }
};

inline double peak_rss_mb() {
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return ru.ru_maxrss / 1024.0;
}

}  // namespace lpm
