// Pruned trees (post-order entry lists), their evaluation and the two pruning oracles:
// the Lipschitz test of Barbier et al. and the interval arithmetic test of Keeter.
#pragma once
#include <cmath>
#include <vector>

#include "lpm/sdf.h"
#include "lpm/stats.h"

namespace lpm {

struct TreeEntry {
    uint32_t node;   // index in Scene::nodes
    int32_t parent;  // index in the same entry list, -1 for the root
    int8_t sign;     // the value of the entry is multiplied by sign
};

struct TreeView {
    const TreeEntry* e = nullptr;
    uint32_t n = 0;
};

inline std::vector<TreeEntry> full_tree(const Scene& s) {
    std::vector<TreeEntry> v(s.nodes.size());
    for (size_t i = 0; i < s.nodes.size(); i++)
        v[i] = TreeEntry{(uint32_t)i, s.nodes[i].parent, (int8_t)s.nodes[i].sign};
    return v;
}

// Per-thread scratch memory for evaluation and pruning.
struct Scratch {
    std::vector<double> dstack;
    std::vector<Interval> istack;
    std::vector<int> idx_stack;
    std::vector<uint8_t> state, inactive_anc, active_global;
    std::vector<int32_t> new_parent, remap;
    std::vector<int8_t> sign;
    void reserve(size_t n) {
        if (dstack.size() < n + 2) {
            dstack.resize(n + 2); istack.resize(n + 2); idx_stack.resize(n + 2);
            state.resize(n + 2); inactive_anc.resize(n + 2); active_global.resize(n + 2);
            new_parent.resize(n + 2); remap.resize(n + 2); sign.resize(n + 2);
        }
    }
};

inline double eval_tree(const Scene& s, TreeView t, const double p[3], Scratch& sc, Counters& c) {
    sc.reserve(t.n);
    double* st = sc.dstack.data();
    int sp = 0;
    const V3<double> pw{p[0], p[1], p[2]};
    for (uint32_t i = 0; i < t.n; i++) {
        const TreeEntry& e = t.e[i];
        const Node& n = s.nodes[e.node];
        double d;
        if (n.type == NODE_PRIM) {
            d = eval_primitive(s.prims[n.idx], pw);
            c.prim_evals++;
        } else {
            const double b = st[--sp];
            const double a = st[--sp];
            d = eval_op(s.ops[n.idx], a, b);
            c.op_evals++;
        }
        st[sp++] = e.sign < 0 ? -d : d;
    }
    return st[0];
}

constexpr double kIntervalEps = 1e-12;

inline Interval inflate(Interval v) {
    return Interval(v.lo - kIntervalEps * (1.0 + std::fabs(v.lo)), v.hi + kIntervalEps * (1.0 + std::fabs(v.hi)));
}

inline Interval eval_tree_interval(const Scene& s, TreeView t, const Interval box[3], Scratch& sc, Counters& c) {
    sc.reserve(t.n);
    Interval* st = sc.istack.data();
    int sp = 0;
    const V3<Interval> pw{box[0], box[1], box[2]};
    for (uint32_t i = 0; i < t.n; i++) {
        const TreeEntry& e = t.e[i];
        const Node& n = s.nodes[e.node];
        Interval d;
        if (n.type == NODE_PRIM) {
            d = inflate(eval_primitive(s.prims[n.idx], pw));
            c.prim_ivals++;
        } else {
            const Interval b = st[--sp];
            const Interval a = st[--sp];
            d = eval_op(s.ops[n.idx], a, b);
            c.op_ivals++;
        }
        st[sp++] = e.sign < 0 ? -d : d;
    }
    return st[0];
}

enum LocalState : uint8_t { STATE_INACTIVE = 0, STATE_SKIPPED = 1, STATE_ACTIVE = 2 };

// Second traversal (global state, reparenting, sign propagation) and compaction of the
// surviving entries, shared by both oracles. Appends the pruned tree to `out`.
inline uint32_t compact_pruned(TreeView in, Scratch& sc, std::vector<TreeEntry>& out) {
    const uint32_t n = in.n;
    uint32_t count = 0;
    for (int i = (int)n - 1; i >= 0; i--) {
        if (sc.state[i] == STATE_INACTIVE) {
            sc.active_global[i] = 0;
            sc.inactive_anc[i] = 1;
            continue;
        }
        const int pi = in.e[i].parent;
        const bool ia = pi >= 0 ? sc.inactive_anc[pi] != 0 : false;
        sc.inactive_anc[i] = ia;
        const bool ag = (sc.state[i] == STATE_ACTIVE) && !ia;
        sc.active_global[i] = ag;
        int8_t sg = in.e[i].sign;
        int32_t np = pi;
        if (pi >= 0 && sc.state[pi] == STATE_SKIPPED) {
            sg = (int8_t)(sg * sc.sign[pi]);
            np = sc.new_parent[pi];
        }
        sc.sign[i] = sg;
        sc.new_parent[i] = np;
        if (ag) count++;
    }
    const size_t base = out.size();
    out.resize(base + count);
    uint32_t out_idx = count;
    for (int i = (int)n - 1; i >= 0; i--) {
        if (!sc.active_global[i]) continue;
        --out_idx;
        sc.remap[i] = (int32_t)out_idx;
        const int32_t np = sc.new_parent[i];
        out[base + out_idx] = TreeEntry{in.e[i].node, np >= 0 ? sc.remap[np] : -1, sc.sign[i]};
    }
    return count;
}

// Lipschitz pruning (Procedures 1 and 2 of Barbier et al. 2025) for the closed ball of
// center `center` and radius R. Returns f(center) evaluated with the input tree and
// appends the pruned tree, valid on the whole ball, to `out`.
inline double prune_lipschitz(const Scene& s, TreeView in, const double center[3], double R,
                              std::vector<TreeEntry>& out, uint32_t& out_n, Scratch& sc, Counters& c) {
    const uint32_t n = in.n;
    sc.reserve(n);
    double* st = sc.dstack.data();
    int* is = sc.idx_stack.data();
    int sp = 0;
    const V3<double> p{center[0], center[1], center[2]};
    for (uint32_t i = 0; i < n; i++) {
        const TreeEntry& e = in.e[i];
        const Node& nd = s.nodes[e.node];
        double d;
        if (nd.type == NODE_PRIM) {
            d = eval_primitive(s.prims[nd.idx], p);
            c.prim_evals++;
            sc.state[i] = STATE_ACTIVE;
        } else {
            --sp; const double b = st[sp]; const int ri = is[sp];
            --sp; const double a = st[sp]; const int li = is[sp];
            const BinaryOp& op = s.ops[nd.idx];
            d = eval_op(op, a, b);
            c.op_evals++;
            if (std::fabs(a - b) > op.k + 2.0 * R) {
                sc.state[i] = STATE_SKIPPED;
                if (op.s * a < op.s * b) sc.state[ri] = STATE_INACTIVE;
                else sc.state[li] = STATE_INACTIVE;
            } else {
                sc.state[i] = STATE_ACTIVE;
            }
        }
        st[sp] = e.sign < 0 ? -d : d;
        is[sp] = (int)i;
        sp++;
    }
    const double root = st[0];
    out_n = compact_pruned(in, sc, out);
    return root;
}

// Interval pruning in the spirit of Keeter 2020: an operator is skipped when the
// interval bounds of its operands over the box prove that the blending kernel vanishes
// and that one operand wins everywhere. Returns the interval of f over the box.
inline Interval prune_interval(const Scene& s, TreeView in, const Interval box[3],
                               std::vector<TreeEntry>& out, uint32_t& out_n, Scratch& sc, Counters& c) {
    const uint32_t n = in.n;
    sc.reserve(n);
    Interval* st = sc.istack.data();
    int* is = sc.idx_stack.data();
    int sp = 0;
    const V3<Interval> p{box[0], box[1], box[2]};
    for (uint32_t i = 0; i < n; i++) {
        const TreeEntry& e = in.e[i];
        const Node& nd = s.nodes[e.node];
        Interval d;
        if (nd.type == NODE_PRIM) {
            d = inflate(eval_primitive(s.prims[nd.idx], p));
            c.prim_ivals++;
            sc.state[i] = STATE_ACTIVE;
        } else {
            --sp; const Interval B = st[sp]; const int ri = is[sp];
            --sp; const Interval A = st[sp]; const int li = is[sp];
            const BinaryOp& op = s.ops[nd.idx];
            d = eval_op(op, A, B);
            c.op_ivals++;
            const bool a_below = A.hi + op.k <= B.lo;
            const bool b_below = B.hi + op.k <= A.lo;
            if (a_below || b_below) {
                sc.state[i] = STATE_SKIPPED;
                const bool a_wins = op.s > 0.0 ? a_below : b_below;
                if (a_wins) sc.state[ri] = STATE_INACTIVE;
                else sc.state[li] = STATE_INACTIVE;
            } else {
                sc.state[i] = STATE_ACTIVE;
            }
        }
        st[sp] = e.sign < 0 ? -d : d;
        is[sp] = (int)i;
        sp++;
    }
    const Interval root = st[0];
    out_n = compact_pruned(in, sc, out);
    return root;
}

}  // namespace lpm
