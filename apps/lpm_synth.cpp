// Writes a random scene in the official JSON format.
#include <cstdio>
#include <cstdlib>
#include <string>

#include "lpm/synthetic.h"

using namespace lpm;

int main(int argc, char** argv) {
    SyntheticParams p;
    std::string out;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { if (i + 1 >= argc) exit(1); return argv[++i]; };
        if (a == "--n") p.n = atoi(next().c_str());
        else if (a == "--radius") p.radius = atof(next().c_str());
        else if (a == "--k") p.k = atof(next().c_str());
        else if (a == "--seed") p.seed = (unsigned)atoi(next().c_str());
        else if (a == "--extent") p.extent = atof(next().c_str());
        else if (a == "--shapes") p.shapes = next();
        else if (a == "--ops") p.ops = next();
        else if (a == "--chain") p.chain = true;
        else if (a == "--morton") p.morton = true;
        else if (a == "--out") out = next();
        else { fprintf(stderr, "unknown option %s\n", a.c_str()); return 1; }
    }
    if (out.empty()) { fprintf(stderr, "usage: lpm_synth --n N [--radius R] [--k K] [--seed S] [--extent E] [--shapes sphere|mixed] [--ops union|mixed] [--chain] [--morton] --out FILE\n"); return 1; }
    if (p.n < 1 || !(p.radius > 0.0) || !(p.extent > 0.0) || p.k < 0.0) {
        fprintf(stderr, "--n must be at least 1, --radius and --extent positive and --k not negative\n");
        return 1;
    }
    Scene s = make_synthetic(p);
    if (!write_scene_json(s, out)) {
        fprintf(stderr, "cannot write %s\n", out.c_str());
        return 1;
    }
    printf("%s: %zu nodes (%zu primitives), height %d\n", out.c_str(), s.nodes.size(), s.prims.size(), s.depth);
    return 0;
}
