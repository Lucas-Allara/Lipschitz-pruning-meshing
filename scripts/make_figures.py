"""Tables (results/tables.md) and figures (figures/) from results/raw/*.json.

    python scripts/make_figures.py
"""
import json
import filecmp
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "results" / "raw"
MESHES = ROOT / "results" / "meshes"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)

COLORS = {"full": "#7e848a", "lipschitz": "#2a78d6", "interval": "#eb6834", "libfive": "#1baf7a", "dense": "#b5604e", "hybrid": "#7b3fb8"}
LABELS = {"full": "octree + DC, full tree", "lipschitz": "octree + DC, Lipschitz pruning",
          "interval": "octree + DC, interval pruning", "hybrid": "octree + DC, Lipschitz pruning + interval culling", "libfive": "libfive"}
plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#dcdcd6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False})


def load(tag):
    p = RAW / f"{tag}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return None if "failed" in d else d


def failed(tag):
    p = RAW / f"{tag}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return d.get("failed")


def fmt_time(s):
    return f"{s:.3f}" if s < 10 else f"{s:.1f}"


def fmt_big(n):
    if n >= 1e9:
        return f"{n / 1e9:.2f} G"
    if n >= 1e6:
        return f"{n / 1e6:.1f} M"
    if n >= 1e3:
        return f"{n / 1e3:.1f} k"
    return str(n)


def leaf(d):
    return d["levels"][-1] if d.get("levels") else None


lines = []


def table(title, header, rows):
    lines.append(f"\n### {title}\n")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(x) for x in r) + " |")


# ------------------------------------------------------------------ main table
def main_table():
    rows = []
    for s in ["trees", "monument", "molecule"]:
        for m in ["full", "lipschitz", "interval", "hybrid"]:
            d = load(f"official_{s}_{m}_d8")
            if not d:
                continue
            same = "-"
            a, b = MESHES / f"official_{s}_full_d8.obj", MESHES / f"official_{s}_{m}_d8.obj"
            if a.exists() and b.exists():
                same = "yes" if filecmp.cmp(a, b, shallow=False) else "NO"
            lf = leaf(d)
            rows.append([s if m == "full" else "", d["nodes"], LABELS[m], fmt_time(d["time_total"]), fmt_time(d["time_octree"]),
                         fmt_time(d["time_mesh"]), fmt_big(d["prim_evals"]), f"{lf['avg_tree']:.2f}", lf["max_tree"],
                         f"{d['pool_bytes'] / 2**20:.1f}", f"{d['peak_rss_mb']:.0f}", d["triangles"], same])
        lib = load(f"libfive_{s}_d8")
        if lib:
            rows.append(["", "", "libfive (interval, DC)", fmt_time(lib["time_render"]), "-", "-", "-", "-", "-", "-",
                         f"{lib['peak_rss_mb']:.0f}", lib["triangles"], "see Hausdorff"])
    table("Official scenes, depth 8 (256^3 finest cells), 32 threads",
          ["scene", "nodes", "method", "total s", "octree s", "DC s", "primitive evals", "avg tree (leaves)",
           "max tree", "pruned trees MB", "peak RSS MB", "triangles", "mesh identical to full"], rows)


# ------------------------------------------------------------------ speedups
def speedups():
    rows = []
    for s in ["trees", "monument", "molecule"]:
        f, l, i = load(f"official_{s}_full_d8"), load(f"official_{s}_lipschitz_d8"), load(f"official_{s}_interval_d8")
        if not (f and l):
            continue
        lf = leaf(l)
        rows.append([s, d_ratio := f"{f['prim_evals'] / l['prim_evals']:.0f}x", f"{f['time_total'] / l['time_total']:.1f}x",
                     f"{l['nodes'] / lf['avg_tree']:.0f}x", f"{lf['single_prim'] / lf['cells'] * 100:.0f} %",
                     f"{i['prim_evals'] / l['prim_evals']:.2f}x" if i else "-", f"{i['time_total'] / l['time_total']:.2f}x" if i else "-"])
    table("Speedups of Lipschitz pruning (depth 8)",
          ["scene", "primitive evals, full / Lipschitz", "time, full / Lipschitz", "nodes / avg leaf tree (H2 estimate)",
           "leaf cells with one primitive", "primitive evals, interval / Lipschitz", "time, interval / Lipschitz"], rows)


# ------------------------------------------------------------------ scaling figure
def scaling():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    rows = []
    for k, ls in [("0", "-"), ("0.05", "--")]:
        for m in ["full", "lipschitz", "interval"]:
            ns, ts, es = [], [], []
            for n in [10, 100, 1000, 10000]:
                d = load(f"scaling_n{n}_k{k}_{m}")
                if d:
                    ns.append(n); ts.append(d["time_total"]); es.append(d["prim_evals"])
                    rows.append([n, k, m, fmt_time(d["time_total"]), fmt_big(d["prim_evals"]), f"{leaf(d)['avg_tree']:.2f}", d["triangles"]])
            if ns:
                axes[0].loglog(ns, ts, ls, marker="o", ms=3, color=COLORS[m], label=f"{LABELS[m]}, k = {k}")
                axes[1].loglog(ns, es, ls, marker="o", ms=3, color=COLORS[m])
    axes[0].set_xlabel("spheres"); axes[0].set_ylabel("time (s)")
    axes[1].set_xlabel("spheres"); axes[1].set_ylabel("primitive evaluations")
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "scaling.pdf"); fig.savefig(FIG / "scaling.png", dpi=200)
    table("Scaling: random spheres of radius 0.05 in [-1, 1]^3, depth 8",
          ["spheres", "k", "method", "total s", "primitive evals", "avg leaf tree", "triangles"], rows)


# ------------------------------------------------------------------ k sweep
def ksweep():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    rows = []
    ks = ["0", "0.0125", "0.025", "0.05", "0.1", "0.2", "0.4"]
    for m in ["lipschitz", "interval"]:
        xs, ts, es, avg = [], [], [], []
        for k in ks:
            d = load(f"ksweep_n1000_k{k}_{m}")
            if d:
                xs.append(float(k) / 0.05); ts.append(d["time_total"]); es.append(d["prim_evals"]); avg.append(leaf(d)["avg_tree"])
                rows.append([k, f"{float(k) / 0.05:.2f}", m, fmt_time(d["time_total"]), fmt_big(d["prim_evals"]), f"{leaf(d)['avg_tree']:.2f}", leaf(d)["max_tree"]])
        if xs:
            axes[0].plot(xs, es, marker="o", ms=3, color=COLORS[m], label=LABELS[m])
            axes[1].plot(xs, avg, marker="o", ms=3, color=COLORS[m])
    for k in ["0", "0.4"]:
        d = load(f"ksweep_n1000_k{k}_full")
        if d:
            rows.append([k, f"{float(k) / 0.05:.2f}", "full", fmt_time(d["time_total"]), fmt_big(d["prim_evals"]), "-", "-"])
    axes[0].set_xlabel("blending radius k / r"); axes[0].set_ylabel("primitive evaluations"); axes[0].set_yscale("log")
    axes[1].set_xlabel("blending radius k / r"); axes[1].set_ylabel("avg pruned tree, leaves"); axes[1].set_yscale("log")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "ksweep.pdf"); fig.savefig(FIG / "ksweep.png", dpi=200)
    table("Blending radius sweep: 1000 spheres of radius r = 0.05, depth 8",
          ["k", "k / r", "method", "total s", "primitive evals", "avg leaf tree", "max leaf tree"], rows)


# ------------------------------------------------------------------ adverse
def adverse():
    rows = []
    for k in ["0", "0.05"]:
        for m in ["full", "lipschitz", "interval"]:
            d = load(f"adverse_fluid_k{k}_{m}")
            if d:
                lf = leaf(d)
                rows.append([k, m, fmt_time(d["time_total"]), fmt_big(d["prim_evals"]), f"{lf['avg_tree']:.1f}", lf["max_tree"],
                             f"{lf['single_prim'] / max(lf['cells'], 1) * 100:.0f} %", d["triangles"]])
    table("Adverse case: 5000 spheres of radius 0.05 packed in [-0.25, 0.25]^3, depth 8",
          ["k", "method", "total s", "primitive evals", "avg leaf tree", "max leaf tree", "single-primitive leaves", "triangles"], rows)


# ------------------------------------------------------------------ ablations
def ablations():
    rows = []
    for d8 in [5, 6, 7, 8, 9]:
        f, l = load(f"depth_trees_full_d{d8}"), load(f"depth_trees_lipschitz_d{d8}")
        mo = load(f"depth_monument_lipschitz_d{d8}")
        rows.append([d8, 2 ** d8, fmt_time(f["time_total"]) if f else "-", fmt_time(l["time_total"]) if l else "-",
                     f"{leaf(l)['avg_tree']:.2f}" if l else "-", fmt_big(l["prim_evals"]) if l else "-",
                     fmt_time(mo["time_total"]) if mo else "-", f"{leaf(mo)['avg_tree']:.2f}" if mo else "-", mo["triangles"] if mo else "-"])
    table("Depth sweep (32 threads)", ["depth", "cells per axis", "Trees full s", "Trees Lipschitz s", "Trees avg leaf tree",
                                        "Trees primitive evals", "Monument Lipschitz s", "Monument avg leaf tree", "Monument triangles"], rows)
    rows = []
    for s in ["trees", "monument", "molecule"]:
        a, b = load(f"official_{s}_lipschitz_d8"), load(f"noinherit_{s}_lipschitz_d8")
        if a and b:
            rows.append([s, fmt_time(a["time_octree"]), fmt_time(b["time_octree"]), fmt_big(a["octree_prim_evals"]), fmt_big(b["octree_prim_evals"]),
                         f"{leaf(a)['avg_tree']:.2f}", f"{leaf(b)['avg_tree']:.2f}"])
    table("Pruning from the parent tree versus from the full tree at every cell",
          ["scene", "octree s, inherited", "octree s, from full", "octree primitive evals, inherited", "from full", "avg leaf tree, inherited", "from full"], rows)
    rows = []
    for s in ["molecule", "spheres_n1000_k0", "spheres_n1000_k0.05"]:
        a, b = load(f"analytic_{s}_lipschitz_d8_base"), load(f"analytic_{s}_lipschitz_d8")
        if a and b:
            rows.append([s, fmt_big(a["mesh_prim_evals"]), fmt_big(b["mesh_prim_evals"]), f"{b['analytic_hits']}/{b['crossings']}",
                         fmt_time(a["time_mesh"]), fmt_time(b["time_mesh"])])
    table("Analytic single-sphere leaves (crossing and normal without evaluations)",
          ["scene", "DC primitive evals", "with analytic leaves", "analytic crossings", "DC s", "DC s, analytic"], rows)
    rows = []
    for s in ["spheres_n1000_k0", "chain_n1000_k0", "mixed_n1000_k0.025"]:
        for m in ["lipschitz", "interval"]:
            d = load(f"shape_{s}_{m}")
            if d:
                rows.append([s, d["tree_height"], m, fmt_time(d["time_total"]), fmt_big(d["prim_evals"]), f"{leaf(d)['avg_tree']:.2f}", leaf(d)["max_tree"]])
    table("Tree shape: balanced versus chain, and mixed primitives and operators (1000 primitives)",
          ["scene", "tree height", "method", "total s", "primitive evals", "avg leaf tree", "max leaf tree"], rows)
    rows = []
    for s in ["trees", "monument", "molecule"]:
        a, b = load(f"official_{s}_lipschitz_d8"), load(f"threads1_{s}_lipschitz_d8")
        f1 = load("threads1_trees_full_d8") if s == "trees" else None
        if a and b:
            rows.append([s, fmt_time(b["time_total"]), fmt_time(a["time_total"]), f"{b['time_total'] / a['time_total']:.1f}x",
                         fmt_time(f1["time_total"]) if f1 else "-"])
    table("Threads", ["scene", "Lipschitz, 1 thread s", "Lipschitz, 32 threads s", "parallel speedup", "full tree, 1 thread s"], rows)


# ------------------------------------------------------------------ dense
def dense():
    rows = []
    for s in ["trees", "monument", "molecule"]:
        for m in ["full", "lipschitz"]:
            d, mc = load(f"dense_{s}_{m}_d8"), load(f"dense_{s}_{m}_d8_mc")
            if d:
                rows.append([s, "dense grid, " + ("full tree" if m == "full" else "Lipschitz pruning"), fmt_time(d["time_dense"]),
                             fmt_time(d["time_octree"]), fmt_big(d["prim_evals"]), f"{d['pool_bytes'] / 2**20:.1f}",
                             f"{d['peak_rss_mb']:.0f}", fmt_time(mc["time_mc"]) if mc else "-", mc["triangles"] if mc else "-"])
        o = load(f"official_{s}_lipschitz_d8")
        if o:
            rows.append([s, "octree + DC, Lipschitz pruning", fmt_time(o["time_mesh"]), fmt_time(o["time_octree"]), fmt_big(o["prim_evals"]),
                         f"{o['pool_bytes'] / 2**20:.1f}", f"{o['peak_rss_mb']:.0f}", "-", o["triangles"]])
    table("Dense 257^3 discretization + marching cubes versus the sparse octree (depth 8)",
          ["scene", "pipeline", "sampling / DC s", "pruning s", "primitive evals", "pruned trees MB", "peak RSS MB", "MC s", "triangles"], rows)


# ------------------------------------------------------------------ hausdorff against libfive and MC
def quality():
    import trimesh
    rows = []
    lib_rows = []
    hz = ROOT / "scripts" / "hausdorff.py"
    cache = RAW / "hausdorff.json"
    hd = json.load(open(cache)) if cache.exists() else {}
    pairs = []
    for s in ["trees", "monument", "molecule"]:
        pairs.append((s, "libfive", MESHES / f"official_{s}_lipschitz_d8.obj", MESHES / f"libfive_{s}_d8.obj"))
        pairs.append((s, "dense MC (full tree)", MESHES / f"official_{s}_lipschitz_d8.obj", MESHES / f"dense_{s}_full_d8_mc.obj"))
        pairs.append((s, "dense MC (pruned)", MESHES / f"dense_{s}_full_d8_mc.obj", MESHES / f"dense_{s}_lipschitz_d8_mc.obj"))
    for s, other, a, b in pairs:
        key = f"{s}|{other}"
        if not (a.exists() and b.exists()):
            continue
        if filecmp.cmp(a, b, shallow=False):
            rows.append([s, other, "0 (byte-identical)", "0", "-", "-"])
            continue
        if key not in hd:
            out = subprocess.run([sys.executable, str(hz), str(a), str(b), "20000"], capture_output=True, text=True, check=True).stdout
            hd[key] = json.loads(out.strip().splitlines()[-1])
            json.dump(hd, open(cache, "w"), indent=1)
        h = hd[key]
        rows.append([s, other, f"{h['hausdorff']:.2e}", f"{h['mean']:.2e}", h["vertices_a"], h["vertices_b"]])
    cell = {"trees": 2.34 / 256, "monument": 2.8 / 256, "molecule": 2.2 / 256}
    rows = [r + [f"{cell[r[0]]:.2e}"] for r in rows]
    for lib in ["trees", "monument", "molecule", "spheres_n100_k0", "spheres_n1000_k0", "spheres_n1000_k0.05", "spheres_n10000_k0"]:
        d, f = load(f"libfive_{lib}_d8"), failed(f"libfive_{lib}_d8")
        o = load(f"official_{lib}_lipschitz_d8") or load(f"scaling_{lib.replace('spheres_', '')}_lipschitz")
        ours = fmt_time(o["time_total"]) if o else "-"
        if d:
            lib_rows.append([lib, fmt_time(d["time_render"]), ours, d["triangles"], f"{d['peak_rss_mb']:.0f}"])
        elif f:
            lib_rows.append([lib, "failed: " + ("memory > 24 GB" if f == "CalledProcessError" else "timeout 30 min"), ours, "-", "> 24000"])
    closed = []
    cache2 = RAW / "closedness.json"
    cl = json.load(open(cache2)) if cache2.exists() else {}
    for s in ["trees", "monument", "molecule"]:
        path = MESHES / f"official_{s}_lipschitz_d8.obj"
        if not path.exists():
            continue
        if s not in cl:
            m = trimesh.load(path, process=False)
            b, n = edge_counts(np.asarray(m.faces))
            cl[s] = {"boundary": b, "nonmanifold": n, "triangles": int(len(m.faces))}
            json.dump(cl, open(cache2, "w"), indent=1)
        closed.append(f"{s}: {cl[s]['boundary']} boundary edges, {cl[s]['nonmanifold']} non-manifold edges of {cl[s]['triangles']} triangles")
    table("Distances between meshes (20 000 surface samples plus the vertices of each mesh; the closest-point "
          "routine of trimesh has a noise floor of about 1e-6 to 1e-4 on the sliver triangles of marching cubes, so the "
          "mean is the reliable figure; the DC meshes of the three oracles are byte-identical). Closedness of the DC meshes: " + "; ".join(closed) + ".",
          ["scene", "compared with", "Hausdorff", "mean", "vertices A", "vertices B", "cell size"], rows)
    table("libfive (interval arithmetic octree + dual contouring, 32 workers, depth 8; its Tree::optimized() pass is skipped "
          "because it unfolds shared subtrees and is exponential in the height of blended trees)",
          ["scene", "libfive render s", "octree + DC Lipschitz s", "libfive triangles", "libfive peak RSS MB"], lib_rows)


# ------------------------------------------------------------------ animation: a sphere moving through the scene
def animation():
    rows = []
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for ax, s in zip(axes, ["trees", "monument"]):
        for m, c in [("full", COLORS["full"]), ("scratch", COLORS["lipschitz"]), ("incremental", "#1baf7a"), ("patch", "#7b3fb8")]:
            d = load(f"anim_{s}_{m}")
            if not d:
                continue
            fr = d["frames_data"]
            times = [f["time_total"] for f in fr]
            ax.plot([f["frame"] for f in fr], times, marker="o", ms=2.5, color=c,
                    label={"full": "full tree", "scratch": "Lipschitz, from scratch", "incremental": "Lipschitz, cells reused, index rebuilt",
                           "patch": "Lipschitz, contour patched"}[m])
            cells = np.mean([f["cells"] for f in fr])
            static = np.mean([f["static_leaves"] / max(f["cells"], 1) for f in fr]) if m == "incremental" else 0.0
            reused = np.mean([f["reused_vertices"] / max(f["cells"], 1) for f in fr]) if m == "incremental" else 0.0
            chk = load(f"anim_{s}_check") if m == "incremental" else (load(f"anim_{s}_patch_check") if m == "patch" else None)
            if m == "patch":
                # Quads of the static mesh that the patch keeps (the runs before 2026-11-16 wrote them as static_leaves).
                kept = [f.get("patch_kept_quads", f["static_leaves"]) for f in fr]
                static = np.mean([k / max(k + f["patch_new_quads"], 1) for k, f in zip(kept, fr)])
            rows.append([s, m, len(fr), fmt_time(d["mean_frame_seconds"]), fmt_time(d["max_frame_seconds"]),
                         fmt_time(np.mean([f["time_octree"] for f in fr])), fmt_time(np.mean([f["time_dc"] for f in fr])),
                         f"{cells:.0f}", f"{static * 100:.0f} %" if m in ("incremental", "patch") else "-",
                         f"{reused * 100:.0f} %" if m == "incremental" else "-",
                         fmt_big(int(np.mean([f["prim_evals"] for f in fr]))),
                         fmt_time(d["setup_seconds"]) if m in ("incremental", "patch") else "-",
                         ("yes" if chk and chk.get("all_identical") else "-") if m in ("incremental", "patch") else "-"])
        ax.set_yscale("log"); ax.set_xlabel("frame"); ax.set_ylabel("seconds per frame"); ax.set_title(s, fontsize=9)
    axes[0].legend(fontsize=6.5)
    fig.tight_layout()
    fig.savefig(FIG / "animation.pdf"); fig.savefig(FIG / "animation.png", dpi=200)
    table("A sphere of radius 0.2 moving through the scene (smooth union, k = 0.1), re-meshed every frame at depth 8",
          ["scene", "method", "frames", "mean s/frame", "max s/frame", "octree s", "DC s", "leaf cells (patch: region)",
           "static leaves (patch: quads kept)", "vertices reused", "primitive evals/frame", "setup s", "identical to scratch"], rows)


# ------------------------------------------------------------------ sharp features: dihedral angles of DC versus MC
def sharpness():
    import trimesh
    cache = RAW / "sharpness.json"
    sh = json.load(open(cache)) if cache.exists() else {}
    rows = []
    for s in ["trees", "monument", "molecule"]:
        for label, path in [("octree + DC", MESHES / f"official_{s}_lipschitz_d8.obj"), ("dense MC", MESHES / f"dense_{s}_full_d8_mc.obj")]:
            key = f"{s}|{label}"
            if not path.exists():
                continue
            if key not in sh:
                m = trimesh.load(path, process=False)
                ang = np.degrees(m.face_adjacency_angles)
                sh[key] = {"edges": int(len(ang)), "over30": float((ang > 30).mean()), "over60": float((ang > 60).mean()),
                           "p99": float(np.percentile(ang, 99)), "max": float(ang.max())}
                json.dump(sh, open(cache, "w"), indent=1)
            r = sh[key]
            rows.append([s, label, r["edges"], f"{r['over30'] * 100:.2f} %", f"{r['over60'] * 100:.3f} %", f"{r['p99']:.1f}", f"{r['max']:.1f}"])
    table("Sharp features: dihedral angles between adjacent triangles (dual contouring keeps the creases of the boxes; marching cubes rounds them)",
          ["scene", "mesh", "interior edges", "angle > 30 deg", "angle > 60 deg", "99th percentile (deg)", "max (deg)"], rows)


# ------------------------------------------------------------------ adaptive contouring (coarser leaves)
def adaptive():
    import trimesh
    cache = RAW / "adaptive_quality.json"
    aq = json.load(open(cache)) if cache.exists() else {}
    rows = []
    for s in ["trees", "monument", "molecule"]:
        u = load(f"official_{s}_lipschitz_d8")
        for tag, label in [("single_d8", "single-primitive cells, tol 0.02"), ("all_d8", "any cell, tol 0.02"), ("all_tol0.1_d8", "any cell, tol 0.1")]:
            d = load(f"adaptive_{s}_{tag}")
            if not (d and u):
                continue
            key = f"{s}|{tag}"
            if key not in aq:
                a, b = MESHES / f"official_{s}_lipschitz_d8.obj", MESHES / f"adaptive_{s}_{tag}.obj"
                if not (a.exists() and b.exists()):
                    continue
                m = trimesh.load(b, process=False)
                out = subprocess.run([sys.executable, str(ROOT / "scripts" / "hausdorff.py"), str(a), str(b), "20000"], capture_output=True, text=True, check=True).stdout
                h = json.loads(out.strip().splitlines()[-1])
                boundary, nonmanifold = edge_counts(np.asarray(m.faces))
                aq[key] = {"hausdorff": h["hausdorff"], "mean": h["mean"], "boundary": boundary, "nonmanifold": nonmanifold}
                json.dump(aq, open(cache, "w"), indent=1)
            q = aq[key]
            ad = d["adaptive"]
            cell = {"trees": 2.34 / 256, "monument": 2.8 / 256, "molecule": 2.2 / 256}[s]
            rows.append([s if tag == "single_d8" else "", label, ad["fine_leaves"], ad["leaves"], f"{(1 - ad['leaves'] / ad['fine_leaves']) * 100:.0f} %",
                         d["vertices"], d["triangles"], u["triangles"], fmt_time(d["time_total"]), fmt_time(u["time_total"]),
                         f"{q['hausdorff'] / cell:.2f}", f"{q['mean'] / cell:.4f}", q["boundary"], q["nonmanifold"]])
    table("Adaptive contouring: nodes collapsed bottom-up where the children's quadric fits one vertex (tolerance relative to the cell size) and the topology is kept",
          ["scene", "collapse rule", "fine leaves", "leaves", "leaves saved", "vertices", "triangles", "uniform triangles", "total s", "uniform s",
           "Hausdorff / cell", "mean / cell", "boundary edges", "non-manifold edges"], rows)


def edge_counts(F):
    """Half-edges without their opposite (boundary) and edges with more than two triangles."""
    import collections
    cnt = collections.Counter()
    for a, b, c in F:
        cnt[(a, b)] += 1; cnt[(b, c)] += 1; cnt[(c, a)] += 1
    boundary = sum(1 for (u, v), n in cnt.items() if cnt.get((v, u), 0) != n)
    nonmanifold = sum(1 for (u, v), n in cnt.items() if n > 1 and u < v)
    return int(boundary), int(nonmanifold)


# ------------------------------------------------------------------ Morton-ordered trees (alternative D)
def morton():
    rows = []
    for s in ["fluid_n5000_k0", "fluid_n5000_k0.05", "spheres_n10000_k0", "spheres_n10000_k0.05"]:
        a, b = load(f"morton_{s}_random_lipschitz"), load(f"morton_{s}_lipschitz")
        if not (a and b):
            continue
        rows.append([s, fmt_time(a["time_total"]), fmt_time(b["time_total"]), fmt_big(a["prim_evals"]), fmt_big(b["prim_evals"]),
                     f"{leaf(a)['avg_tree']:.1f}", f"{leaf(b)['avg_tree']:.1f}", "identical (min is associative)" if s.endswith("k0") else "different surface (smooth union is not associative)"])
    table("Primitives sorted along a Morton curve before building the balanced tree (alternative D of the plan)",
          ["scene", "random order s", "Morton order s", "evals, random", "evals, Morton", "avg leaf tree, random", "Morton", "mesh"], rows)


# ------------------------------------------------------------------ per-level tree size figure
def levels_figure():
    fig, ax = plt.subplots(figsize=(3.6, 2.8))
    for s, c in [("trees", "#2a78d6"), ("monument", "#eb6834"), ("molecule", "#1baf7a")]:
        d = load(f"official_{s}_lipschitz_d8")
        if d:
            ax.semilogy([l["depth"] for l in d["levels"]], [l["avg_tree"] for l in d["levels"]], marker="o", ms=3, color=c, label=f"{s} ({d['nodes']} nodes)")
            i = load(f"official_{s}_interval_d8")
            if i:
                ax.semilogy([l["depth"] for l in i["levels"]], [l["avg_tree"] for l in i["levels"]], ":", color=c)
    ax.set_xlabel("octree depth"); ax.set_ylabel("avg active nodes per cell")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "levels.pdf"); fig.savefig(FIG / "levels.png", dpi=200)


if __name__ == "__main__":
    main_table(); speedups(); scaling(); ksweep(); adverse(); ablations(); dense(); quality(); animation(); sharpness(); adaptive(); morton(); levels_figure()
    (ROOT / "results" / "tables.md").write_text("# Results\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))
