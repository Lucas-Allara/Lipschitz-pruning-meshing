"""Runs the experiments and stores one JSON per run in results/raw/.

    python scripts/run_experiments.py [group ...]

Groups: official, scaling, ksweep, adverse, ablation, dense, libfive. Without arguments
all groups run. Runs whose output already exists are skipped (delete the JSON to redo).
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build" / "lpm_mesh"
LIBFIVE_BIN = ROOT / "build" / "lpm_libfive"
PY = sys.executable
RAW = ROOT / "results" / "raw"
MESHES = ROOT / "results" / "meshes"
DEPTH = 8
THREADS = int(os.environ.get("LPM_THREADS", "32"))
TIMEOUT = 1800          # seconds per run
MEMORY_GB = 24          # address space limit per run


def limit_memory():
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (MEMORY_GB * 2**30, MEMORY_GB * 2**30))


OFFICIAL = ["trees", "monument", "molecule"]
METHODS = ["full", "lipschitz", "interval"]


def scene_path(name: str) -> Path:
    p = ROOT / "scenes" / f"{name}.json"
    return p if p.exists() else ROOT / "scenes" / "synthetic" / f"{name}.json"


def run_mesh(tag: str, scene: str, method: str, depth: int = DEPTH, extra=(), keep_mesh=False, attr=False):
    out = RAW / f"{tag}.json"
    if out.exists():
        return json.load(open(out))
    args = [str(BIN), "--scene", str(scene_path(scene)), "--method", method, "--depth", str(depth),
            "--threads", str(THREADS), "--stats", str(out)]
    if keep_mesh:
        args += ["--out", str(MESHES / f"{tag}.obj")]
    if attr:
        args += ["--attr", str(MESHES / f"{tag}_tree_n.txt")]
    args += list(extra)
    print("run", tag, flush=True)
    t0 = time.perf_counter()
    try:
        subprocess.run(args, check=True, timeout=TIMEOUT, preexec_fn=limit_memory)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        # A run that exceeds the memory or time budget is recorded as failed.
        d = {"tag": tag, "scene_name": scene, "method": method, "depth": depth, "failed": type(e).__name__,
             "wall_seconds": time.perf_counter() - t0, "memory_limit_gb": MEMORY_GB, "timeout_s": TIMEOUT}
        json.dump(d, open(out, "w"), indent=1)
        print("failed", tag, d["failed"], flush=True)
        return d
    d = json.load(open(out))
    d["wall_seconds"] = time.perf_counter() - t0
    d["tag"] = tag
    d["scene_name"] = scene
    json.dump(d, open(out, "w"), indent=1)
    return d


def group_official():
    for s in OFFICIAL:
        for m in METHODS + ["hybrid"]:
            run_mesh(f"official_{s}_{m}_d{DEPTH}", s, m, keep_mesh=True, attr=(m == "lipschitz"))


def group_scaling():
    for n in [10, 100, 1000, 10000]:
        for k in ["0", "0.05"]:
            for m in METHODS:
                if m == "full" and n == 10000:
                    continue  # several hundred billion evaluations; the curve is extrapolated
                run_mesh(f"scaling_n{n}_k{k}_{m}", f"spheres_n{n}_k{k}", m)


def group_ksweep():
    for k in ["0", "0.0125", "0.025", "0.05", "0.1", "0.2", "0.4"]:
        for m in ["lipschitz", "interval"]:
            run_mesh(f"ksweep_n1000_k{k}_{m}", f"spheres_n1000_k{k}", m)
    run_mesh("ksweep_n1000_k0_full", "spheres_n1000_k0", "full")
    run_mesh("ksweep_n1000_k0.4_full", "spheres_n1000_k0.4", "full")


def group_adverse():
    for k in ["0", "0.05"]:
        for m in METHODS:
            run_mesh(f"adverse_fluid_k{k}_{m}", f"fluid_n5000_k{k}", m, keep_mesh=(m == "lipschitz"))


def group_ablation():
    # Depth sweep.
    for d in [5, 6, 7, 8, 9]:
        for m in ["full", "lipschitz"]:
            if m == "full" and d == 9:
                continue
            run_mesh(f"depth_trees_{m}_d{d}", "trees", m, depth=d)
        run_mesh(f"depth_monument_lipschitz_d{d}", "monument", "lipschitz", depth=d)
    # Pruning from the full tree at every cell instead of from the parent.
    for s in OFFICIAL:
        run_mesh(f"noinherit_{s}_lipschitz_d{DEPTH}", s, "lipschitz", extra=["--no-inherit"])
    # Analytic single-sphere leaves.
    for s in ["molecule", "spheres_n1000_k0", "spheres_n1000_k0.05"]:
        run_mesh(f"analytic_{s}_lipschitz_d{DEPTH}", s, "lipschitz", extra=["--analytic"])
        run_mesh(f"analytic_{s}_lipschitz_d{DEPTH}_base", s, "lipschitz")
    # Tree shape: chain versus balanced, and mixed primitives and operators.
    for s in ["chain_n1000_k0", "spheres_n1000_k0", "mixed_n1000_k0.025"]:
        for m in ["lipschitz", "interval"]:
            run_mesh(f"shape_{s}_{m}", s, m)
    # Threads: single-threaded wall time of the main scenes.
    global THREADS
    saved = THREADS
    THREADS = 1
    for s in OFFICIAL:
        run_mesh(f"threads1_{s}_lipschitz_d{DEPTH}", s, "lipschitz")
    run_mesh(f"threads1_trees_full_d{DEPTH}", "trees", "full")
    THREADS = saved


def group_dense():
    for s in OFFICIAL:
        for m in ["full", "lipschitz"]:
            tag = f"dense_{s}_{m}_d{DEPTH}"
            values = MESHES / f"{tag}.f32"
            d = run_mesh(tag, s, m, extra=["--dense", str(values)])
            mc = RAW / f"{tag}_mc.json"
            if not mc.exists():
                subprocess.run([PY, str(ROOT / "scripts" / "marching_cubes.py"), str(values), str(scene_path(s)),
                                str(DEPTH), str(MESHES / f"{tag}_mc.obj"), str(mc)], check=True)
            if values.exists():
                values.unlink()


def group_libfive():
    if not LIBFIVE_BIN.exists():
        print("lpm_libfive not built; skipping")
        return
    scenes = OFFICIAL + ["spheres_n100_k0", "spheres_n1000_k0", "spheres_n1000_k0.05", "spheres_n10000_k0"]
    for s in scenes:
        out = RAW / f"libfive_{s}_d{DEPTH}.json"
        if out.exists():
            continue
        print("run libfive", s, flush=True)
        t0 = time.perf_counter()
        try:
            subprocess.run([str(LIBFIVE_BIN), "--scene", str(scene_path(s)), "--depth", str(DEPTH), "--threads", str(THREADS),
                            "--out", str(MESHES / f"libfive_{s}_d{DEPTH}.obj"), "--stats", str(out)],
                           check=True, timeout=TIMEOUT, preexec_fn=limit_memory)
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            json.dump({"tag": f"libfive_{s}_d{DEPTH}", "scene_name": s, "method": "libfive", "failed": type(e).__name__,
                       "wall_seconds": time.perf_counter() - t0}, open(out, "w"), indent=1)
            print("failed libfive", s, flush=True)


GROUPS = {"official": group_official, "scaling": group_scaling, "ksweep": group_ksweep, "adverse": group_adverse,
          "ablation": group_ablation, "dense": group_dense, "libfive": group_libfive}

if __name__ == "__main__":
    RAW.mkdir(parents=True, exist_ok=True)
    MESHES.mkdir(parents=True, exist_ok=True)
    for g in (sys.argv[1:] or list(GROUPS)):
        GROUPS[g]()
