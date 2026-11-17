"""Measures in one session the GPU times of the poster (the comment of TIMES in
poster/poster_figures.py names them), with the GPU already prepared and the mesh kept on it.

    python scripts/measure_gpu.py [--out FILE.json] [--runs N] [--repeat R] [--frames F] [--depth D] [--no-viewer] [--no-full]

At depth 8, for Trees, Monument and Molecule:
  strip    build/lpm_mesh --gpu --repeat with intervals and Lipschitz pruning: the median of the
           last R warm runs of a process (the first three are left out, see below), then the median
           of N processes; the full tree, which takes minutes on Monument, runs once per scene
           (--repeat 3, its time_resident) after everything else, so that its load does not fall on
           the short runs;
  frame    build/lpm_anim --method gpu-patch --keep-on-gpu: the median frame of F frames, then the
           median of N runs;
  scratch  the same with --method gpu (the mesh rebuilt from scratch every frame), on Monument;
  cad      the same on six parts of scenes/inversecsg with Lipschitz pruning, uniform and with
           coarse leaves (--adaptive-topdown --adaptive-tol 0.005), when the scenes are there;
  viewer   build/lpm_view monument D --dc --bench 300, dual contouring as on the poster (it opens a window;
           --no-viewer skips it).

The GPU clocks follow the power state and the temperature, and an idle GPU takes a few seconds to
raise them, also after a light run: Monument with Lipschitz pruning measured 105, 169 and 189 ms
in three processes of a cold session, and 103 to 105 ms warm. So the session starts with a warm-up
that is not recorded, a process leaves out its first three warm runs, the runs of a group alternate
(a drift falls on all of them alike), every time is reported with its range over the processes, and
the state of the GPU and of the power supply is recorded before and after. Nothing is written
unless --out is given: the table goes to stdout.
"""
import argparse
import json
import re
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "build"
SCENES = ["trees", "monument", "molecule"]
METHODS = ["full", "interval", "lipschitz"]
CAD = ["lock", "159", "161", "145", "base", "mount"]


def state():
    """The GPU and the power supply, as far as the system tells."""
    s = {"time": time.strftime("%Y-%m-%d %H:%M:%S")}
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=name,power.draw,clocks.gr,clocks.mem,temperature.gpu,memory.used",
                            "--format=csv,noheader"], capture_output=True, text=True, timeout=20)
        s["nvidia_smi"] = q.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        s["nvidia_smi"] = None
    supply = Path("/sys/class/power_supply")
    for p in sorted(supply.glob("*")) if supply.exists() else []:
        for name in ("online", "status", "capacity"):
            f = p / name
            if f.exists():
                try:
                    s[f"{p.name}.{name}"] = f.read_text().strip()
                except OSError:
                    pass
    return s


def stats_of(args):
    """Runs a program that writes its statistics with --stats and returns them."""
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "stats.json"
        subprocess.run([str(a) for a in args] + ["--stats", str(out)], check=True, stderr=subprocess.DEVNULL)
        return json.load(open(out))


RAMP = 3  # warm runs of a process left out: the clocks are still rising


def mesh_time(scene_path, method, depth, repeat, extra=(), ramp=RAMP):
    d = stats_of([BIN / "lpm_mesh", "--gpu", "--scene", scene_path, "--method", method, "--depth", depth, "--repeat", repeat + ramp, *extra])
    return statistics.median(d["time_resident_runs"][ramp:]), d


def frame_time(scene, method, depth, frames):
    d = stats_of([BIN / "lpm_anim", "--scene", ROOT / "scenes" / f"{scene}.json", "--depth", depth, "--method", method,
                  "--keep-on-gpu", "--frames", frames])
    return statistics.median(f["time_total"] for f in d["frames_data"])


def ms(v):
    return f"{v:.1f} s" if v >= 1 else (f"{v * 1e3:.0f} ms" if v >= 0.01 else f"{v * 1e3:.2f} ms")


def spread(v):
    """Median of the processes, with their range when there are several."""
    m = ms(statistics.median(v))
    return m if len(v) < 2 else f"{m} ({ms(min(v))} to {ms(max(v))})"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, help="write every measurement as JSON")
    ap.add_argument("--runs", type=int, default=3, help="processes per measurement (the full tree runs once)")
    ap.add_argument("--repeat", type=int, default=5, help="warm runs per process of lpm_mesh that count")
    ap.add_argument("--frames", type=int, default=48)
    ap.add_argument("--depth", type=int, default=8)
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--no-full", action="store_true", help="skip the full tree")
    a = ap.parse_args()
    res = {"depth": a.depth, "runs": a.runs, "repeat": a.repeat, "frames": a.frames, "before": state()}
    print("GPU:", res["before"]["nvidia_smi"], file=sys.stderr)
    # Warm-up, not recorded: the clocks of an idle GPU are low.
    for _ in range(2):
        mesh_time(ROOT / "scenes" / "monument.json", "lipschitz", a.depth, a.repeat)
    res["warm"] = state()

    # Strip of times: the methods alternate inside every round.
    strip = {s: {m: [] for m in METHODS} for s in SCENES}
    counts = {}
    for r in range(a.runs):
        for s in SCENES:
            for m in ("interval", "lipschitz"):
                t, d = mesh_time(ROOT / "scenes" / f"{s}.json", m, a.depth, a.repeat)
                strip[s][m].append(t)
                counts[s] = {"triangles": d["triangles"], "device": d.get("gpu")}
                print(f"  {s} {m}: {ms(t)}", file=sys.stderr)

    # Dynamic frame, and the frame from scratch on Monument, alternating too.
    frame = {s: [] for s in SCENES}
    scratch = []
    for r in range(a.runs):
        for s in SCENES:
            frame[s].append(frame_time(s, "gpu-patch", a.depth, a.frames))
        scratch.append(frame_time("monument", "gpu", a.depth, a.frames))
        print(f"  frames, round {r + 1}: " + ", ".join(f"{s} {ms(frame[s][-1])}" for s in SCENES) + f", from scratch {ms(scratch[-1])}", file=sys.stderr)
    res["frame"] = {s: statistics.median(v) for s, v in frame.items()}
    res["frame_runs"] = frame
    res["scratch_monument"] = statistics.median(scratch)
    res["scratch_monument_runs"] = scratch

    # CAD parts: uniform and coarse leaves, alternating.
    cad_dir = ROOT / "scenes" / "inversecsg"
    cad = {}
    if all((cad_dir / f"{p}.json").exists() for p in CAD):
        runs = {p: {"uniform": [], "coarse": []} for p in CAD}
        for r in range(a.runs):
            for p in CAD:
                tu, du = mesh_time(cad_dir / f"{p}.json", "lipschitz", a.depth, a.repeat)
                tc, dc = mesh_time(cad_dir / f"{p}.json", "lipschitz", a.depth, a.repeat, ["--adaptive-topdown", "--adaptive-tol", 0.005])
                runs[p]["uniform"].append(tu)
                runs[p]["coarse"].append(tc)
                cad[p] = {"triangles_uniform": du["triangles"], "triangles_coarse": dc["triangles"]}
        for p in CAD:
            cad[p]["uniform"] = statistics.median(runs[p]["uniform"])
            cad[p]["coarse"] = statistics.median(runs[p]["coarse"])
            cad[p]["runs"] = runs[p]
    else:
        print("scenes/inversecsg is not here: the CAD table is skipped", file=sys.stderr)
    res["cad"] = cad

    # Viewer.
    fps = []
    view = BIN / "lpm_view"
    if not a.no_viewer and view.exists():
        for r in range(a.runs):
            out = subprocess.run([str(view), "monument", str(a.depth), "--dc", "--bench", "300"], cwd=ROOT, capture_output=True, text=True)
            m = re.search(r"([\d.]+) fps", out.stdout)
            if out.returncode != 0 or not m:
                print("lpm_view --bench failed:", out.stderr.strip()[-200:], file=sys.stderr)
                break
            fps.append(float(m.group(1)))
    res["viewer_fps_runs"] = fps
    res["viewer_fps"] = statistics.median(fps) if fps else None

    # The full tree last: one process per scene.
    if not a.no_full:
        for s in SCENES:
            t, _ = mesh_time(ROOT / "scenes" / f"{s}.json", "full", a.depth, 3, ramp=0)
            strip[s]["full"].append(t)
            print(f"  {s} full: {ms(t)}", file=sys.stderr)
    res["strip"] = {s: {m: statistics.median(v) for m, v in strip[s].items() if v} for s in SCENES}
    res["strip_runs"] = strip
    res["counts"] = counts
    res["after"] = state()

    # The table of the poster.
    print(f"\nGPU times at depth {a.depth} (warm, the mesh kept on the GPU)")
    print(f"{'':10s} {'full tree':>10s} {'intervals':>26s} {'Lipschitz':>26s} {'frame':>30s}   full/Lip.  int./Lip.")
    for s in SCENES:
        v = res["strip"][s]
        full = v.get("full")
        print(f"{s:10s} {ms(full) if full else '-':>10s} {spread(strip[s]['interval']):>26s} {spread(strip[s]['lipschitz']):>26s} {spread(frame[s]):>30s}"
              f"   {full / v['lipschitz'] if full else float('nan'):8.0f}x  {v['interval'] / v['lipschitz']:8.2f}x")
    print(f"Monument rebuilt from scratch: {spread(scratch)} per frame")
    if cad:
        print("\nCAD parts: uniform and coarse leaves (time, triangles)")
        for p in CAD:
            c = cad[p]
            print(f"{p:6s} {ms(c['uniform']):>8s} {c['triangles_uniform']:>9d}   {ms(c['coarse']):>8s} {c['triangles_coarse']:>9d}   x{c['coarse'] / c['uniform']:.2f}")
    if fps:
        print(f"\nlpm_view monument {a.depth} --dc --bench 300: {res['viewer_fps']:.1f} fps (" + ", ".join(f"{f:.1f}" for f in fps) + ")")
    print(f"\nbefore: {res['before']['nvidia_smi']}\nwarm:   {res['warm']['nvidia_smi']}\nafter:  {res['after']['nvidia_smi']}")
    print("power:  " + ", ".join(f"{k} {v}" for k, v in res["before"].items() if k not in ("time", "nvidia_smi")))
    if a.out:
        json.dump(res, open(a.out, "w"), indent=1)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
