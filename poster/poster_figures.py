"""Figures of the poster. The LaTeX sources set the captions; the plots carry only their axis
labels, and the strip of times also its scene titles and row names.

Uses results/raw (run scripts/run_experiments.py first), results/meshes, the GPU times in TIMES,
and build/lpm_anim and build/lpm_mesh, which mesh the dynamic frame and the coarse leaves on the
GPU (Vulkan).

    python poster/poster_figures.py [teaser] [plots] [times] [dynamic] [coarse]

Without arguments it writes every figure into poster/figures:
  teaser   the three meshes colored by the size of the pruned tree of the cell of each vertex
           (teaser.png) with their color scale (colorbar.pdf);
  plots    the scaling plot (scaling.pdf);
  times    the strip of GPU times per scene (times.pdf);
  dynamic  one frame of the sphere crossing Monument, with the triangles rebuilt by the patch in
           burgundy (dynamic.png);
  coarse   the CAD part 145 with the uniform mesh and with coarse leaves, edges drawn: the whole
           part (coarse_overview.png) and the same detail of both (coarse_uniform_zoom.png,
           coarse_coarse_zoom.png).
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import render_meshes as rm  # noqa: E402

RAW = ROOT / "results" / "raw"
MESHES = ROOT / "results" / "meshes"
OUT = Path(__file__).resolve().parent / "figures"
FONTS = Path.home() / ".local" / "share" / "TinyTeX" / "texmf-dist" / "fonts" / "opentype" / "public" / "montserrat"

INK = "#2b2b2d"
INK2 = "#5e5e63"
GRID = "#e4dde0"
COLORS = {"full": "#b4b4b8", "lipschitz": "#820a32", "interval": "#008c9a"}
BURGUNDY = LinearSegmentedColormap.from_list("burgundy", ["#f6e6ec", "#c9748f", "#820a32", "#3a0416"])
MM = 1 / 25.4


def setup_fonts():
    for f in FONTS.glob("Montserrat-*.otf"):
        fm.fontManager.addfont(str(f))
    plt.rcParams.update({"font.family": "Montserrat", "font.size": 20, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0, "axes.axisbelow": True,
                         "legend.frameon": False, "lines.linewidth": 3.0, "lines.markersize": 9})


def load(tag):
    p = RAW / f"{tag}.json"
    if not p.exists():
        return None
    d = json.load(open(p))
    return None if "failed" in d else d


# ------------------------------------------------------------------ teaser: meshes colored by tree size
def crop_transparent(path):
    a = np.asarray(Image.open(path).convert("RGBA")).copy()
    a[(a[:, :, 0] > 250) & (a[:, :, 1] > 250) & (a[:, :, 2] > 250), 3] = 0
    im = Image.fromarray(a)
    return im.crop(im.getbbox())


def teaser():
    panels = []
    for s in ["trees", "monument", "molecule"]:
        mesh = rm.load_obj(MESHES / f"official_{s}_lipschitz_d8.obj")
        mesh.point_data["tree_n"] = np.loadtxt(MESHES / f"official_{s}_lipschitz_d8_tree_n.txt", dtype=float)
        path = OUT / f"mesh_{s}.png"
        size = (2600, 2100) if s == "monument" else (1600, 1300)
        rm.render(mesh, path, rm.VIEWS[s], scalars="tree_n", clim=(1, 20), size=size, cmap=BURGUNDY)
        panels.append(crop_transparent(path))
        path.unlink()
    # The label boxes of \Teaser in common.tex follow the widths of these panels.
    h, gap = 1270, 60
    panels = [p.resize((int(p.width * h / p.height), h), Image.LANCZOS) for p in panels]
    canvas = Image.new("RGBA", (sum(p.width for p in panels) + gap * (len(panels) - 1), h), (255, 255, 255, 0))
    x = 0
    for p in panels:
        canvas.paste(p, (x, 0), p)
        x += p.width + gap
    canvas.save(OUT / "teaser.png")
    print("teaser panel fractions:", [round((p.width + gap / 2 * (1 if i in (0, 2) else 2)) / canvas.width, 4)
                                      for i, p in enumerate(panels)])
    fig, ax = plt.subplots(figsize=(120 * MM, 10 * MM))
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=BURGUNDY, extent=[1, 20, 0, 1])
    ax.set_yticks([]); ax.set_xticks([1, 5, 10, 15, 20]); ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "colorbar.pdf", transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ plots
def scaling():
    fig, ax = plt.subplots(figsize=(150 * MM, 105 * MM))
    for k, ls in [("0", "-"), ("0.05", "--")]:
        for m in ["full", "lipschitz", "interval"]:
            ns, es = [], []
            for n in [10, 100, 1000, 10000]:
                d = load(f"scaling_n{n}_k{k}_{m}")
                if d:
                    ns.append(n); es.append(d["prim_evals"])
            if ns:
                ax.loglog(ns, es, ls, marker="o", color=COLORS[m])
    ax.set_xlabel("spheres"); ax.set_ylabel("primitive evaluations")
    fig.tight_layout(pad=0.4)
    fig.savefig(OUT / "scaling.pdf", transparent=True)
    plt.close(fig)


# ------------------------------------------------------------------ results block: GPU times, dynamic frame, coarse leaves
FRAME = "#c9748f"  # the dynamic frame in the strip of times
# GPU times at depth 8, in seconds: full tree, intervals, Lipschitz, dynamic frame. Measured in one
# session on 2026-11-15 (RTX 5070 Laptop on battery), with the GPU already prepared and the mesh
# kept on it: the median of the warm runs of build/lpm_mesh --gpu --repeat 3 (3 processes, 1 for
# the full tree) and the median frame of build/lpm_anim --method gpu-patch --keep-on-gpu over 48
# frames (3 runs on Monument). The same session gave the CAD table, the 107 ms per frame from
# scratch (lpm_anim --method gpu --keep-on-gpu) and the 123 fps of lpm_view --bench 300.
TIMES = {"Trees": [0.52, 0.048, 0.0269, 0.00268], "Monument": [23.5, 0.159, 0.103, 0.00678], "Molecule": [1.92, 0.074, 0.0344, 0.00421]}
TIME_NAMES = ["full tree", "intervals", "Lipschitz", "dynamic frame"]


def fmt_time(v):
    if v >= 1:
        return f"{v:.1f} s"
    ms = v * 1e3
    return f"{ms:.0f} ms" if ms >= 10 else f"{ms:.1f} ms"


def times():
    rc = {"font.family": "Montserrat", "font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
          "xtick.color": INK2, "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False,
          "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 1.0,
          "axes.axisbelow": True}
    colors = [COLORS["full"], COLORS["interval"], COLORS["lipschitz"], FRAME]
    with plt.rc_context(rc):
        fig, axes = plt.subplots(1, 3, figsize=(315 * MM, 57 * MM), sharey=True,
                                 gridspec_kw=dict(wspace=0.07, left=0.178, right=0.998, top=0.865, bottom=0.165))
        y = np.arange(4)[::-1]
        for ax, (scene, v) in zip(axes, TIMES.items()):
            for k, (yi, x, c) in enumerate(zip(y, v, colors)):
                ax.barh(yi, x - 1e-3, left=1e-3, height=0.62, color=c)  # the bars start at 1 ms
                label = fmt_time(x)
                if k == 2:  # the speedup over the full tree next to the time, as the paper prints it
                    label += f"  (×{v[0] / x:.0f})"
                ax.text(x * 1.18, yi, label, va="center", ha="left", color=INK, fontsize=17)
            ax.set_xscale("log")
            ax.set_xlim(1e-3, 400)
            ax.set_xticks([1e-3, 1e-2, 1e-1, 1, 10, 100])
            ax.set_xticklabels(["1 ms", "", "0.1 s", "", "10 s", ""], fontsize=15)
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_yticks(y)
            ax.set_yticklabels(TIME_NAMES, fontsize=17)
            ax.tick_params(axis="y", length=0)
            ax.grid(False, axis="y")
            ax.set_title(scene, fontsize=17, color=INK, loc="left", pad=6, fontweight="semibold")
            ax.set_ylim(-0.6, 3.6)
        fig.savefig(OUT / "times.pdf", transparent=True)
        plt.close(fig)


def load_mesh(path):
    path = str(path)
    if path.endswith(".ply"):  # binary PLY of write_ply: doubles, then (3, a, b, c) records
        b = open(path, "rb").read()
        h = b.index(b"end_header\n") + 11
        head = b[:h].decode()
        nv = int(head.split("element vertex ")[1].split()[0])
        nf = int(head.split("element face ")[1].split()[0])
        V = np.frombuffer(b, dtype="<f8", count=3 * nv, offset=h).reshape(-1, 3)
        rec = np.frombuffer(b, dtype=np.dtype([("n", "u1"), ("f", "<u4", 3)]), count=nf, offset=h + 24 * nv)
        return np.array(V), rec["f"].astype(np.int64)
    V, F = [], []
    with open(path) as fh:
        for line in fh:
            if line[0] == "v":
                V.append(line.split()[1:4])
            elif line[0] == "f":
                F.append(line.split()[1:4])
    return np.array(V, float), np.array(F, np.int64) - 1


def poly(V, F):
    return pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())


def shot(actors, out, cam, size):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    for mesh, kw in actors:
        p.add_mesh(mesh, **kw)
    if cam == "iso":
        p.view_isometric()
    else:
        p.camera_position = cam
    p.enable_anti_aliasing("ssaa")
    p.screenshot(str(out), transparent_background=False)
    cam = p.camera_position
    p.close()
    return cam


def crop_white(im, pad=10):
    """The image with its white made transparent, cropped to the rest plus pad pixels."""
    a = np.asarray(im.convert("RGBA")).copy()
    a[(a[:, :, 0] > 250) & (a[:, :, 1] > 250) & (a[:, :, 2] > 250), 3] = 0
    im = Image.fromarray(a)
    box = im.getbbox()
    return im.crop((max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, im.width), min(box[3] + pad, im.height)))


def dynamic(tmp, frame=6, frames=12):
    """A frame of lpm_anim gpu-patch: the kept quads come first in the faces, then the rebuilt ones."""
    d = tmp / "frames"
    d.mkdir()
    subprocess.run([str(ROOT / "build" / "lpm_anim"), "--scene", str(ROOT / "scenes" / "monument.json"), "--method", "gpu-patch",
                    "--frames", str(frames), "--depth", "8", "--stats", str(d / "stats.json"), "--out-dir", str(d)], check=True)
    st = json.load(open(d / "stats.json"))["frames_data"][frame]
    V, F = load_mesh(d / f"frame_{frame:03d}.obj")
    kept = 2 * st["patch_kept_quads"]  # the static quads that gpu-patch keeps
    assert len(F) == st["triangles"] and len(F) - kept == 2 * st["patch_new_quads"]
    cam = [(3.3, 2.5, 3.5), (0.0, 0.25, 0.0), (0, 1, 0)]  # the view of Monument of scripts/render_meshes.py
    raw = tmp / "dynamic.png"
    shot([(poly(V, F[:kept]), dict(color="#dedee2", smooth_shading=False, specular=0.1)),
          (poly(V, F[kept:]), dict(color=COLORS["lipschitz"], smooth_shading=False, specular=0.1))], raw, cam, (3000, 2300))
    crop_white(Image.open(raw)).save(OUT / "dynamic.png")
    print("dynamic frame", frame, "triangles", len(F), "rebuilt", len(F) - kept, "kept %.1f %%" % (100 * kept / len(F)))


def coarse(tmp, part="145", center=(3180, 2500), window=(1560, 700)):
    """The same view of a CAD part with the uniform mesh and with coarse leaves, edges drawn, and a detail of both."""
    W, H = 5200, 4160
    cam = "iso"
    for mode, extra in [("uniform", []), ("coarse", ["--adaptive-topdown", "--adaptive-tol", "0.005"])]:
        ply = tmp / f"{part}_{mode}.ply"
        subprocess.run([str(ROOT / "build" / "lpm_mesh"), "--scene", str(ROOT / "scenes" / "inversecsg" / f"{part}.json"), "--method",
                        "lipschitz", "--depth", "8", "--gpu", "--out", str(ply), "--stats", "/dev/null"] + extra, check=True)
        V, F = load_mesh(ply)
        raw = tmp / f"{mode}.png"
        cam = shot([(poly(V, F), dict(color="#f6e6ec", show_edges=True, edge_color=COLORS["lipschitz"], line_width=1.0,
                                      smooth_shading=False, specular=0.0, ambient=0.35, diffuse=0.75))], raw, cam, (W, H))
        im = Image.open(raw).convert("RGB")
        box = (center[0] - window[0] // 2, center[1] - window[1] // 2, center[0] + window[0] // 2, center[1] + window[1] // 2)
        im.crop(box).save(OUT / f"coarse_{mode}_zoom.png")
        if mode == "coarse":
            ImageDraw.Draw(im).rectangle(box, outline=INK, width=14)
            crop_white(im.resize((W // 4, H // 4), Image.LANCZOS), pad=6).save(OUT / "coarse_overview.png")
        print(part, mode, len(F), "triangles")


if __name__ == "__main__":
    steps = sys.argv[1:] or ["teaser", "plots", "times", "dynamic", "coarse"]
    unknown = set(steps) - {"teaser", "plots", "times", "dynamic", "coarse"}
    if unknown:
        sys.exit(f"unknown figures: {', '.join(sorted(unknown))}")
    setup_fonts()
    pv.OFF_SCREEN = True
    Image.MAX_IMAGE_PIXELS = None
    if "plots" in steps:
        scaling()
    if "times" in steps:
        times()
    with tempfile.TemporaryDirectory() as t:
        if "dynamic" in steps:
            dynamic(Path(t))
        if "coarse" in steps:
            coarse(Path(t))
    if "teaser" in steps:
        teaser()
    print("poster figures written to", OUT)
