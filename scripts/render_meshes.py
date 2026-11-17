"""Renders the meshes of results/meshes with PyVista (off screen).

    python scripts/render_meshes.py

Writes figures/mesh_<scene>.png (shaded), and figures/mesh_<scene>_tree.png and _grey.png (colored
by the size of the pruned tree of the cell of each vertex, in viridis and in greys).
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
MESHES = ROOT / "results" / "meshes"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
pv.OFF_SCREEN = True

VIEWS = {
    "trees": dict(position=(2.6, 1.9, 3.2), focal=(0.0, 0.7, 0.15), up=(0, 1, 0)),
    "monument": dict(position=(3.3, 2.5, 3.5), focal=(0.0, 0.25, 0.0), up=(0, 1, 0)),
    "molecule": dict(position=(2.2, 1.8, 2.6), focal=(0.0, 0.0, 0.0), up=(0, 1, 0)),
    "fluid": dict(position=(0.8, 0.6, 0.9), focal=(0.0, 0.0, 0.0), up=(0, 1, 0)),
}


def load_obj(path: Path) -> pv.PolyData:
    v, f = [], []
    with open(path) as fh:
        for line in fh:
            if line.startswith("v "):
                v.append([float(x) for x in line.split()[1:4]])
            elif line.startswith("f "):
                f.append([int(x.split("/")[0]) - 1 for x in line.split()[1:4]])
    V = np.asarray(v, float)
    F = np.asarray(f, np.int64)
    return pv.PolyData(V, np.hstack([np.full((len(F), 1), 3), F]).ravel())


def render(mesh: pv.PolyData, out: Path, view: dict, scalars=None, size=(1400, 1100), clim=None, edges=False, cmap="viridis"):
    p = pv.Plotter(off_screen=True, window_size=size)
    p.set_background("white")
    if scalars is None:
        p.add_mesh(mesh, color="#cfd6de", smooth_shading=False, specular=0.2, show_edges=edges, edge_color="#5a6270", line_width=0.3)
    else:
        p.add_mesh(mesh, scalars=scalars, cmap=cmap, clim=clim, smooth_shading=False, show_scalar_bar=False, log_scale=False)
    p.camera_position = [view["position"], view["focal"], view["up"]]
    p.enable_anti_aliasing("ssaa")
    p.screenshot(str(out), transparent_background=False)
    p.close()


def main():
    names = sys.argv[1:] or ["trees", "monument", "molecule"]
    for s in names:
        path = MESHES / f"official_{s}_lipschitz_d8.obj"
        attr = MESHES / f"official_{s}_lipschitz_d8_tree_n.txt"
        if s == "fluid":
            path = MESHES / "adverse_fluid_k0.05_lipschitz.obj"
            attr = None
        if not path.exists():
            print("missing", path)
            continue
        mesh = load_obj(path)
        view = VIEWS[s]
        render(mesh, FIG / f"mesh_{s}.png", view)
        if attr and attr.exists():
            tree_n = np.loadtxt(attr, dtype=float)
            mesh.point_data["tree_n"] = tree_n
            render(mesh, FIG / f"mesh_{s}_tree.png", view, scalars="tree_n", clim=(1, 20))
            render(mesh, FIG / f"mesh_{s}_grey.png", view, scalars="tree_n", clim=(1, 20), cmap="Greys")
            print(s, "tree_n: mean %.2f, median %d, max %d" % (tree_n.mean(), np.median(tree_n), tree_n.max()))
        print("rendered", s, mesh.n_points, "vertices")


if __name__ == "__main__":
    main()
