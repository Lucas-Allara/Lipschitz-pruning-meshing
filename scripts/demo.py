"""Interactive viewer of the meshes of results/meshes, colored by the size of the pruned
tree of the cell of each vertex.

    python scripts/demo.py [trees|monument|molecule] [--meshes DIR]

DIR holds official_<scene>_lipschitz_d8.obj and _tree_n.txt (default results/meshes), for
example the output of lpm_mesh --gpu.

Keys: 1, 2, 3 switch the scene; c toggles the coloring; the mouse rotates and zooms.
"""
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
MESHES = ROOT / "results" / "meshes"
sys.path.insert(0, str(ROOT / "scripts"))
from render_meshes import VIEWS, load_obj  # noqa: E402

pv.OFF_SCREEN = False  # render_meshes sets it for its own off-screen renders

SCENES = ["trees", "monument", "molecule"]
NODES = {"trees": 371, "monument": 6323, "molecule": 2001}


class Demo:
    def __init__(self, scene: str):
        self.plotter = pv.Plotter(window_size=(1500, 1000), title="Lipschitz pruning meshing")
        self.plotter.set_background("white")
        self.colored = True
        self.meshes = {}
        self.actor = None
        self.text = None
        self.scene = scene
        for i, s in enumerate(SCENES):
            self.plotter.add_key_event(str(i + 1), lambda s=s: self.show(s))
        self.plotter.add_key_event("c", self.toggle)
        self.show(scene)

    def mesh(self, s: str) -> pv.PolyData:
        if s not in self.meshes:
            m = load_obj(MESHES / f"official_{s}_lipschitz_d8.obj")
            m.point_data["active nodes"] = np.loadtxt(MESHES / f"official_{s}_lipschitz_d8_tree_n.txt", dtype=float)
            self.meshes[s] = m
        return self.meshes[s]

    def show(self, s: str):
        self.scene = s
        m = self.mesh(s)
        if self.actor is not None:
            self.plotter.remove_actor(self.actor)
        if self.text is not None:
            self.plotter.remove_actor(self.text)
        if self.colored:
            self.actor = self.plotter.add_mesh(m, scalars="active nodes", cmap="viridis", clim=(1, 20),
                                               scalar_bar_args={"title": "active nodes per cell", "color": "black"})
        else:
            self.actor = self.plotter.add_mesh(m, color="#cfd6de", specular=0.2)
        v = VIEWS[s]
        self.plotter.camera_position = [v["position"], v["focal"], v["up"]]
        a = m.point_data["active nodes"]
        self.text = self.plotter.add_text(
            f"{s}: {NODES[s]} nodes in the CSG tree, {m.n_points} vertices, {m.n_cells} triangles\n"
            f"active nodes per vertex cell: mean {a.mean():.1f}, median {int(np.median(a))}, max {int(a.max())}\n"
            "keys: 1 Trees, 2 Monument, 3 Molecule, c color on/off", position="upper_left", font_size=11, color="black")
        self.plotter.render()

    def toggle(self):
        self.colored = not self.colored
        self.show(self.scene)


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--meshes" in args:
        i = args.index("--meshes")
        MESHES = Path(args[i + 1])
        del args[i:i + 2]
    scene = args[0] if args else "monument"
    Demo(scene).plotter.show()
