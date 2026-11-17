"""Drag a sphere through a scene and watch the mesh follow: the dynamic update re-meshes
only the cells the sphere reaches, through lpm_server (on the GPU; --cpu runs it on the CPU).

    python scripts/interactive.py [trees|monument|molecule|scene.json] [depth] [--cpu]

Drag the blue wireframe sphere with the mouse (depth 7 by default; 8 is finer and slower).
Keys 1, 2, 3 switch between Trees, Monument and Molecule (a server per scene starts on demand).
"""
import struct
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pyvista as pv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from render_meshes import VIEWS  # noqa: E402

pv.OFF_SCREEN = False
RADIUS, K = 0.2, 0.1


class Server:
    def __init__(self, scene: Path, depth: int, gpu: bool):
        self.proc = subprocess.Popen([str(ROOT / "build" / "lpm_server"), "--scene", str(scene), "--depth", str(depth),
                                      "--radius", str(RADIUS), "--k", str(K)] + (["--gpu"] if gpu else []),
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=sys.stderr)

    def _read(self, n: int) -> bytearray:
        # A new writable buffer per frame: VTK keeps pointing to the arrays built on it.
        buf = bytearray(n)
        view, got = memoryview(buf), 0
        while got < n:
            k = self.proc.stdout.readinto(view[got:])
            if not k:
                raise RuntimeError("lpm_server stopped")
            got += k
        return buf

    def mesh(self, center):
        self.proc.stdin.write(f"{center[0]:.9g} {center[1]:.9g} {center[2]:.9g}\n".encode())
        self.proc.stdin.flush()
        magic, nv, nf, seconds, cells, statics = struct.unpack("<4sIIdII", self._read(28))
        assert magic == b"LPM1"
        V = np.frombuffer(self._read(12 * nv), dtype=np.float32).reshape(nv, 3)
        F = np.frombuffer(self._read(12 * nf), dtype=np.uint32).reshape(nf, 3)
        return V, F, seconds, cells, statics

    def close(self):
        try:
            self.proc.stdin.write(b"quit\n")
            self.proc.stdin.flush()
        except Exception:
            pass


SCENES = ["trees", "monument", "molecule"]


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    gpu = "--cpu" not in sys.argv[1:]
    name = args[0] if args else "trees"
    depth = int(args[1]) if len(args) > 1 else 7
    servers = {}
    state = {"scene": None, "poly": None, "text": None, "widget": None}
    plotter = pv.Plotter(window_size=(1500, 1000), title="Lipschitz pruning meshing: drag the sphere")
    plotter.set_background("white")

    def server_for(scene_name: str) -> Server:
        if scene_name not in servers:
            path = Path(scene_name) if scene_name.endswith(".json") else ROOT / "scenes" / f"{scene_name}.json"
            servers[scene_name] = Server(path, depth, gpu)
        return servers[scene_name]

    def update(c):
        c = np.asarray(c, float)
        t0 = time.perf_counter()
        V, F, seconds, cells, kept = server_for(state["scene"]).mesh(c)
        # Triangles as a fixed-size cell array on the indices as they arrive (no copy).
        tris = pv.CellArray.from_regular_cells(F.view(np.int32))
        if state["poly"] is None:
            # Empty first: PolyData(V) would also add a vertex cell per point, drawn as dots.
            state["poly"] = pv.PolyData()
            state["poly"].points = V
            state["poly"].SetPolys(tris)
            plotter.add_mesh(state["poly"], color="#cfd6de", specular=0.2, render=False)
        else:
            poly = state["poly"]
            poly.points = V
            poly.SetPolys(tris)
            poly.Modified()
        msg = (f"{state['scene']}, depth {depth}, {'GPU' if gpu else 'CPU'}   sphere at ({c[0]:.2f}, {c[1]:.2f}, {c[2]:.2f})   server {seconds * 1000:.0f} ms, "
               f"round trip {(time.perf_counter() - t0) * 1000:.0f} ms   region {cells} cells, {kept} static quads kept   "
               f"{len(F)} triangles\nkeys: 1 Trees, 2 Monument, 3 Molecule")
        if state["text"] is None:
            state["text"] = plotter.add_text(msg, position="upper_left", font_size=11, color="black", render=False)
        else:
            state["text"].SetText(2, msg)
        plotter.render()

    def show(scene_name: str):
        state["scene"] = scene_name
        view = VIEWS.get(scene_name, VIEWS["trees"])
        center = np.array(view["focal"], float)
        plotter.clear_sphere_widgets()
        update(center)
        state["widget"] = plotter.add_sphere_widget(update, center=center, radius=RADIUS, theta_resolution=24,
                                                   phi_resolution=24, style="wireframe", color="#2a78d6",
                                                   interaction_event="always")
        plotter.camera_position = [view["position"], view["focal"], view["up"]]
        plotter.render()

    for i, sc in enumerate(SCENES):
        plotter.add_key_event(str(i + 1), lambda sc=sc: show(sc))
    show(name)
    try:
        plotter.show()
    finally:
        for sv in servers.values():
            sv.close()


if __name__ == "__main__":
    main()
