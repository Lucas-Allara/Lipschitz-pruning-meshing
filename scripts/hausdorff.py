"""Symmetric Hausdorff and mean distances between two meshes.

    python scripts/hausdorff.py a.obj b.obj [samples]

Distances are measured from points sampled on each surface (plus its vertices) to the
closest point of the other mesh.
"""
import json
import sys

import numpy as np
import trimesh


def one_sided(src: trimesh.Trimesh, dst: trimesh.Trimesh, samples: int, seed: int):
    pts = src.vertices
    if samples > 0:
        extra, _ = trimesh.sample.sample_surface(src, samples, seed=seed)
        pts = np.vstack([pts, extra])
    _, d, _ = trimesh.proximity.closest_point(dst, pts)
    return float(d.max()), float(d.mean())


def hausdorff(path_a: str, path_b: str, samples: int = 20000):
    a = trimesh.load(path_a, process=False)
    b = trimesh.load(path_b, process=False)
    hab, mab = one_sided(a, b, samples, 1)
    hba, mba = one_sided(b, a, samples, 2)
    return {"hausdorff": max(hab, hba), "mean": 0.5 * (mab + mba), "a_to_b": hab, "b_to_a": hba,
            "vertices_a": int(len(a.vertices)), "vertices_b": int(len(b.vertices))}


if __name__ == "__main__":
    samples = int(sys.argv[3]) if len(sys.argv) > 3 else 20000
    print(json.dumps(hausdorff(sys.argv[1], sys.argv[2], samples)))
