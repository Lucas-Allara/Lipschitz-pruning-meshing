"""Marching cubes (scikit-image, Lewiner) on a dense grid written by `lpm_mesh --dense`.

    python scripts/marching_cubes.py values.f32 scene.json DEPTH out.obj [stats.json]

Prints the extraction time; the discretization time is in the stats of lpm_mesh.
"""
import json
import sys
import time

import numpy as np
from skimage import measure


def main():
    values, scene_path, depth, out = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    stats_path = sys.argv[5] if len(sys.argv) > 5 else None
    n1 = (1 << depth) + 1
    vol = np.fromfile(values, dtype=np.float32).reshape(n1, n1, n1)
    with open(scene_path) as f:
        head = f.read(400)
    # The bounding box is at the top of the file; avoid parsing the whole scene.
    aabb_min = json.loads(head[head.index("[", head.index("aabb_min")):head.index("]", head.index("aabb_min")) + 1])
    aabb_max = json.loads(head[head.index("[", head.index("aabb_max")):head.index("]", head.index("aabb_max")) + 1])
    spacing = tuple((np.array(aabb_max) - np.array(aabb_min)) / (1 << depth))
    t0 = time.perf_counter()
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.0, spacing=spacing)
    seconds = time.perf_counter() - t0
    verts = verts + np.array(aabb_min)
    with open(out, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.9g} {v[1]:.9g} {v[2]:.9g}\n")
        for t in faces:
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")
    stats = {"time_mc": seconds, "vertices": int(len(verts)), "triangles": int(len(faces))}
    if stats_path:
        with open(stats_path, "w") as f:
            json.dump(stats, f)
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
