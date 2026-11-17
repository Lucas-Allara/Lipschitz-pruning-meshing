"""Renders the frames written by `lpm_anim --out-dir` into a GIF.

    python scripts/animate.py FRAME_DIR OUT.gif [scene] [width] [--grey] [--jobs J] [--duration MS] [--render WxH]

The scene name picks the camera of scripts/render_meshes.py (default: trees). With --grey the
frames are cropped to the box that contains the scene in every frame and share one grey
palette, so that each frame of the GIF stores only the pixels that change.
"""
import argparse
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import render_meshes as rm  # noqa: E402


def render_one(job):
    obj, scene, size = job
    png = obj.with_suffix(".png")
    rm.render(rm.load_obj(obj), png, rm.VIEWS[scene], size=size)
    return png


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("frame_dir", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("scene", nargs="?", default="trees")
    ap.add_argument("width", nargs="?", type=int, default=720)
    ap.add_argument("--grey", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--duration", type=int, default=80, help="milliseconds per frame")
    ap.add_argument("--render", default="1200x960", help="render size before resizing")
    a = ap.parse_args()
    size = tuple(int(v) for v in a.render.split("x"))
    files = sorted(a.frame_dir.glob("frame_*.obj"))
    jobs = [(f, a.scene, size) for f in files]
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            pngs = pool.map(render_one, jobs)
    else:
        pngs = [render_one(j) for j in jobs]
    if a.grey:
        ims = [Image.open(p).convert("L") for p in pngs]
        box = None
        for im in ims:
            ys, xs = np.where(np.asarray(im) < 250)
            b = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
            box = b if box is None else (min(box[0], b[0]), min(box[1], b[1]), max(box[2], b[2]), max(box[3], b[3]))
        pad = 12
        box = (max(box[0] - pad, 0), max(box[1] - pad, 0), min(box[2] + pad, ims[0].width), min(box[3] + pad, ims[0].height))
        h = round((box[3] - box[1]) * a.width / (box[2] - box[0]))
        frames = [im.crop(box).resize((a.width, h), Image.LANCZOS) for im in ims]
    else:
        frames = []
        for p in pngs:
            im = Image.open(p).convert("RGB")
            im = im.resize((a.width, int(im.height * a.width / im.width)), Image.LANCZOS)
            frames.append(im.quantize(colors=128, method=Image.Quantize.MEDIANCUT))
    frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=a.duration, loop=0, optimize=True)
    print("wrote", a.out, a.out.stat().st_size // 1024, "KB,", len(frames), "frames of", frames[0].size)


if __name__ == "__main__":
    main()
