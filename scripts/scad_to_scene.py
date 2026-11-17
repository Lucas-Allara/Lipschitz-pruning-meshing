"""OpenSCAD CSG tree (.csg, as `openscad -o model.csg model.scad` writes it) to a scene of
lpm (the JSON of the official scenes), scaled into [-1, 1]:

    python scripts/scad_to_scene.py model.csg scene.json

cube, sphere and cylinder become box, sphere and cylinder; a cylinder with r1 != r2 becomes a
cone, cut by a slab when truncated. multmatrix is composed down to the primitives; scaling is
folded into their sizes, so it must be uniform (or along the axes of a box or of a cylinder,
keeping its section round). union, difference and intersection become sharp operators
(blendRadius 0), in balanced trees.
"""
import json
import re
import sys

import numpy as np


# ---------------------------------------------------------------------------- parser
TOKEN = re.compile(r"\s*(?:(//[^\n]*)|([A-Za-z_$][A-Za-z_0-9$]*)|(-?\d+\.?\d*(?:[eE][-+]?\d+)?|-?\.\d+(?:[eE][-+]?\d+)?)|(\"[^\"]*\")|(.))")


def tokens(text):
    for m in TOKEN.finditer(text):
        comment, name, number, string, sym = m.groups()
        if comment:
            continue
        if name is not None:
            yield ("name", name)
        elif number is not None:
            yield ("num", float(number))
        elif string is not None:
            yield ("str", string[1:-1])
        elif sym is not None and sym.strip():
            yield ("sym", sym)


class Parser:
    def __init__(self, text):
        self.t = list(tokens(text))
        self.i = 0

    def peek(self):
        return self.t[self.i] if self.i < len(self.t) else (None, None)

    def take(self, kind=None, value=None):
        tok = self.peek()
        if kind and tok[0] != kind or value is not None and tok[1] != value:
            raise ValueError(f"expected {kind} {value}, got {tok}")
        self.i += 1
        return tok

    def value(self):
        kind, v = self.peek()
        if kind == "sym" and v == "[":
            self.take()
            items = []
            while self.peek() != ("sym", "]"):
                items.append(self.value())
                if self.peek() == ("sym", ","):
                    self.take()
            self.take("sym", "]")
            return items
        if kind == "name" and v in ("true", "false"):
            self.take()
            return v == "true"
        self.take()
        return v

    def statement(self):
        _, name = self.take("name")
        self.take("sym", "(")
        args, pos = {}, []
        while self.peek() != ("sym", ")"):
            if self.peek()[0] == "name" and self.t[self.i + 1] == ("sym", "="):
                key = self.take()[1]
                self.take("sym", "=")
                args[key] = self.value()
            else:
                pos.append(self.value())
            if self.peek() == ("sym", ","):
                self.take()
        self.take("sym", ")")
        children = []
        if self.peek() == ("sym", "{"):
            self.take()
            while self.peek() != ("sym", "}"):
                children.append(self.statement())
            self.take("sym", "}")
        elif self.peek() == ("sym", ";"):
            self.take()
        return {"name": name, "args": args, "pos": pos, "children": children}

    def program(self):
        out = []
        while self.peek()[0] is not None:
            out.append(self.statement())
        return out


# ---------------------------------------------------------------------------- conversion
def translate(v):
    m = np.eye(4)
    m[:3, 3] = v
    return m


ROT_Y_TO_Z = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], float)   # local y -> object z
ROT_Y_TO_MZ = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], float)  # local y -> object -z


TOL = 1e-4  # OpenSCAD writes the matrices with 6 significant digits


def frame(W):
    """Rotation (or reflection), per-axis scale and translation of an affine map; None if skewed.
    The rotation is the nearest orthogonal matrix (polar decomposition), so that the primitives
    stay distance fields despite the rounding of the file."""
    A, t = W[:3, :3], W[:3, 3]
    G = A.T @ A
    s = np.sqrt(np.diag(G))
    if np.max(np.abs(G - np.diag(np.diag(G)))) > TOL * np.max(np.diag(G)):
        return None
    U, _, Vt = np.linalg.svd(A / s)
    return U @ Vt, s, t


def placed(prim, W):
    """The primitive with the matrix world -> primitive (row-major 3x4) of the rigid part of W."""
    R, _, t = frame(W)
    L = np.zeros((3, 4))
    L[:, :3] = R.T
    L[:, 3] = -R.T @ t
    prim["matrix"] = [float(x) for x in L.reshape(-1)]
    return prim


class Unsupported(Exception):
    pass


def box(W, sides):
    f = frame(W)
    if f is None:
        raise Unsupported("skewed box")
    return placed({"nodeType": "primitive", "primitiveType": "box", "sides": [float(a * b) for a, b in zip(sides, f[1])],
                   "bevel": [0.0, 0.0, 0.0, 0.0]}, W)


def op(mode, a, b):
    return {"nodeType": "binaryOperator", "blendMode": mode, "blendRadius": 0.0, "leftChild": a, "rightChild": b}


def balanced(mode, items):
    if not items:
        return None
    if len(items) == 1:
        return items[0]
    h = len(items) // 2
    return op(mode, balanced(mode, items[:h]), balanced(mode, items[h:]))


def convert(node, W):
    """The solid of a statement under the object -> world map W, or None when empty."""
    name, a, kids = node["name"], node["args"], node["children"]
    if name == "multmatrix":
        M = np.array(node["pos"][0] if node["pos"] else a["m"], float)
        return balanced("union", [s for s in (convert(c, W @ M) for c in kids) if s is not None])
    if name in ("group", "union", "render", "color"):
        return balanced("union", [s for s in (convert(c, W) for c in kids) if s is not None])
    if name == "difference":
        parts = [convert(c, W) for c in kids]
        if not parts or parts[0] is None:
            return None
        rest = [p for p in parts[1:] if p is not None]
        return parts[0] if not rest else op("sub", parts[0], balanced("union", rest))
    if name == "intersection":
        parts = [convert(c, W) for c in kids]
        if any(p is None for p in parts) or not parts:
            return None
        out = parts[0]
        for p in parts[1:]:
            out = op("inter", out, p)
        return out
    if name == "cube":
        size = a.get("size", node["pos"][0] if node["pos"] else 1.0)
        size = [float(size)] * 3 if not isinstance(size, list) else [float(x) for x in size]
        Wb = W if a.get("center", False) else W @ translate(np.array(size) / 2)
        return box(Wb, size)
    if name == "sphere":
        r = float(a.get("r", a.get("d", 2.0) / 2 if "d" in a else 1.0))
        f = frame(W)
        if f is None or np.ptp(f[1]) > TOL * f[1].max():
            raise Unsupported("non-uniform scale of a sphere")
        return placed({"nodeType": "primitive", "primitiveType": "sphere", "radius": r * float(f[1][0])}, W)
    if name == "cylinder":
        h = float(a.get("h", 1.0))
        r = a.get("r", None)
        r1 = float(a.get("r1", r if r is not None else 1.0))
        r2 = float(a.get("r2", r if r is not None else 1.0))
        z0 = -h / 2 if a.get("center", False) else 0.0
        f = frame(W)
        if f is None or abs(f[1][0] - f[1][1]) > TOL * f[1].max():
            raise Unsupported("non-round section of a cylinder")
        sr, sh = float(f[1][0]), float(f[1][2])
        if abs(r1 - r2) <= 1e-12 * max(r1, r2, 1.0):
            Wc = W @ translate([0, 0, z0 + h / 2]) @ ROT_Y_TO_Z
            return placed({"nodeType": "primitive", "primitiveType": "cylinder", "height": h * sh, "radius": r1 * sr}, Wc)
        # A cone of base max(r1, r2) whose apex lies where the radius vanishes; cut by a slab
        # when truncated.
        up = r1 > r2
        rb, rt = (r1, r2) if up else (r2, r1)
        H = h * rb / (rb - rt)
        if up:
            Wc = W @ translate([0, 0, z0 + H / 2]) @ ROT_Y_TO_Z
        else:
            Wc = W @ translate([0, 0, z0 + h - H / 2]) @ ROT_Y_TO_MZ
        cone = placed({"nodeType": "primitive", "primitiveType": "cone", "radius": rb * sr, "height": H * sh}, Wc)
        if rt <= 1e-12:
            return cone
        slab = box(W @ translate([0, 0, z0 + h / 2]), [2.2 * rb, 2.2 * rb, h])
        return op("inter", cone, slab)
    raise Unsupported(name)


def bounds(n):
    """Bounding box of a solid: union of the children, the first of a difference, the overlap of an intersection."""
    if n["nodeType"] == "binaryOperator":
        a, b = bounds(n["leftChild"]), bounds(n["rightChild"])
        if n["blendMode"] == "union":
            return np.minimum(a[0], b[0]), np.maximum(a[1], b[1])
        if n["blendMode"] == "sub":
            return a
        return np.maximum(a[0], b[0]), np.minimum(a[1], b[1])
    L = np.array(n["matrix"]).reshape(3, 4)
    R, t = L[:, :3].T, -L[:, :3].T @ L[:, 3]  # primitive -> world
    if n["primitiveType"] == "sphere":
        e = np.full(3, n["radius"])
    elif n["primitiveType"] == "box":
        e = np.abs(R) @ (np.array(n["sides"]) / 2)
        return t - e, t + e
    else:
        r, hh = n["radius"], n["height"] / 2
        e = np.abs(R) @ np.array([r, hh, r])
    return t - e, t + e


def main():
    src, dst = sys.argv[1], sys.argv[2]
    prog = Parser(open(src).read()).program()
    root = balanced("union", [s for s in (convert(st, np.eye(4)) for st in prog) if s is not None])
    lo, hi = bounds(root)
    # Scaled into [-1, 1] around the center, with a margin of 5 % of the size.
    c, s = 0.5 * (lo + hi), 2.0 / np.max(hi - lo)
    N = np.diag([s, s, s, 1.0]) @ translate(-c)
    root = balanced("union", [x for x in (convert(st, N) for st in prog) if x is not None])
    lo, hi = bounds(root)
    pad = 0.05 * np.max(hi - lo)
    scene = {"aabb_min": [float(x - pad) for x in lo], "aabb_max": [float(x + pad) for x in hi],
             "source": {"file": src.split("/")[-1], "center": [float(x) for x in c], "scale": float(s)}}  # p_scene = scale (p - center)
    scene.update(root)
    json.dump(scene, open(dst, "w"))


if __name__ == "__main__":
    try:
        main()
    except Unsupported as e:
        print(f"unsupported: {e}", file=sys.stderr)
        sys.exit(2)
