"""Bees make 3D models on their own, and you edit them in Mnx Studio (a Blender-style editor at /studio.html).

  "make a 3d model of a sports car"      "create a low poly house"      "design a 3d robot for my game"
  "change my sports-car model: make it red and add a spoiler"

How a Bee does it (no steps for you):
  1. designs the model as parts (boxes, spheres, cylinders, cones, tori, capsules, planes), each with its size,
     place, turn, colour and finish; the design is a small JSON file, the same one Studio opens and saves
  2. builds a real .glb file from it (glTF 2.0, which Blender, Unity, Godot and three.js all open)
  3. saves both in its Vortex OS computer, ~/models/<name>.json and .glb
  4. puts a 3D viewer for it on a private /d/<id>/ link, opens it in its own browser and checks it loads
  5. tells you what it made; games it makes later can use the model ("make a racing game with my sports-car model")
"""

from __future__ import annotations

import json
import math
import re
import struct
from pathlib import Path

from . import brain, sites
from .core import Task

VIEWER = Path(__file__).resolve().parent / "templates" / "model_view"
SHAPES = ("box", "sphere", "cylinder", "cone", "torus", "capsule", "plane")
MAX_PARTS = 160
FOLDER = "models"

MAKE = re.compile(r"\b(?:make|create|build|design|sculpt|model|generate|craft|draw)\b[^.?!]*?"
                  r"(?:\b3[\s-]?d\s+(?:model|asset|object|mesh|version)s?\b|\b(?:low|high)[\s-]?poly\b|\bglb\b|\bgltf\b|\bmesh\b|"
                  r"\bblender\s+model\b|\b3[\s-]?d\s+(?!game|world|website|site|app|web|chart|graph|plot|print)[a-z]+)", re.I)
CHANGE = re.compile(r"\b(?:change|edit|update|improve|fix|add|remove|make)\b[^.?!]*\b(?:my|the)\s+([\w-]+(?:[\s-][\w-]+){0,2}?)\s+(?:3d\s+)?model\b", re.I)
OPEN_STUDIO = re.compile(r"\b(?:open|show|launch)\s+(?:the\s+)?(?:mnx\s+)?(?:studio|blender|3d\s+editor|model\s+editor)\b", re.I)

SPEC = """You design 3D models out of simple parts. Answer with ONE JSON object and nothing else:

{"name": "sports car", "parts": [
  {"name": "body", "shape": "box", "size": [1.9, 0.45, 4.2], "pos": [0, 0.55, 0], "rot": [0, 0, 0], "color": "#d1202a", "metal": 0.6, "rough": 0.35},
  {"name": "wheel front left", "shape": "cylinder", "size": [0.7, 0.3, 0.7], "pos": [-0.95, 0.35, 1.3], "rot": [0, 0, 90], "color": "#151515", "rough": 0.9},
  {"name": "headlight left", "shape": "sphere", "size": [0.25, 0.15, 0.08], "pos": [-0.6, 0.6, 2.1], "color": "#ffffff", "glow": 1.5}
]}

Rules
- Shapes: box, sphere, cylinder, cone, torus, capsule, plane. "size" is the part's width, height and depth in metres
  (x, y, z) before turning; "pos" is its centre; "rot" is degrees around x, y, z. A cylinder or cone stands upright
  (round in x and z); turn it 90° around z for a wheel. A torus lies flat (a ring around the y axis). A plane is flat
  and faces up.
- y is up, the model stands on the ground (lowest point at y = 0), its front faces +z, real-world size in metres.
- Optional: "metal" and "rough" (0..1), "glow" (0..3, the part shines in its colour), "opacity" (0..1 for glass).
- Use 12 to 120 parts: enough to look like the thing, with details (windows, lights, doors, handles, eyes …).
- If the user names a famous design or character, make an original model in that style; copy no logos or text.
"""


# ---------------------------------------------------------------- the design (JSON) ----------
def _nums(v, n: int, default: list[float], lo: float, hi: float) -> list[float]:
    try:
        vals = [float(x) for x in (v if isinstance(v, (list, tuple)) else [v] * n)][:n]
    except (TypeError, ValueError):
        return list(default)
    vals += default[len(vals):]
    return [round(min(hi, max(lo, x)), 4) if math.isfinite(x) else d for x, d in zip(vals, default)]


def _color(v) -> str:
    v = str(v or "").strip()
    if re.fullmatch(r"#?[0-9a-fA-F]{6}", v):
        return "#" + v.lstrip("#").lower()
    if re.fullmatch(r"#?[0-9a-fA-F]{3}", v):
        return "#" + "".join(c * 2 for c in v.lstrip("#")).lower()
    return "#9aa0a6"


def clean(spec: dict) -> dict:
    """A design anyone could have written (the model, Studio, a person): only known fields, sane numbers."""
    if not isinstance(spec, dict):
        raise ValueError("a 3D design is a JSON object with parts")
    parts = []
    for i, p in enumerate((spec.get("parts") or [])[:MAX_PARTS]):
        if not isinstance(p, dict):
            continue
        shape = str(p.get("shape", "box")).lower()
        shape = {"cube": "box", "ball": "sphere", "cyl": "cylinder", "pyramid": "cone", "ring": "torus", "donut": "torus",
                 "pill": "capsule", "quad": "plane"}.get(shape, shape)
        if shape not in SHAPES:
            shape = "box"
        part = {"name": str(p.get("name") or f"{shape} {i + 1}")[:60], "shape": shape,
                "size": _nums(p.get("size"), 3, [1, 1, 1], 0.001, 500),
                "pos": _nums(p.get("pos"), 3, [0, 0, 0], -1000, 1000),
                "rot": _nums(p.get("rot"), 3, [0, 0, 0], -720, 720),
                "color": _color(p.get("color"))}
        for k, lo, hi, d in (("metal", 0, 1, 0.0), ("rough", 0, 1, 0.6), ("glow", 0, 3, 0.0), ("opacity", 0.05, 1, 1.0)):
            val = _nums(p.get(k, d), 1, [d], lo, hi)[0]
            if val != d:
                part[k] = val
        parts.append(part)
    if not parts:
        raise ValueError("the design has no parts")
    return {"name": str(spec.get("name") or "model")[:60], "parts": parts}


def parse_spec(raw: str) -> dict:
    """The JSON object in the model's answer (it may wrap it in ``` or say a word first)."""
    raw = re.sub(r"```(?:json)?", "", raw or "")
    start = raw.find("{")
    if start < 0:
        raise ValueError("no design in the answer")
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        depth += ch == "{"
        depth -= ch == "}"
        if depth == 0:
            text = re.sub(r",\s*([}\]])", r"\1", raw[start:i + 1])  # trailing commas, a common slip
            return clean(json.loads(text))
    raise ValueError("the design was cut off")


# ---------------------------------------------------------------- unit shapes (fit in a 1×1×1 box) ----------
def _box():
    pos, nor, idx = [], [], []
    for axis in range(3):
        for sign in (1, -1):
            n = [0, 0, 0]
            n[axis] = sign
            u, v = [(1, 2), (2, 0), (0, 1)][axis]
            base = len(pos)
            for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                p = [0.0, 0.0, 0.0]
                p[axis] = 0.5 * sign
                p[u], p[v] = 0.5 * a, 0.5 * b
                pos.append(p)
                nor.append(n)
            idx += [base, base + 1, base + 2, base, base + 2, base + 3] if sign > 0 else [base, base + 2, base + 1, base, base + 3, base + 2]
    return pos, nor, idx


def _lathe(profile: list[tuple[float, float, float, float]], seg: int = 24):
    """Spin a profile of (radius, y, normal-r, normal-y) points around the y axis."""
    pos, nor, idx = [], [], []
    for r, y, nr, ny in profile:
        for s in range(seg + 1):
            a = 2 * math.pi * s / seg
            c, si = math.cos(a), math.sin(a)
            pos.append([r * si, y, r * c])
            nor.append([nr * si, ny, nr * c])
    row = seg + 1
    for j in range(len(profile) - 1):
        for s in range(seg):
            a, b = j * row + s, (j + 1) * row + s
            idx += [a, a + 1, b, b, a + 1, b + 1]
    return pos, nor, idx


def _join(*meshes):
    pos, nor, idx = [], [], []
    for p, n, i in meshes:
        off = len(pos)
        pos += p
        nor += n
        idx += [x + off for x in i]
    return pos, nor, idx


def _disc(y: float, up: bool, r: float = 0.5, seg: int = 24):
    pos, nor = [[0, y, 0]], [[0, 1 if up else -1, 0]]
    for s in range(seg + 1):
        a = 2 * math.pi * s / seg
        pos.append([r * math.sin(a), y, r * math.cos(a)])
        nor.append([0, 1 if up else -1, 0])
    idx = []
    for s in range(1, seg + 1):
        idx += [0, s, s + 1] if up else [0, s + 1, s]
    return pos, nor, idx


def _sphere(rings: int = 16):
    prof = []
    for j in range(rings + 1):
        t = math.pi * j / rings
        prof.append((0.5 * math.sin(t), -0.5 * math.cos(t), math.sin(t), -math.cos(t)))
    return _lathe(prof)


def _cylinder():
    side = _lathe([(0.5, -0.5, 1, 0), (0.5, 0.5, 1, 0)])
    return _join(side, _disc(0.5, True), _disc(-0.5, False))


def _cone():
    k = 1 / math.sqrt(1 + 0.25)  # the side's normal leans up by the slope
    side = _lathe([(0.5, -0.5, k, 0.5 * k), (0.0, 0.5, k, 0.5 * k)])
    return _join(side, _disc(-0.5, False))


def _capsule(rings: int = 8):
    r, half = 0.25, 0.25  # 0.5 wide, 1 tall
    prof = []
    for j in range(rings + 1):
        t = math.pi / 2 * j / rings
        prof.append((r * math.sin(t), -half - r * math.cos(t), math.sin(t), -math.cos(t)))
    for j in range(rings + 1):
        t = math.pi / 2 * j / rings
        prof.append((r * math.cos(t), half + r * math.sin(t), math.cos(t), math.sin(t)))
    return _lathe(prof)


def _torus(seg: int = 32, tube_seg: int = 12):
    R, r = 0.4, 0.1  # 1 wide, 0.2 thick
    pos, nor, idx = [], [], []
    for i in range(seg + 1):
        a = 2 * math.pi * i / seg
        for j in range(tube_seg + 1):
            b = 2 * math.pi * j / tube_seg
            cx, cz = math.sin(a), math.cos(a)
            pos.append([(R + r * math.cos(b)) * cx, r * math.sin(b), (R + r * math.cos(b)) * cz])
            nor.append([math.cos(b) * cx, math.sin(b), math.cos(b) * cz])
    row = tube_seg + 1
    for i in range(seg):
        for j in range(tube_seg):
            a, b = i * row + j, (i + 1) * row + j
            idx += [a, b, a + 1, a + 1, b, b + 1]
    return pos, nor, idx


def _plane():
    return [[-0.5, 0, -0.5], [0.5, 0, -0.5], [0.5, 0, 0.5], [-0.5, 0, 0.5]], [[0, 1, 0]] * 4, [0, 2, 1, 0, 3, 2]


UNIT = {"box": _box, "sphere": _sphere, "cylinder": _cylinder, "cone": _cone, "capsule": _capsule, "torus": _torus, "plane": _plane}
# The unit shape's own size, so a part's "size" is its real width, height and depth.
NATURAL = {"capsule": (0.5, 1, 0.5), "torus": (1, 0.2, 1), "plane": (1, 1, 1)}


def _quat(deg: list[float]) -> list[float]:
    """three.js Euler XYZ (degrees) as a glTF quaternion [x, y, z, w]."""
    x, y, z = (math.radians(d) / 2 for d in deg)
    c1, c2, c3, s1, s2, s3 = math.cos(x), math.cos(y), math.cos(z), math.sin(x), math.sin(y), math.sin(z)
    return [s1 * c2 * c3 + c1 * s2 * s3, c1 * s2 * c3 - s1 * c2 * s3, c1 * c2 * s3 + s1 * s2 * c3, c1 * c2 * c3 - s1 * s2 * s3]


def _linear(hex_color: str) -> list[float]:
    out = []
    for i in (1, 3, 5):
        c = int(hex_color[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4)
    return out


def to_glb(spec: dict) -> bytes:
    """The design as a binary glTF 2.0 file: one shared mesh per shape, one node per part."""
    spec = clean(spec)
    blob = bytearray()
    views, accessors, meshes, materials, nodes = [], [], [], [], []
    geo: dict[str, tuple[int, int, int]] = {}
    mesh_of: dict[tuple, int] = {}

    def view(data: bytes, target: int) -> int:
        while len(blob) % 4:
            blob.append(0)
        views.append({"buffer": 0, "byteOffset": len(blob), "byteLength": len(data), "target": target})
        blob.extend(data)
        return len(views) - 1

    for part in spec["parts"]:
        shape = part["shape"]
        if shape not in geo:
            pos, nor, idx = UNIT[shape]()
            flat_p = [c for p in pos for c in p]
            flat_n = []
            for n in nor:
                ln = math.sqrt(sum(c * c for c in n)) or 1
                flat_n += [c / ln for c in n]
            pv = view(struct.pack(f"<{len(flat_p)}f", *flat_p), 34962)
            nv = view(struct.pack(f"<{len(flat_n)}f", *flat_n), 34962)
            iv = view(struct.pack(f"<{len(idx)}H", *idx), 34963)
            accessors.append({"bufferView": pv, "componentType": 5126, "count": len(pos), "type": "VEC3",
                              "min": [min(p[k] for p in pos) for k in range(3)], "max": [max(p[k] for p in pos) for k in range(3)]})
            accessors.append({"bufferView": nv, "componentType": 5126, "count": len(nor), "type": "VEC3"})
            accessors.append({"bufferView": iv, "componentType": 5123, "count": len(idx), "type": "SCALAR"})
            geo[shape] = (len(accessors) - 3, len(accessors) - 2, len(accessors) - 1)
        look = (part["color"], part.get("metal", 0.0), part.get("rough", 0.6), part.get("glow", 0.0), part.get("opacity", 1.0))
        if (shape, look) not in mesh_of:
            color, metal, rough, glow, opacity = look
            mat = {"name": part["name"], "pbrMetallicRoughness": {"baseColorFactor": _linear(color) + [opacity],
                                                                   "metallicFactor": metal, "roughnessFactor": rough},
                   "doubleSided": shape == "plane"}
            if glow:
                lin = _linear(color)
                mat["emissiveFactor"] = [min(1.0, c * min(glow, 1.0)) for c in lin]
                if glow > 1:
                    mat.setdefault("extensions", {})["KHR_materials_emissive_strength"] = {"emissiveStrength": glow}
            if opacity < 1:
                mat["alphaMode"] = "BLEND"
            materials.append(mat)
            p_acc, n_acc, i_acc = geo[shape]
            meshes.append({"name": f"{shape}-{len(meshes)}", "primitives": [
                {"attributes": {"POSITION": p_acc, "NORMAL": n_acc}, "indices": i_acc, "material": len(materials) - 1}]})
            mesh_of[(shape, look)] = len(meshes) - 1
        nat = NATURAL.get(shape, (1, 1, 1))
        size = part["size"]
        scale = [size[0] / nat[0], (size[1] if shape != "plane" else 1) / nat[1], size[2] / nat[2]]
        nodes.append({"name": part["name"], "mesh": mesh_of[(shape, look)], "translation": part["pos"],
                      "rotation": _quat(part["rot"]), "scale": scale})

    gltf = {"asset": {"version": "2.0", "generator": "Mnx Hive models3d"}, "scene": 0,
            "scenes": [{"name": spec["name"], "nodes": [len(nodes)]}],
            "nodes": nodes + [{"name": spec["name"], "children": list(range(len(nodes)))}],
            "meshes": meshes, "materials": materials, "accessors": accessors, "bufferViews": views,
            "buffers": [{"byteLength": len(blob) + (-len(blob) % 4)}]}
    if any("extensions" in m for m in materials):
        gltf["extensionsUsed"] = ["KHR_materials_emissive_strength"]
    js = json.dumps(gltf, separators=(",", ":")).encode()
    js += b" " * (-len(js) % 4)
    blob += b"\0" * (-len(blob) % 4)
    total = 12 + 8 + len(js) + 8 + len(blob)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob))


def bounds(spec: dict) -> tuple[list[float], list[float]]:
    """Roughly where the design sits (each part's box, not turned): for its size in words and the ground check."""
    lo, hi = [1e9] * 3, [-1e9] * 3
    for p in spec["parts"]:
        for k in range(3):
            lo[k] = min(lo[k], p["pos"][k] - p["size"][k] / 2)
            hi[k] = max(hi[k], p["pos"][k] + p["size"][k] / 2)
    return lo, hi


def size_words(spec: dict) -> str:
    lo, hi = bounds(spec)
    return " × ".join(f"{hi[k] - lo[k]:.1f}" for k in (0, 1, 2)) + " m (w × h × d)"


# ---------------------------------------------------------------- ready-made designs (no model running) ----------
def _p(name, shape, size, pos, color, rot=(0, 0, 0), **extra):
    return {"name": name, "shape": shape, "size": list(size), "pos": list(pos), "rot": list(rot), "color": color, **extra}


def _car():
    parts = [_p("body", "box", (1.9, 0.5, 4.2), (0, 0.6, 0), "#d1202a", metal=0.6, rough=0.35),
             _p("cabin", "box", (1.6, 0.5, 2.0), (0, 1.1, -0.2), "#d1202a", metal=0.6, rough=0.35),
             _p("windscreen", "box", (1.5, 0.42, 0.06), (0, 1.1, 0.82), "#9fd3ff", rot=(-25, 0, 0), opacity=0.5, metal=0.2, rough=0.05),
             _p("rear window", "box", (1.5, 0.4, 0.06), (0, 1.1, -1.22), "#9fd3ff", rot=(25, 0, 0), opacity=0.5, rough=0.05),
             _p("bumper front", "box", (1.95, 0.2, 0.15), (0, 0.4, 2.12), "#222222"),
             _p("bumper rear", "box", (1.95, 0.2, 0.15), (0, 0.4, -2.12), "#222222"),
             _p("spoiler", "box", (1.7, 0.06, 0.35), (0, 1.0, -1.95), "#111111")]
    for side, x in (("left", -0.95), ("right", 0.95)):
        for end, z in (("front", 1.35), ("rear", -1.35)):
            parts.append(_p(f"wheel {end} {side}", "cylinder", (0.72, 0.32, 0.72), (x, 0.36, z), "#141414", rot=(0, 0, 90), rough=0.9))
            parts.append(_p(f"rim {end} {side}", "cylinder", (0.42, 0.34, 0.42), (x, 0.36, z), "#c0c4c8", rot=(0, 0, 90), metal=0.9, rough=0.25))
        parts.append(_p(f"headlight {side}", "sphere", (0.3, 0.16, 0.08), (x * 0.65, 0.65, 2.11), "#fff6d8", glow=1.5))
        parts.append(_p(f"tail light {side}", "box", (0.35, 0.12, 0.05), (x * 0.65, 0.7, -2.11), "#ff2a2a", glow=1.2))
        parts.append(_p(f"mirror {side}", "box", (0.18, 0.12, 0.1), (x * 0.9, 1.05, 0.6), "#d1202a"))
    return {"name": "sports car", "parts": parts}


def _house():
    parts = [_p("walls", "box", (6, 3, 5), (0, 1.5, 0), "#f1e3c8"), _p("roof", "cone", (8.2, 2.2, 7), (0, 4.1, 0), "#9c3b2e", rot=(0, 45, 0)),
             _p("door", "box", (1, 2, 0.08), (0, 1, 2.52), "#6b3f1f"), _p("knob", "sphere", (0.1, 0.1, 0.1), (0.35, 1, 2.58), "#d4af37", metal=1, rough=0.3),
             _p("chimney", "box", (0.6, 1.6, 0.6), (1.8, 4.6, -0.8), "#7a4a3a"), _p("step", "box", (1.6, 0.2, 0.6), (0, 0.1, 2.8), "#9e9e9e")]
    for i, x in enumerate((-1.9, 1.9)):
        parts.append(_p(f"window front {i + 1}", "box", (1.1, 1, 0.06), (x, 1.7, 2.52), "#9fd3ff", opacity=0.7, rough=0.1))
        parts.append(_p(f"window frame {i + 1}", "box", (1.25, 1.15, 0.04), (x, 1.7, 2.5), "#ffffff"))
    parts.append(_p("lawn", "plane", (10, 1, 9), (0, 0.001, 0.5), "#5fa83a"))
    return {"name": "house", "parts": parts}


def _tree():
    return {"name": "tree", "parts": [_p("trunk", "cylinder", (0.35, 2.2, 0.35), (0, 1.1, 0), "#6b4423", rough=0.9),
                                      _p("leaves low", "sphere", (2.2, 1.6, 2.2), (0, 2.6, 0), "#2f8f3a"),
                                      _p("leaves mid", "sphere", (1.7, 1.4, 1.7), (0.2, 3.4, 0.1), "#38a845"),
                                      _p("leaves top", "sphere", (1.1, 1.0, 1.1), (-0.1, 4.1, 0), "#45bf52")]}


def _robot():
    parts = [_p("body", "box", (1, 1.2, 0.6), (0, 1.5, 0), "#c9ced6", metal=0.8, rough=0.3),
             _p("head", "box", (0.7, 0.55, 0.55), (0, 2.45, 0), "#c9ced6", metal=0.8, rough=0.3),
             _p("antenna", "cylinder", (0.04, 0.35, 0.04), (0, 2.9, 0), "#555555"),
             _p("antenna tip", "sphere", (0.12, 0.12, 0.12), (0, 3.1, 0), "#ff3b3b", glow=1.5),
             _p("chest light", "sphere", (0.25, 0.25, 0.05), (0, 1.7, 0.31), "#38d6ff", glow=2)]
    for side, x in (("left", -1), ("right", 1)):
        parts += [_p(f"eye {side}", "sphere", (0.14, 0.14, 0.05), (x * 0.17, 2.5, 0.28), "#38d6ff", glow=2),
                  _p(f"arm {side}", "capsule", (0.22, 1.0, 0.22), (x * 0.68, 1.45, 0), "#8d939c", metal=0.7, rough=0.4),
                  _p(f"leg {side}", "capsule", (0.28, 0.95, 0.28), (x * 0.25, 0.48, 0), "#8d939c", metal=0.7, rough=0.4)]
    return {"name": "robot", "parts": parts}


def _rocket():
    parts = [_p("body", "cylinder", (1, 4, 1), (0, 2.6, 0), "#f2f2f2", metal=0.3, rough=0.3),
             _p("nose", "cone", (1, 1.4, 1), (0, 5.3, 0), "#e3342f"),
             _p("window", "sphere", (0.45, 0.45, 0.1), (0, 3.6, 0.5), "#9fd3ff", glow=0.4),
             _p("engine", "cone", (0.8, 0.6, 0.8), (0, 0.45, 0), "#444444", rot=(180, 0, 0), metal=0.8),
             _p("flame", "cone", (0.5, 0.8, 0.5), (0, -0.1, 0), "#ffb020", rot=(180, 0, 0), glow=2.5, opacity=0.85)]
    for i in range(4):
        a = math.radians(90 * i)
        parts.append(_p(f"fin {i + 1}", "box", (0.08, 1.1, 0.8), (math.sin(a) * 0.65, 1.1, math.cos(a) * 0.65), "#e3342f", rot=(0, 90 * i + 90, 0)))
    return {"name": "rocket", "parts": parts}


PRESETS = {"car": _car, "house": _house, "home": _house, "tree": _tree, "robot": _robot, "rocket": _rocket, "spaceship": _rocket}


def preset_for(text: str) -> dict | None:
    for word, make in PRESETS.items():
        if re.search(rf"\b{word}s?\b", text, re.I):
            return make()
    return None


# ---------------------------------------------------------------- the Bee's computer: ~/models/ ----------
def _cell(b: dict):
    from .cells import cells
    return cells.get(b["id"], b["name"], b["computer"])


async def save(b: dict, name: str, spec: dict) -> tuple[str, bytes]:
    spec = clean(spec)
    glb = to_glb(spec)
    cell = _cell(b)
    for ext, data in (("json", json.dumps(spec, indent=1).encode()), ("glb", glb)):
        rel = f"{FOLDER}/{name}.{ext}"
        if cell.mode == "remote":
            await cell.rwrite(rel, data)
        else:
            p = cell.safe_path(rel)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
    return f"~/{FOLDER}/{name}.glb", glb


async def load(b: dict, name: str) -> dict | None:
    cell = _cell(b)
    rel = f"{FOLDER}/{sites.slug(name)}.json"
    try:
        raw = await cell.rread(rel) if cell.mode == "remote" else cell.safe_path(rel).read_bytes()
        return clean(json.loads(raw))
    except Exception:  # not there, or not a design
        return None


async def names(b: dict) -> list[str]:
    cell = _cell(b)
    if cell.mode == "remote":
        try:
            rows = await cell.rlist(FOLDER)
        except Exception:
            return []
        return sorted(r["name"][:-5] for r in rows if str(r.get("name", "")).endswith(".json"))
    root = cell.safe_path(FOLDER)
    return sorted(p.stem for p in root.glob("*.json")) if root.is_dir() else []


async def glb_of(b: dict, name: str) -> bytes | None:
    spec = await load(b, name)
    return to_glb(spec) if spec else None


# ---------------------------------------------------------------- in the chat ----------
EXPLICIT = re.compile(r"\b3[\s-]?d\s+(?:model|asset|object|mesh)s?\b|\bmodel\s+of\b|\b(?:low|high)[\s-]?poly\b|\bglb\b|\bgltf\b|\bmesh\b", re.I)


def parse(text: str) -> dict | None:
    t = " ".join((text or "").split())
    if OPEN_STUDIO.search(t):
        return {"open": True, "goal": t}
    if re.search(r"\bgames?\b", t, re.I):  # a game is the games' work; a model "for my game", or named outright, is ours
        uses = re.search(r"\b(?:with|using|use|from)\s+(?:my|the|these|those)\b[^.?!]*\bmodels?\b", t, re.I)
        for_game = re.search(r"\b(?:for|in|into)\s+(?:my|the|a|an|our)\s+(?:[\w-]+\s+)?game\b", t, re.I)
        if uses or not (EXPLICIT.search(t) or for_game):
            return None
    change = CHANGE.search(t)
    if change and not re.search(r"\b(?:ai|ml|language|llm|neural|trained?)\s+model\b", t, re.I):
        return {"goal": t, "change": sites.slug(change.group(1))}
    if MAKE.search(t) and not re.search(r"\b(?:train|fine[\s-]?tune|ai\s+model|language\s+model|llm|3d\s+print(?:er|ing)?)\b", t, re.I):
        return {"goal": t}
    return None


def name_for(goal: str) -> str:
    words = re.sub(r"\b(?:make|create|build|design|sculpt|model|models|generate|craft|draw|me|a|an|the|my|of|in|for|3d|3-d|"
                   r"low|high|poly|asset|object|mesh|glb|gltf|blender|please|with|and|game|that|i|can|use|it|some)\b", " ", goal.lower())
    return sites.slug(" ".join(words.split()[:3])) or "model"


async def design(task: Task, goal: str, current: dict | None = None) -> dict:
    ask = (f"The model now:\n{json.dumps(current)}\n\nChange it: {goal}\nAnswer with the whole new design." if current
           else f"Design this 3D model: {goal}")
    raw = await brain.brain.chat([{"role": "system", "content": SPEC}, {"role": "user", "content": ask}],
                                 task.effort, max_tokens=8000, temperature=0.4)
    return parse_spec(raw)


def viewer_files(name: str, glb: bytes) -> dict[str, str | bytes]:
    files: dict[str, str | bytes] = {p.name: p.read_text() for p in sorted(VIEWER.iterdir()) if p.is_file()}
    files["index.html"] = files["index.html"].replace("{{name}}", re.sub(r"[<>&\"]", "", name))
    files["model.glb"] = glb
    return files


async def handle(task: Task, intent: dict) -> None:
    from . import games, sandbox
    from .apps import _shot_id
    b = games._pick_bee(task)
    if intent.get("open"):
        await task.answer("**[Open Mnx Studio](/studio.html" + (f"?bee={b['id']}" if b else "") + ")**: add shapes, move, turn and "
                          "scale them (G, R, S like Blender), colour them, and save to your Bee's computer; or tell a Bee what to change.")
        return
    if not b:
        await task.answer("A Bee makes the model on its own computer. Make one first (New Bee in Peace, e.g. “Maker: makes my 3D models”), then ask again.")
        return
    bee = b["name"]
    current = await load(b, intent["change"]) if intent.get("change") else None
    name = intent["change"] if current else name_for(intent["goal"])
    did, note = [], []
    if brain.is_connected():
        s = await task.step("writing", f"Designing the {name.replace('-', ' ')} model" if not current else f"Changing {name}", bee=bee)
        try:
            spec = await design(task, intent["goal"], current)
            await task.finish_step(s, text=f"{len(spec['parts'])} parts")
            did.append(f"{'Changed' if current else 'Designed'} it from {len(spec['parts'])} parts")
        except (ValueError, brain.BrainNotConnected) as exc:
            await task.finish_step(s, status="error", text=str(exc)[:160])
            spec = current or preset_for(intent["goal"])
            if not spec:
                await task.answer(f"The model's design didn't come out right ({exc}). Ask again, or open **[Mnx Studio](/studio.html?bee={b['id']})** and build it there.")
                return
            note.append("The model's design didn't come out whole, so this is a ready-made one; ask for changes.")
    else:
        spec = preset_for(intent["goal"])
        if not spec:
            await task.answer("No model is running on this Hive, so the Bee can't design that yet. Ready-made ones it can make now: "
                              "a car, house, tree, robot or rocket. Or build it yourself in **[Mnx Studio](/studio.html?bee=" + b["id"] + ")**.")
            return
        note.append("No model is running on this Hive, so this is a ready-made design; change it in Studio.")
        did.append(f"Took the ready-made {spec['name']} ({len(spec['parts'])} parts)")
    spec["name"] = name.replace("-", " ")

    s = await task.step("running", "Building the .glb file and saving it", bee=bee)
    where, glb = await save(b, name, spec)
    await task.finish_step(s, text=f"{len(glb) // 1024 or 1} KB")
    did.append(f"Built {name}.glb ({len(glb) // 1024 or 1} KB, glTF 2.0) and saved it with its design at {where}")

    url, result = "", {}
    s = await task.step("running", "Putting a 3D viewer on a private link and checking it", bee=bee)
    try:
        await sites.chroma.ensure()
        d = await sites.chroma.deploy(task.user, f"{name}-model", viewer_files(spec["name"], glb), [], private=True)
        url = d["url"]
        result = await games.play(url)
        if result.get("skipped"):
            await task.finish_step(s, text=f"Not checked: {result['skipped']}")
        elif result["ok"]:
            await task.finish_step(s, text="It loads: no errors")
            did.append("Opened it in its browser: it loads with no errors")
        else:
            await task.finish_step(s, status="error", text=(result["errors"] or ["the viewer didn't finish loading"])[0][:160])
    except Exception as exc:  # the sites server: the model is saved either way
        await task.finish_step(s, status="error", text=str(exc)[:200])
        note.append(f"The viewer didn't go up ({str(exc)[:120]}), but the model is saved.")
    if url:
        did.append("Put a 3D viewer for it on a private link")
        sandbox.record(task.user, "model", name, bee, url, studio=f"/studio.html?bee={b['id']}&model={name}")
    sid = _shot_id(task.user, result.get("shot") or b"", "image/jpeg")
    if sid:
        await task.emit("screen", bee=bee, title=f"{name} in {bee}'s browser", shot=sid, live=url)
    studio = f"/studio.html?bee={b['id']}&model={name}"
    await task.emit("result", bee=bee, service="3D model", title=f"{name} is ready", badge="Private",
                    details=[["View", url or "not deployed"], ["Edit in Studio", studio], ["File", f"{where} (glTF 2.0: Blender, Unity, Godot, three.js)"],
                             ["Size", size_words(spec)], ["Parts", str(len(spec["parts"]))]])
    lines = "\n".join(f"- {x}" for x in did)
    await task.answer(f"{bee} made **{name}**" + (f": **[view it]({url})**" if url else "") + f" · **[edit it in Studio]({studio})**\n\nWhat I did:\n{lines}\n\n"
                      + (" ".join(note) + " " if note else "") + f"Use it in a game: “make a racing game with my {name} model”.")
