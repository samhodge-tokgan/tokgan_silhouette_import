"""Converter tests (standard library only: python -m unittest discover -s tests).

``fixtures/v3_real_cut.json`` is cut from a real Rotobot Next 0.10.0 run: two
body parts over 24 4K frames with the real (projective) camera and pelvis.

The Silhouette conventions the hierarchy test re-composes with were checked
in Silhouette on that shot (2026-10-08): shapes and layer positions are
centre-origin, Y-down, divided by the image height; the corner pin is in
image fractions; rotation is degrees clockwise; a parent layer's pin carries
down to its children.
"""
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
SCRIPT = os.path.join(REPO, "data", "tokgan_json_to_fxs.py")
V3 = os.path.join(HERE, "fixtures", "v3_real_cut.json")


def convert(src, *args):
    out = os.path.join(tempfile.mkdtemp(), "out.fxs")
    r = subprocess.run([sys.executable, SCRIPT, src, out, *args],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise AssertionError(r.stdout + r.stderr)
    return ET.parse(out).getroot(), r.stdout


def solve_h(src, dst):
    a = []
    for (x, y), (u, v) in zip(src, dst):
        a.append([x, y, 1, 0, 0, 0, -u * x, -u * y, u])
        a.append([0, 0, 0, x, y, 1, -v * x, -v * y, v])
    for c in range(8):
        p = max(range(c, 8), key=lambda r: abs(a[r][c]))
        a[c], a[p] = a[p], a[c]
        for r in range(8):
            if r != c:
                k = a[r][c] / a[c][c]
                a[r] = [x - k * y for x, y in zip(a[r], a[c])]
    return [a[i][8] / a[i][i] for i in range(8)] + [1.0]


def tup(text):
    return tuple(float(v) for v in text.strip().strip("()").split(","))


def keys(el, pid):
    for prop in el.find("Properties"):
        if prop.get("id") == pid:
            return {int(k.get("frame")): k for k in prop.findall("Key")}
    return {}


def children(el):
    for prop in el.find("Properties"):
        if prop.get("id") == "objects":
            return list(prop)
    return []


class TestHierarchy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(V3, encoding="utf-8") as fh:
            cls.src = json.load(fh)
        cls.root, cls.stdout = convert(V3, "--frame-offset", "0")

    def test_tree(self):
        (camera,) = list(self.root)
        self.assertEqual(camera.get("label"), "camera")
        (person,) = children(camera)
        self.assertEqual(person.get("label"), "p0")
        self.assertEqual(sorted(p.get("label") for p in children(person)),
                         ["p0_arm_L_forearm", "p0_leg_R_thigh"])
        self.assertIn("camera (corner pin)", self.stdout)

    def test_every_vertex_lands_on_the_source(self):
        w, h = self.src["resolution"]
        (camera,) = list(self.root)
        pins = {c: keys(camera, f"transform.pin_{c}") for c in ("ul", "ur", "lr", "ll")}
        (person,) = children(camera)
        pelvis = keys(person, "transform.position")
        worst, checked = 0.0, 0
        for part in children(person):
            obj = self.src["objects"][part.get("label").replace("_", ":", 3)]
            pos, rot = keys(part, "transform.position"), keys(part, "transform.rotate")
            (shape,) = children(part)
            for f, key in keys(shape, "path").items():
                hm = solve_h([(0, 0), (w, 0), (w, h), (0, h)],
                             [(u * w, v * h) for u, v in
                              (tup(pins[c][f].text) for c in ("ul", "ur", "lr", "ll"))])
                px, py = tup(pelvis[f].text)
                tx, ty = tup(pos[f].text)
                a = math.radians(float(rot[f].text))
                for pt, want in zip(key.iter("Point"), obj["frames"][str(f)]["points"]):
                    lx, ly = (v * h for v in tup(pt.text))
                    sx = px * h + w / 2 + tx * h + math.cos(a) * lx - math.sin(a) * ly
                    sy = py * h + h / 2 + ty * h + math.sin(a) * lx + math.cos(a) * ly
                    d = hm[6] * sx + hm[7] * sy + hm[8]
                    x = (hm[0] * sx + hm[1] * sy + hm[2]) / d
                    y = (hm[3] * sx + hm[4] * sy + hm[5]) / d
                    worst = max(worst, math.hypot(x - want["x"], y - want["y"]))
                    checked += 1
        self.assertGreater(checked, 0)
        self.assertLess(worst, 0.05, f"worst {worst:.4f} px over {checked} vertices")

    def test_no_hierarchy_flag(self):
        root, _ = convert(V3, "--frame-offset", "0", "--no-hierarchy", "--layers")
        self.assertNotIn("camera", [e.get("label") for e in root])


class TestV2(unittest.TestCase):
    def test_v2_is_unchanged_flat_output(self):
        with open(V3, encoding="utf-8") as fh:
            data = json.load(fh)
        for k in ("camera", "persons"):
            data.pop(k)
        data["schema_version"] = 2
        src = os.path.join(tempfile.mkdtemp(), "v2.json")
        with open(src, "w", encoding="utf-8") as fh:
            json.dump(data, fh)
        root, _ = convert(src)
        self.assertEqual({e.get("type") for e in root}, {"Shape"})


if __name__ == "__main__":
    unittest.main()
