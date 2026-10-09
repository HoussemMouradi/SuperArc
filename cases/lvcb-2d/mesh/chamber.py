"""Parametric planar circuit-breaker chamber used by the Gmsh mesh.

All public coordinates are millimetres. Boxes are (name, x, y, z, dx, dy, dz).
The z extent is the device depth used to scale planar current to amperes.
"""

from __future__ import annotations

import json
from pathlib import Path

SPEC_PATH = Path(__file__).with_name("chamber_spec.json")


def load_spec(path: Path | None = None) -> dict:
    return json.loads((path or SPEC_PATH).read_text())


def moving_tip_top(spec: dict) -> float:
    return spec["moving_tip_closed"] - spec["contact_gap"]


def _box(name: str, x0: float, x1: float, y0: float, y1: float, z0: float, z1: float):
    return (name, x0, y0, z0, x1 - x0, y1 - y0, z1 - z0)


def copper_boxes(spec: dict) -> list[tuple]:
    """Conductors, including the terminal stubs that leave the rear wall."""
    z0, z1 = spec["contact_z"]
    dz0, dz1 = 0.0, spec["depth"]
    ur = spec["upper_runner"]
    lr = spec["lower_runner"]
    ft = spec["fixed_tip"]
    tip_top = moving_tip_top(spec)
    stub = spec["terminal_stub"] + spec["wall"]
    return [
        _box("upper_runner", ur[0], ur[1], ur[2], ur[3], dz0, dz1),
        _box("upper_stub", -stub, ur[0], ur[2], ur[3], dz0, dz1),
        _box("fixed_tip", ft[0], ft[1], ft[2], ft[3], z0, z1),
        _box("lower_runner", lr[0], lr[1], lr[2], lr[3], dz0, dz1),
        _box("lower_stub", -stub, lr[0], lr[2], lr[3], dz0, dz1),
        _box("moving_tip", ft[0], ft[1], spec["moving_base"], tip_top, z0, z1),
    ]


def plate_boxes(spec: dict) -> list[tuple]:
    z0, z1 = spec["plate_z"]
    return [_box(f"plate_{i+1}", *rect, z0, z1) for i, rect in enumerate(spec["plates"])]


def solid_boxes(spec: dict) -> list[tuple]:
    return copper_boxes(spec) + plate_boxes(spec)


def cavity_box(spec: dict) -> tuple:
    return _box("cavity", 0.0, spec["Lx"], 0.0, spec["Ly"], 0.0, spec["depth"])


def housing_outer_box(spec: dict) -> tuple:
    w = spec["wall"]
    return _box("housing_outer", -w, spec["Lx"] + w, -w, spec["Ly"] + w, -w, spec["depth"] + w)


def boxes_inside_cavity(spec: dict) -> list[tuple]:
    """Solids clipped to the inner cavity (what the gas domain is cut by)."""
    _, cx, cy, cz, cdx, cdy, cdz = cavity_box(spec)
    out = []
    for name, x, y, z, dx, dy, dz in solid_boxes(spec):
        x0, y0, z0 = max(x, cx), max(y, cy), max(z, cz)
        x1 = min(x + dx, cx + cdx)
        y1 = min(y + dy, cy + cdy)
        z1 = min(z + dz, cz + cdz)
        if x1 > x0 and y1 > y0 and z1 > z0:
            out.append(_box(name, x0, x1, y0, y1, z0, z1))
    return out


def section_rects(spec: dict, z: float | None = None) -> list[tuple]:
    """Mid-plane rectangles (name, x0, x1, y0, y1) in mm. The mesh uses these."""
    z = spec["depth"] / 2 if z is None else z
    rects = []
    for name, x, y, zz, dx, dy, dz in boxes_inside_cavity(spec):
        if zz < z < zz + dz:
            rects.append((name, x, x + dx, y, y + dy))
    return rects


def gas_area_mm2(spec: dict) -> float:
    area = spec["Lx"] * spec["Ly"]
    for _name, x0, x1, y0, y1 in section_rects(spec):
        area -= (x1 - x0) * (y1 - y0)
    return area


def rects_overlap(a, b, tol: float = 1e-9) -> bool:
    return a[0] < b[1] - tol and b[0] < a[1] - tol and a[2] < b[3] - tol and b[2] < a[3] - tol


def boxes_overlap(a, b, tol: float = 1e-6) -> bool:
    """True when two (name, x, y, z, dx, dy, dz) boxes share volume."""
    return all(a[i] < b[i] + b[i + 3] - tol and b[i] < a[i] + a[i + 3] - tol for i in (1, 2, 3))


def overlapping_solids(spec: dict) -> list[tuple[str, str]]:
    boxes = solid_boxes(spec)
    hits = []
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if boxes_overlap(boxes[i], boxes[j]):
                hits.append((boxes[i][0], boxes[j][0]))
    return hits
