"""Planar Elmer mesh and one-cell-thick OpenFOAM mesh of the open-contact chamber.

Units in the meshes are metres. Geometry comes from chamber_spec.json next to this file.
Run from anywhere:

    python3 cases/lvcb-2d/mesh/make_mesh.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import gmsh

sys.path.insert(0, str(Path(__file__).resolve().parent))

from chamber import load_spec, section_rects  # noqa: E402

MM = 1e-3
# One prism through the thickness. OpenFOAM treats front/back as empty, so the
# case is planar; the real 10 mm depth only scales the current in the circuit.
THICKNESS = 0.5e-3
# 0.4 mm on the arc route (contact gap, runners, splitter channels); 0.8 mm elsewhere.
LC_FINE = 0.4e-3
LC = 0.8e-3


def _refine_arc_route():
    """Background size field. Point sizes are ignored so the box actually wins."""
    field = gmsh.model.mesh.field
    field.add("Box", 1)
    field.setNumber(1, "VIn", LC_FINE)
    field.setNumber(1, "VOut", LC)
    field.setNumber(1, "XMin", 6e-3)
    field.setNumber(1, "XMax", 48e-3)
    field.setNumber(1, "YMin", 2e-3)
    field.setNumber(1, "YMax", 20e-3)
    field.setNumber(1, "Thickness", 2e-3)
    field.setAsBackgroundMesh(1)
    gmsh.option.setNumber("Mesh.MeshSizeExtendFromBoundary", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromPoints", 0)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 0)
    gmsh.option.setNumber("Mesh.MeshSizeMin", LC_FINE)
    gmsh.option.setNumber("Mesh.MeshSizeMax", LC)


def _rects(spec):
    return section_rects(spec)


def _classify_body(name: str) -> tuple[int, str]:
    if name == "gas":
        return 1, "gas"
    if name in ("upper_runner", "fixed_tip"):
        return 2, "copper_fixed"
    if name in ("lower_runner", "moving_tip"):
        return 3, "copper_moving"
    if name.startswith("plate_"):
        return 3 + int(name.split("_")[1]), name
    raise KeyError(name)


def build_elmer_mesh(path: Path, spec) -> None:
    gmsh.initialize()
    gmsh.model.add("lvcb-2d")
    occ = gmsh.model.occ
    lx, ly = spec["Lx"] * MM, spec["Ly"] * MM
    gas = occ.addRectangle(0, 0, 0, lx, ly)
    tools = []
    names = []
    for name, x0, x1, y0, y1 in _rects(spec):
        tools.append(occ.addRectangle(x0 * MM, y0 * MM, 0, (x1 - x0) * MM, (y1 - y0) * MM))
        names.append(name)
    occ.fragment([(2, gas)], [(2, t) for t in tools])
    occ.synchronize()

    groups: dict[int, list[int]] = {}
    for dim, tag in gmsh.model.getEntities(2):
        cx, cy, _cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        # The gas surface is multiply connected; its centre of mass can fall
        # inside a hole, so the large surface is identified by its area.
        body = 1 if gmsh.model.occ.getMass(dim, tag) > 2e-4 else None
        if body is None:
            for name, x0, x1, y0, y1 in _rects(spec):
                if x0 * MM < cx < x1 * MM and y0 * MM < cy < y1 * MM:
                    body = _classify_body(name)[0]
                    break
            if body is None:
                raise RuntimeError(f"unclassified surface at ({cx}, {cy})")
        groups.setdefault(body, []).append(tag)
    for body, tags in groups.items():
        label = "gas" if body == 1 else f"body{body}"
        gmsh.model.addPhysicalGroup(2, tags, body, name=label)

    # Boundary curves belong to a single surface. Split the vent from the walls.
    vent0, vent1 = (v * MM for v in spec["vent_y"])
    runner_u = spec["upper_runner"]
    runner_l = spec["lower_runner"]
    boundary_groups = {k: [] for k in ("terminal_plus", "terminal_zero", "inlet", "outlet", "wall")}
    for dim, tag in gmsh.model.getEntities(1):
        up, _down = gmsh.model.getAdjacencies(dim, tag)
        if len(up) != 1:
            continue
        cx, cy, _cz = gmsh.model.occ.getCenterOfMass(dim, tag)
        tol = 1e-6
        if abs(cx - 0.0) < tol and runner_u[2] * MM - tol < cy < runner_u[3] * MM + tol:
            boundary_groups["terminal_plus"].append(tag)
        elif abs(cx - 0.0) < tol and runner_l[2] * MM - tol < cy < runner_l[3] * MM + tol:
            boundary_groups["terminal_zero"].append(tag)
        elif abs(cx - 0.0) < tol and vent0 - tol < cy < vent1 + tol:
            boundary_groups["inlet"].append(tag)
        elif abs(cx - lx) < tol and vent0 - tol < cy < vent1 + tol:
            boundary_groups["outlet"].append(tag)
        else:
            boundary_groups["wall"].append(tag)
    ids = {"terminal_plus": 1, "terminal_zero": 2, "inlet": 3, "outlet": 4, "wall": 5}
    for name, tags in boundary_groups.items():
        if tags:
            gmsh.model.addPhysicalGroup(1, tags, ids[name], name=name)

    _refine_arc_route()
    gmsh.option.setNumber("Mesh.MshFileVersion", 2.2)
    gmsh.model.mesh.generate(2)
    path.parent.mkdir(parents=True, exist_ok=True)
    gmsh.write(str(path))
    write_elmer_directory(path.parent.parent / "em" / "work" / "elmermesh", groups, boundary_groups, ids)
    print(f"Elmer mesh {path}: " + ", ".join(f"{k} {len(v)}" for k, v in boundary_groups.items()))
    gmsh.finalize()


def write_elmer_directory(directory: Path, surface_groups, boundary_groups, boundary_ids) -> None:
    """Write mesh.header/nodes/elements/boundary from the current Gmsh model."""
    directory.mkdir(parents=True, exist_ok=True)
    node_tags, coords, _ = gmsh.model.mesh.getNodes()
    node_tags = [int(t) for t in node_tags]
    flat = [float(v) for v in coords]
    points = [flat[i:i + 3] for i in range(0, len(flat), 3)]
    order = sorted(range(len(node_tags)), key=node_tags.__getitem__)
    node_tags = [node_tags[i] for i in order]
    points = [points[i] for i in order]
    tag_to_id = {tag: i + 1 for i, tag in enumerate(node_tags)}

    elements = []  # body, nodes
    for body, surfaces in surface_groups.items():
        for surface in surfaces:
            types, _etags, nodes = gmsh.model.mesh.getElements(2, surface)
            for etype, enodes in zip(types, nodes):
                nper = 3 if int(etype) == 2 else None
                if nper is None:
                    raise RuntimeError(f"expected linear triangles, got gmsh element type {etype}")
                enodes = [int(n) for n in enodes]
                for k in range(0, len(enodes), nper):
                    elements.append((body, [tag_to_id[n] for n in enodes[k:k + nper]]))

    edge_parent = {}
    for i, (_body, nodes) in enumerate(elements, start=1):
        a, b, c = nodes
        for edge in ((a, b), (b, c), (c, a)):
            edge_parent[frozenset(edge)] = i
    boundaries = []  # bc id, parent element, nodes
    for name, curves in boundary_groups.items():
        bc = boundary_ids[name]
        for curve in curves:
            types, _etags, nodes = gmsh.model.mesh.getElements(1, curve)
            for etype, enodes in zip(types, nodes):
                if int(etype) != 1:
                    continue
                enodes = [int(n) for n in enodes]
                for k in range(0, len(enodes), 2):
                    pair = [tag_to_id[n] for n in enodes[k:k + 2]]
                    boundaries.append((bc, edge_parent.get(frozenset(pair), 0), pair))

    # Elmer nodes are: id, constraint (-1 = none), x, y, z
    (directory / "mesh.nodes").write_text("".join(
        f"{i+1} -1 {points[i][0]:.8e} {points[i][1]:.8e} {points[i][2]:.8e}\n" for i in range(len(node_tags))))
    elements_text = "".join(
        f"{i+1} {body} 303 {n[0]} {n[1]} {n[2]}\n" for i, (body, n) in enumerate(elements))
    (directory / "mesh.elements").write_text(elements_text)
    # assign_bins rewrites mesh.elements; this copy is restored at the start of every run.
    (directory / "mesh.elements.orig").write_text(elements_text)
    # id, boundary id, left parent, right parent, element type 202, nodes
    (directory / "mesh.boundary").write_text("".join(
        f"{i+1} {bc} {parent} 0 202 {n[0]} {n[1]}\n" for i, (bc, parent, n) in enumerate(boundaries)))
    (directory / "mesh.header").write_text(
        f"{len(node_tags)} {len(elements)} {len(boundaries)}\n2\n303 {len(elements)}\n202 {len(boundaries)}\n")
    print(f"Elmer directory {directory}: {len(node_tags)} nodes, {len(elements)} triangles, {len(boundaries)} edges")


def _foam(obj: str, cls: str, body: str) -> str:
    return (
        "FoamFile\n{\n    version     2.0;\n    format      ascii;\n"
        f"    class       {cls};\n    object      {obj};\n}}\n{body}"
    )


def write_of_polymesh(case: Path, spec) -> None:
    """One prism layer of the gas triangulation. Gmsh 4.12 subdivides extrusions
    into tetrahedra, which cannot carry an `empty` front/back, so the polyMesh
    is written directly."""
    mesh_dir = case / "constant" / "polyMesh"
    elmer = case.parent / "em" / "work" / "elmermesh"
    nodes = {}
    for line in (elmer / "mesh.nodes").read_text().splitlines():
        p = line.split()
        nums = [float(v) for v in p[1:]]
        # id, constraint, x, y, z   or   id, x, y, z
        x, y = (nums[-3], nums[-2]) if len(nums) >= 4 else (nums[0], nums[1])
        nodes[int(p[0])] = (x, y)
    tris = []
    for line in (elmer / "mesh.elements").read_text().splitlines():
        p = line.split()
        if int(p[1]) == 1:
            a, b, c = (int(p[3]), int(p[4]), int(p[5]))
            ax, ay = nodes[a]
            bx, by = nodes[b]
            cx, cy = nodes[c]
            if (bx - ax) * (cy - ay) - (by - ay) * (cx - ax) < 0:
                b, c = c, b
            tris.append((a, b, c))

    used = sorted({n for tri in tris for n in tri})
    remap = {n: i for i, n in enumerate(used)}
    nxy = len(used)
    points = [nodes[n] for n in used] + [nodes[n] for n in used]
    tris = [tuple(remap[n] for n in tri) for tri in tris]

    # edge (lo, hi) -> [(cell, a, b)] with a->b following the cell orientation
    edges: dict[tuple[int, int], list] = {}
    for cell, (a, b, c) in enumerate(tris):
        for u, v in ((a, b), (b, c), (c, a)):
            key = (u, v) if u < v else (v, u)
            edges.setdefault(key, []).append((cell, u, v))

    lx = spec["Lx"] * MM
    vent0, vent1 = (v * MM for v in spec["vent_y"])
    internal = []  # (owner, neighbour, verts)
    boundary = {k: [] for k in ("inlet", "outlet", "walls", "front", "back")}
    for (_key, hits) in edges.items():
        if len(hits) == 2:
            (c0, a0, b0), (c1, _a1, _b1) = hits
            owner, neighbour, a, b = (c0, c1, a0, b0) if c0 < c1 else (c1, c0, _a1, _b1)
            internal.append((owner, neighbour, (a, b, b + nxy, a + nxy)))
        elif len(hits) == 1:
            cell, a, b = hits[0]
            ax, ay = points[a]
            bx, by = points[b]
            mx, my = 0.5 * (ax + bx), 0.5 * (ay + by)
            tol = 1e-8
            if abs(ax) < tol and abs(bx) < tol:
                name = "inlet"
            elif abs(ax - lx) < tol and abs(bx - lx) < tol and vent0 - tol < my < vent1 + tol:
                name = "outlet"
            else:
                name = "walls"
            boundary[name].append((cell, (a, b, b + nxy, a + nxy)))
        else:
            raise RuntimeError(f"edge shared by {len(hits)} cells")
    for cell, (a, b, c) in enumerate(tris):
        boundary["front"].append((cell, (a, c, b)))
        boundary["back"].append((cell, (a + nxy, b + nxy, c + nxy)))

    internal.sort(key=lambda item: (item[1], item[0]))
    faces = [verts for _o, _n, verts in internal]
    owner = [o for o, _n, _v in internal]
    neighbour = [n for _o, n, _v in internal]
    patches = []
    for name in ("inlet", "outlet", "walls", "front", "back"):
        start = len(faces)
        for cell, verts in boundary[name]:
            faces.append(verts)
            owner.append(cell)
        kind = "empty" if name in ("front", "back") else ("wall" if name == "walls" else "patch")
        patches.append((name, kind, start, len(boundary[name])))
        if not boundary[name]:
            raise RuntimeError(f"patch {name} is empty")

    def face_str(verts):
        return f"{len(verts)}(" + " ".join(str(v) for v in verts) + ")"

    mesh_dir.mkdir(parents=True, exist_ok=True)
    (mesh_dir / "points").write_text(_foam("points", "vectorField",
        f"{len(points)}\n(\n" + "\n".join(f"({x:.8e} {y:.8e} {z:.8e})"
        for (x, y), z in zip(points, [0.0] * nxy + [THICKNESS] * nxy)) + "\n)\n"))
    (mesh_dir / "faces").write_text(_foam("faces", "faceList",
        f"{len(faces)}\n(\n" + "\n".join(face_str(f) for f in faces) + "\n)\n"))
    (mesh_dir / "owner").write_text(_foam("owner", "labelList",
        f"{len(owner)}\n(\n" + "\n".join(str(v) for v in owner) + "\n)\n"))
    (mesh_dir / "neighbour").write_text(_foam("neighbour", "labelList",
        f"{len(neighbour)}\n(\n" + "\n".join(str(v) for v in neighbour) + "\n)\n"))
    blocks = []
    for name, kind, start, count in patches:
        blocks.append(f"    {name}\n    {{\n        type        {kind};\n        nFaces      {count};\n        startFace   {start};\n    }}")
    (mesh_dir / "boundary").write_text(_foam("boundary", "polyBoundaryMesh",
        f"{len(patches)}\n(\n" + "\n".join(blocks) + "\n)\n"))
    print(f"OpenFOAM polyMesh {mesh_dir}: {len(tris)} prisms, "
          + ", ".join(f"{n} {c}" for n, _k, _s, c in patches))


def main():
    spec = load_spec()
    out = Path(__file__).resolve().parent
    build_elmer_mesh(out / "mesh_2d.msh", spec)
    write_of_polymesh(out.parent / "fluid", spec)


if __name__ == "__main__":
    main()
