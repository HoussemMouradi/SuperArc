"""Elmer current solve, circuit scaling, and the planar self-field.

The OpenFOAM mesh is one cell thick. Elmer sees the same cross-section as a 2D
mesh (metal + gas). Conductivity in the gas is binned from the LTE table so each
Elmer material is constant. The self-field uses the planar stream function: Elmer's
MagnetoDynamics2D solver is the other 2D reduction (current out of the page).
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from arcsim.properties import PlasmaProperties

MU0 = 4e-7 * np.pi


def bz_from_jy(x, y, jy, area, nx: int = 80, ny: int = 36) -> np.ndarray:
    """Bz = mu0 * integral_x^Lx Jy dx' on an unstructured cloud, binned in x-y."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    jy = np.asarray(jy, float)
    area = np.asarray(area, float)
    xedges = np.linspace(x.min(), x.max() + 1e-15, nx + 1)
    yedges = np.linspace(y.min(), y.max() + 1e-15, ny + 1)
    dx = float(xedges[1] - xedges[0])
    sum_jA, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=jy * area)
    sum_A, _, _ = np.histogram2d(x, y, bins=[xedges, yedges], weights=area)
    Jy = np.divide(sum_jA, sum_A, out=np.zeros_like(sum_jA), where=sum_A > 0)
    psi = np.cumsum(Jy[::-1], axis=0)[::-1] * dx - 0.5 * Jy * dx
    ix = np.clip(np.digitize(x, xedges) - 1, 0, nx - 1)
    iy = np.clip(np.digitize(y, yedges) - 1, 0, ny - 1)
    return MU0 * psi[ix, iy]


def net_heat(q_joule: np.ndarray, temperature: np.ndarray, props: PlasmaProperties) -> np.ndarray:
    """Joule heating minus net-emission radiation, in W/m^3."""
    return np.asarray(q_joule, float) - 4.0 * np.pi * props.nec_of_T(temperature)


class ElmerMesh:
    def __init__(self, directory: Path):
        self.directory = Path(directory)
        node_lines = (self.directory / "mesh.nodes").read_text().splitlines()
        ids, coords = [], []
        for line in node_lines:
            if not line.strip():
                continue
            parts = line.split()
            ids.append(int(parts[0]))
            floats = [float(v) for v in parts[1:]]
            # id, constraint, x, y, z   or   id, x, y, z
            xyz = floats[-3:] if len(floats) >= 4 else floats
            coords.append(xyz + [0.0] * (3 - len(xyz)))
        self.node_id = {i: k for k, i in enumerate(ids)}
        self.nodes = np.array(coords, float)

        self.elements = []  # (elem_id, body, type, node_ids)
        for line in (self.directory / "mesh.elements").read_text().splitlines():
            if not line.strip():
                continue
            p = line.split()
            self.elements.append((int(p[0]), int(p[1]), int(p[2]), [int(v) for v in p[3:]]))
        self.bodies = np.array([e[1] for e in self.elements], int)
        centers = []
        areas = []
        for _eid, _body, _typ, nodes in self.elements:
            pts = self.nodes[[self.node_id[n] for n in nodes[:3]]]
            centers.append(pts.mean(axis=0))
            d = pts[1, :2] - pts[0, :2]
            e = pts[2, :2] - pts[0, :2]
            areas.append(0.5 * abs(d[0] * e[1] - d[1] * e[0]))
        self.centers = np.array(centers, float)
        self.areas = np.array(areas, float)
        self.gas = self.bodies == 1

        self.boundaries = []  # (boundary_id, node pair)
        for line in (self.directory / "mesh.boundary").read_text().splitlines():
            if not line.strip():
                continue
            p = [int(v) for v in line.split()]
            # id, boundary id, left parent, right parent, element type, nodes...
            self.boundaries.append((p[1], p[5:7]))

    def boundary_ids_touching(self, predicate) -> list[int]:
        """Boundary ids whose edges, taken together, satisfy predicate(mean midpoint)."""
        groups: dict[int, list] = {}
        for bc, nodes in self.boundaries:
            if len(nodes) < 2 or any(n not in self.node_id for n in nodes[:2]):
                continue
            a = self.nodes[self.node_id[nodes[0]]]
            b = self.nodes[self.node_id[nodes[1]]]
            groups.setdefault(bc, []).append(0.5 * (a + b))
        hits = []
        for bc, mids in groups.items():
            mean = sum(mids) / len(mids)
            if predicate(mean):
                hits.append(bc)
        return sorted(hits)

    def write_bodies(self, bodies: np.ndarray, path: Path | None = None):
        path = path or (self.directory / "mesh.elements")
        lines = []
        for (eid, _body, typ, nodes), body in zip(self.elements, bodies):
            lines.append(f"{eid} {int(body)} {typ} " + " ".join(str(n) for n in nodes))
        path.write_text("\n".join(lines) + "\n")


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def write_sif(path: Path, mesh: ElmerMesh, n_metal: int, gas_materials: dict[int, float],
              terminal_plus: list[int], terminal_zero: list[int]):
    """Body ids 1..n_metal are conductors. gas_materials maps later body ids to sigma.

    Elmer requires every body id from 1 to the maximum to be defined, so the
    ids written into the mesh are compacted with no gaps.
    """
    bodies = []
    materials = ["Material 1\n  Electric Conductivity = 1.0e5\nEnd\n"]
    for body in range(1, n_metal + 1):
        bodies.append(
            f"Body {body}\n  Target Bodies(1) = {body}\n  Equation = 1\n  Material = 1\nEnd\n")
    for i, (body, sigma) in enumerate(sorted(gas_materials.items()), start=2):
        materials.append(f"Material {i}\n  Electric Conductivity = {sigma:.6e}\nEnd\n")
        bodies.append(
            f"Body {body}\n  Target Bodies(1) = {body}\n  Equation = 1\n  Material = {i}\nEnd\n")
    bcs = []
    if terminal_plus:
        bcs.append("Boundary Condition 1\n  Target Boundaries(" + str(len(terminal_plus)) + ") = "
                   + " ".join(str(b) for b in terminal_plus) + "\n  Potential = Real 1.0\nEnd\n")
    if terminal_zero:
        bcs.append("Boundary Condition 2\n  Target Boundaries(" + str(len(terminal_zero)) + ") = "
                   + " ".join(str(b) for b in terminal_zero) + "\n  Potential = Real 0.0\nEnd\n")
    path.write_text(f"""Header
  CHECK KEYWORDS Warn
  Mesh DB "." "{mesh.directory.name}"
  Results Directory "."
End

Simulation
  Max Output Level = 4
  Coordinate System = Cartesian 2D
  Simulation Type = Steady state
  Steady State Max Iterations = 1
  Output Intervals(1) = 1
End

Solver 1
  Equation = Stat Current Solver
  Procedure = "StatCurrentSolve" "StatCurrentSolver"
  Variable = Potential
  Calculate Volume Current = Logical True
  Calculate Joule Heating = Logical True
  Calculate Electric Field = Logical True
  Linear System Solver = Iterative
  Linear System Iterative Method = BiCGStab
  Linear System Preconditioning = ILU1
  Linear System Max Iterations = 2000
  Linear System Convergence Tolerance = 1.0e-8
  Nonlinear System Max Iterations = 1
  Nonlinear System Convergence Tolerance = 1.0e-6
  Steady State Convergence Tolerance = 1.0e-6
End

Solver 2
  Exec Solver = After Saving
  Equation = Result Output
  Procedure = "ResultOutputSolve" "ResultOutputSolver"
  Output File Name = em
  Output Format = Vtu
  Binary Output = Logical False
  Single Precision = Logical False
End

Equation 1
  Active Solvers(1) = 1
End

{''.join(materials)}
{''.join(bodies)}
{''.join(bcs)}
""")


def run_elmer(work: Path, elmer_solver: str = "ElmerSolver") -> None:
    env = dict(**{k: v for k, v in __import__("os").environ.items()})
    result = subprocess.run([elmer_solver, "case.sif"], cwd=work, env=env, text=True,
                            capture_output=True, timeout=180)
    (work / "elmer.log").write_text(result.stdout + "\n" + result.stderr)
    if result.returncode != 0:
        tail = "\n".join((result.stdout + result.stderr).splitlines()[-40:])
        raise RuntimeError(f"ElmerSolver failed (exit {result.returncode})\n{tail}")


def latest_vtu(work: Path) -> Path:
    files = sorted(work.rglob("em*.vtu"), key=lambda p: p.stat().st_mtime)
    if not files:
        raise FileNotFoundError(f"no Elmer VTU result in {work}")
    return files[-1]


def read_cell_fields(vtu: Path) -> dict[str, np.ndarray]:
    import meshio

    mesh = meshio.read(vtu)
    tri = next(c for c in mesh.cells if c.type in ("triangle", "triangle6"))
    block = mesh.cells.index(tri)
    out = {"centers": mesh.points[tri.data[:, :3]].mean(axis=1)}
    data = {}
    for name, blocks in mesh.cell_data.items():
        data[name.lower()] = np.asarray(blocks[block])
    for name, values in mesh.point_data.items():
        key = name.lower()
        if key not in data:
            data[key] = np.asarray(values)[tri.data[:, :3]].mean(axis=1)
    out["fields"] = data
    return out


def field(data: dict, *needles: str) -> np.ndarray:
    for name, values in data.items():
        if all(n in name for n in needles):
            return np.asarray(values, float)
    raise KeyError(f"none of {list(data)} matches {needles}")


class ArcEM:
    """Maps OpenFOAM cell centres onto the Elmer gas mesh and scales by the circuit."""

    def __init__(self, mesh: ElmerMesh, props: PlasmaProperties, depth: float):
        self.mesh = mesh
        self.props = props
        self.depth = depth
        self.of_tree = None
        self.elmer_of = None  # Elmer gas element -> nearest OpenFOAM cell

    def map_to(self, of_xy: np.ndarray):
        gas_xy = self.mesh.centers[self.mesh.gas, :2]
        self.of_tree = cKDTree(np.asarray(of_xy, float)[:, :2])
        _dist, self.elmer_of = self.of_tree.query(gas_xy, k=1)
        self.of_elmer = cKDTree(gas_xy).query(np.asarray(of_xy, float)[:, :2], k=1)

    def conductivities(self, temperature_of: np.ndarray) -> np.ndarray:
        """Per OpenFOAM cell, from the LTE table."""
        return np.maximum(self.props.sigma_of_T(temperature_of), 1e-4)

    def assign_bins(self, temperature_of: np.ndarray, n_bins: int = 12) -> dict[int, float]:
        sigma = self._sheath(self.conductivities(temperature_of)[self.elmer_of])
        edges = np.geomspace(5e-2, 2e4, n_bins + 1)
        bins = np.clip(np.digitize(sigma, edges) - 1, 0, n_bins - 1)
        bodies = np.zeros(len(self.mesh.elements), int)
        metal_ids = [b for b in sorted(set(self.mesh.bodies.tolist())) if b != 1]
        n_metal = 0
        for original in metal_ids:
            n_metal += 1
            bodies[self.mesh.bodies == original] = n_metal
        gas_materials = {}
        placed = self._gas_bins(bins)
        next_id = n_metal + 1
        for b in range(n_bins):
            mask = self.mesh.gas & (placed == b)
            if not np.any(mask):
                continue
            bodies[mask] = next_id
            gas_materials[next_id] = float(np.mean(sigma[bins == b]))
            next_id += 1
        self.mesh.write_bodies(bodies)
        return n_metal, gas_materials

    def _sheath(self, sigma_gas: np.ndarray) -> np.ndarray:
        """Gas elements touching metal take the hottest conductivity at that node."""
        sigma = np.asarray(sigma_gas, float).copy()
        full = np.zeros(len(self.mesh.elements))
        full[self.mesh.gas] = sigma
        metal_nodes: set[int] = set()
        node_max: dict[int, float] = {}
        for (_eid, _body, _typ, nodes), is_gas, value in zip(self.mesh.elements, self.mesh.gas, full):
            if is_gas:
                for node in nodes:
                    node_max[node] = max(node_max.get(node, 0.0), value)
            else:
                metal_nodes.update(nodes)
        for local, elem_i in enumerate(np.flatnonzero(self.mesh.gas)):
            nodes = self.mesh.elements[elem_i][3]
            if any(node in metal_nodes for node in nodes):
                sigma[local] = max(sigma[local], max(node_max.get(node, 0.0) for node in nodes))
        return sigma

    def _gas_bins(self, bins: np.ndarray) -> np.ndarray:
        full = np.zeros(self.mesh.gas.shape, int)
        full[self.mesh.gas] = bins
        return full

    def sample_to_of(self, elmer_values: np.ndarray, n_of: int) -> np.ndarray:
        """Elmer gas-element values -> OpenFOAM cells (nearest element)."""
        _dist, index = self.of_elmer
        return np.asarray(elmer_values, float)[index]
