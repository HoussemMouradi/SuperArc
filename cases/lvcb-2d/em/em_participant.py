"""preCICE participant: Elmer current solve + circuit, sources for OpenFOAM.

Run from cases/lvcb-2d (Allrun does this):

    uv run python em/em_participant.py
"""

from __future__ import annotations

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CASE = Path(__file__).resolve().parents[1]
REPO = CASE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from arcsim.circuit import Circuit, CircuitConfig, fall_voltage  # noqa: E402
from arcsim.properties import PlasmaProperties  # noqa: E402
from em_model import (ArcEM, ElmerMesh, bz_from_jy, field, latest_vtu, net_heat,  # noqa: E402
                      read_cell_fields, run_elmer, write_sif)


def _vector(data, *needles):
    values = field(data, *needles)
    if values.ndim == 1:
        raise KeyError(needles)
    return values


def solve_window(em: ArcEM, mesh: ElmerMesh, temperature: np.ndarray, circuit: Circuit,
                 arc_cfg, depth: float, self_field_scale: float, b_ext: float, work: Path):
    n_metal, gas_materials = em.assign_bins(temperature)
    plus = mesh.boundary_ids_touching(lambda m: abs(m[0]) < 1e-5 and 0.0195 < m[1] < 0.0225)
    zero = mesh.boundary_ids_touching(lambda m: abs(m[0]) < 1e-5 and -1e-5 < m[1] < 0.0025)
    if not plus or not zero:
        raise RuntimeError(f"terminal boundaries not found (plus={plus}, zero={zero})")
    write_sif(work / "case.sif", mesh, n_metal, gas_materials, plus, zero)
    for old in work.rglob("em*.vtu"):
        old.unlink()
    run_elmer(work)
    result = read_cell_fields(latest_vtu(work))
    tree_centres = result["centers"][:, :2]
    from scipy.spatial import cKDTree
    gas_xy = mesh.centers[mesh.gas, :2]
    _dist, index = cKDTree(tree_centres).query(gas_xy, k=1)
    data = result["fields"]
    try:
        joule = field(data, "joule")
    except KeyError:
        electric = _vector(data, "electric")
        joule = None
        names = list(data)
        raise KeyError(f"no joule-heating field in {names}") from None
    current = _vector(data, "current")
    joule_e = np.ravel(np.asarray(joule, float))[index]
    current_e = np.asarray(current, float)[index]
    jx, jy = current_e[:, 0], current_e[:, 1]
    area = mesh.areas[mesh.gas]
    # Unit-voltage solution: power per unit depth integrated, times depth, is the current.
    conductance = max(float(np.sum(joule_e * area) * depth), 1e-12)
    # Cap the column voltage. A collapsing conductance would otherwise scale the
    # unit-voltage Joule field by thousands of volts and blow the temperature up.
    u_ohm = float(np.clip(circuit.I / conductance, -600.0, 600.0))
    jx_p, jy_p = jx * u_ohm, jy * u_ohm
    bz = self_field_scale * bz_from_jy(gas_xy[:, 0], gas_xy[:, 1], jy_p, area) + b_ext
    q_e = joule_e * u_ohm**2
    q_of = em.sample_to_of(q_e, len(temperature))
    q_net = np.clip(net_heat(q_of, temperature, em.props), -5e9, 5e11)
    fx = em.sample_to_of(jy_p * bz, len(temperature))
    fy = em.sample_to_of(-jx_p * bz, len(temperature))
    lorentz = np.column_stack([fx, fy, np.zeros_like(fx)])
    force = np.linalg.norm(lorentz, axis=1)
    cap = 1e8
    if np.any(force > cap):
        lorentz *= np.minimum(1.0, cap / np.maximum(force, 1e-30))[:, None]
    # Splitter plates are body ids 4..9. A plate counts once current crowds onto it.
    n_series = 1
    total = max(float(np.sum(np.hypot(jx_p, jy_p) * area)), 1e-12)
    for body in range(4, 10):
        if body not in set(mesh.bodies.tolist()):
            continue
        near = _near_plate(mesh, body, gas_xy)
        if near.any() and float(np.sum(np.hypot(jx_p[near], jy_p[near]) * area[near])) > 0.25 * total:
            n_series += 1
    u_fall = fall_voltage(circuit.I, n_series, arc_cfg.U_fall, arc_cfg.I_smooth)
    positive = q_of > 0
    x_arc = float(np.sum(q_of[positive] * em.of_xy[positive, 0]) / np.sum(q_of[positive])) if positive.any() else float("nan")
    info = {
        "G": conductance, "U_ohm": u_ohm, "U_fall": float(u_fall), "U_arc": float(circuit.I / conductance + u_fall),
        "n_series": n_series, "T_max": float(np.max(temperature)), "x_arc_mm": x_arc * 1e3,
    }
    return q_net, lorentz, info


def _near_plate(mesh: ElmerMesh, body: int, gas_xy: np.ndarray) -> np.ndarray:
    plate = mesh.centers[mesh.bodies == body, :2]
    if len(plate) == 0:
        return np.zeros(len(gas_xy), bool)
    from scipy.spatial import cKDTree
    dist, _ = cKDTree(plate).query(gas_xy, k=1)
    return dist < 1.5e-3


@dataclass
class ArcSettings:
    U_fall: float
    I_smooth: float


def load_settings(path: Path):
    """Circuit, fall voltage, depth and self-field scale from the case TOML."""
    data = tomllib.loads(Path(path).read_text())
    c, a, s, g = data["circuit"], data["arc"], data["solver"], data["geometry"]
    circuit = CircuitConfig(
        V_rms=float(c["V_rms"]),
        frequency=float(c["frequency"]),
        R=float(c["R"]),
        L=float(c["L"]),
        current_phase_deg=float(c.get("current_phase_deg", 0.0)),
    )
    arc = ArcSettings(U_fall=float(a.get("U_fall", 20.0)), I_smooth=float(a.get("I_smooth", 1.0)))
    return circuit, arc, float(g["depth"]) * 1e-3, float(s.get("self_field_scale", 1.0)), float(s.get("B_ext", 0.0))


def main():
    import precice

    circuit_cfg, arc_cfg, depth, self_field_scale, b_ext = load_settings(REPO / "configs" / "lvcb_2d.toml")
    work = CASE / "em" / "work"
    elmer_dir = work / "elmermesh"
    original = elmer_dir / "mesh.elements.orig"
    current = elmer_dir / "mesh.elements"
    if original.exists():
        current.write_text(original.read_text())
    mesh = ElmerMesh(elmer_dir)
    props = PlasmaProperties.from_csv()
    em = ArcEM(mesh, props, depth=depth)
    circuit = Circuit(circuit_cfg)
    participant = precice.Participant("EM", str(CASE / "precice-config.xml"), 0, 1)
    mesh_name = "Fluid-Mesh"
    # Direct access to the mesh OpenFOAM provides: the whole chamber, one cell thick.
    participant.set_mesh_access_region(mesh_name, [-1e-4, 0.051, -1e-4, 0.023, -1e-4, 1e-3])
    participant.initialize()
    vertex_ids, coords = participant.get_mesh_vertex_ids_and_coordinates(mesh_name)
    coords = np.asarray(coords, float)
    em.map_to(coords[:, :2])
    em.of_xy = coords[:, :2]
    history = CASE / "em" / "history.csv"
    history.write_text("t,I,U_arc,G,T_max,n_series,x_arc_mm\n")
    t = 0.0
    print(f"[em] {coords.shape[0]} fluid cells, {int(mesh.gas.sum())} gas elements", flush=True)
    while participant.is_coupling_ongoing():
        dt = participant.get_max_time_step_size()
        temperature = np.asarray(participant.read_data(mesh_name, "Temperature", vertex_ids, dt), float).reshape(-1)
        t += dt
        q_net, lorentz, info = solve_window(
            em, mesh, temperature, circuit, arc_cfg, em.depth, self_field_scale, b_ext, work)
        circuit.step(t, dt, info["G"], info["U_fall"])
        participant.write_data(mesh_name, "Qnet", vertex_ids, q_net)
        participant.write_data(mesh_name, "Lorentz", vertex_ids, lorentz)
        participant.advance(dt)
        info["t"] = t
        info["I"] = circuit.I
        with history.open("a") as handle:
            handle.write(f"{t:.6e},{circuit.I:.6e},{info['U_arc']:.6e},{info['G']:.6e},"
                         f"{info['T_max']:.6e},{info['n_series']},{info['x_arc_mm']:.6e}\n")
        print(f"[em] t={t*1e3:.3f} ms  I={circuit.I:.1f} A  U_arc={info['U_arc']:.1f} V  "
              f"T_max={info['T_max']:.0f} K  n={info['n_series']}", flush=True)
    participant.finalize()


if __name__ == "__main__":
    main()
