"""Plot the coupled run: arc waveforms and the temperature on the front faces."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CASE = Path(__file__).resolve().parent


def _internal_scalars(path: Path) -> np.ndarray:
    lines = path.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if "nonuniform" in line)
    count = int(lines[start + 1])
    return np.array([float(lines[start + 3 + i]) for i in range(count)])


def _front_triangles(mesh: Path):
    def body(name):
        text = (mesh / name).read_text().split("}")[-1]
        return text
    points = []
    for line in body("points").splitlines():
        line = line.strip()
        if line.startswith("(") and line.endswith(")") and line.count(" ") >= 2:
            x, y, z = (float(v) for v in line[1:-1].split())
            points.append((x, y, z))
    points = np.array(points)
    faces = []
    for line in body("faces").splitlines():
        line = line.strip()
        if "(" in line and line.endswith(")") and line[0].isdigit():
            verts = line[line.index("(") + 1:-1].split()
            faces.append([int(v) for v in verts])
    boundary = (mesh / "boundary").read_text()
    blocks = boundary.split("front")[1].split("}")[0]
    n_faces = int(blocks.split("nFaces")[1].split(";")[0])
    start = int(blocks.split("startFace")[1].split(";")[0])
    tris = np.array(faces[start:start + n_faces])
    if tris.shape[1] != 3:
        raise RuntimeError(f"front faces are not triangles: {tris.shape}")
    return points, tris


def plot_temperature(path: Path):
    times = sorted(
        (p for p in (CASE / "fluid").iterdir() if p.is_dir() and p.name[:1].isdigit() and p.name != "0"),
        key=lambda p: float(p.name),
    )
    if not times or not (times[-1] / "T").exists():
        print("no temperature field to plot")
        return
    points, tris = _front_triangles(CASE / "fluid" / "constant" / "polyMesh")
    temperature = _internal_scalars(times[-1] / "T")
    fig, ax = plt.subplots(figsize=(9, 4.2))
    triangles = ax.tripcolor(
        points[:, 0] * 1e3, points[:, 1] * 1e3, tris, temperature,
        shading="flat", cmap="inferno", vmin=300, vmax=max(8000.0, float(np.percentile(temperature, 99))),
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_title(f"OpenFOAM temperature at t = {float(times[-1].name) * 1e3:.3f} ms")
    fig.colorbar(triangles, ax=ax, label="T [K]")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_waveforms(path: Path):
    history = CASE / "em" / "history.csv"
    if not history.exists():
        print("no em/history.csv")
        return
    data = np.genfromtxt(history, delimiter=",", names=True)
    if data.size == 0:
        return
    t = np.atleast_1d(data["t"]) * 1e3
    fig, axes = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    axes[0].plot(t, np.atleast_1d(data["I"]), color="C0")
    axes[0].set_ylabel("current [A]")
    axes[1].plot(t, np.atleast_1d(data["U_arc"]), color="C3")
    axes[1].set_ylabel("arc voltage [V]")
    axes[1].set_xlabel("time [ms]")
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.suptitle("2D OpenFOAM–Elmer arc")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    plot_temperature(CASE / "temperature.png")
    plot_waveforms(CASE / "waveforms.png")
    print(f"wrote {CASE / 'temperature.png'} and {CASE / 'waveforms.png'}")
