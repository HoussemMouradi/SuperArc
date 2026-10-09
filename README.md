# SuperArc

Planar model of a switching arc in a low-voltage circuit breaker. The contact is already open. OpenFOAM integrates the compressible flow and the energy equation, Elmer integrates current continuity in the gas and the metal, and preCICE exchanges temperature, Joule heating and the Lorentz force. An R–L–source circuit closes the current.

The geometry below is a parametric chamber: runners, an open contact gap, and a stack of splitter plates.

```
y=22 +-----------------------------------------------+
     |#### upper runner / fixed contact  (+) ####     |
     |     [fixed tip]                                |  vent
     |                      ===== splitter plates ==  |
     |     [moving tip]     =====   (6, floating) ==  |
     |        | gap 6 mm                              |
     |#### lower runner / moving contact (0) ####     |
y=0  +-----------------------------------------------+
     x=0 terminals                               x=50 mm
```

- Chamber 50 mm by 22 mm. Device depth 10 mm, used only to turn the planar current into amperes. The flow mesh is one cell thick (0.5 mm); front and back are `empty`.
- Runners from x = 0 to 46 mm. Fixed tip x = 8–12 mm, y = 16–20 mm. Moving tip held open, top at y = 10 mm, so the gap is 6 mm.
- Six floating plates, x = 30–44 mm, each 1 mm thick, starting at y = 3.71, 6.43, 9.14, 11.86, 14.57 and 17.29 mm.
- Vents on the left and right, y = 2–20 mm. Terminals at x = 0: potential 1 on the upper runner, potential 0 on the lower runner.
- Ignition: 12 kK in the box x = 9–11 mm, y = 10–16 mm.
- Circuit: 230 V rms, 50 Hz, R = 0.08 Ω, L = 0.9 mH. Contact separation is taken at 20° of the prospective current. Prospective peak is about 1.11 kA.
- Mesh: 0.4 mm on the arc route (x = 6–48 mm, y = 2–20 mm), 0.8 mm elsewhere.
- Time step 0.25 µs. Coupling window 10 µs. The preCICE run stops at 2.6 ms.

On that mesh the column leaves the gap, runs along the runners and enters the splitter plates around 1.5–1.6 ms. At 1.30 ms the current is about 620 A, the arc voltage about 200 V and the hottest cell is near 39 kK, with the heating centroid near x = 23 mm. At 1.56 ms the centroid is near x = 40 mm, inside the plate stack (the plates occupy x = 30–44 mm). By 2.6 ms the current has fallen to zero.

## Install

Ubuntu 24.04, with the versions used for the run above:

```bash
wget https://github.com/precice/precice/releases/download/v3.4.1/libprecice3_3.4.1_noble.deb
sudo apt install ./libprecice3_3.4.1_noble.deb
wget -q -O - https://dl.openfoam.com/add-debian-repo.sh | sudo bash
sudo apt install openfoam2606-dev gmsh python3-gmsh gfortran g++ libsuitesparse-dev
# OpenFOAM adapter v1.4.0, from a shell where the OpenFOAM bashrc is sourced
wget https://github.com/precice/openfoam-adapter/archive/refs/tags/v1.4.0.tar.gz
tar -xzf v1.4.0.tar.gz && cd openfoam-adapter-1.4.0 && ./Allwmake
```

Elmer 26.2 is expected on `PATH` (this run used `/opt/elmer`, built with the GUI and MPI off). Then, in this repository:

```bash
uv sync
```

`uv` installs the Python side: NumPy, SciPy, pyprecice, meshio and matplotlib. The participant imports `arcsim.circuit` and `arcsim.properties`.

## Run

```bash
bash cases/lvcb-2d/Allrun          # mesh, couple for 2.6 ms, plot
bash cases/lvcb-2d/Allrun mesh     # meshes only
bash cases/lvcb-2d/Allrun solve    # reuse existing meshes
bash cases/lvcb-2d/Allclean
```

`Allrun` starts `buoyantPimpleFoam` first (preCICE acceptor), then `em/em_participant.py`. Outputs:

- `cases/lvcb-2d/em/history.csv` — time, current, arc voltage, conductance, peak temperature, series roots, arc position
- `cases/lvcb-2d/temperature.png`, `cases/lvcb-2d/waveforms.png`
- OpenFOAM time directories under `cases/lvcb-2d/fluid/`

ParaView reads the case through the empty file `cases/lvcb-2d/fluid/lvcb.foam`. Open that file, select `internalMesh`, and colour the cells by `T`. The time toolbar is in seconds (0.0016 is 1.6 ms). A threshold `T > 8000` isolates the column. After a finished run:

```bash
pvpython cases/lvcb-2d/paraview/render_case.py
```

Frames land in `cases/lvcb-2d/paraview/frames/`.

## What the case solves

| Piece | Where |
|---|---|
| Flow and energy | OpenFOAM `buoyantPimpleFoam` |
| Current in the gas, the contacts and the floating plates | Elmer `StatCurrentSolver`, input [`cases/lvcb-2d/em/case.sif`](cases/lvcb-2d/em/case.sif) |
| Temperature, net heating, Lorentz force | preCICE, OpenFOAM adapter, volume centres |
| R–L circuit, LTE conductivity, radiation, planar self-field | `cases/lvcb-2d/em/em_participant.py` |

Elmer solves a unit-voltage current field. The participant scales it so the ohmic voltage is `I/G`, clipped to ±600 V, and adds an electrode fall `n * U_fall * tanh(I/I_s)` with `U_fall = 20 V`. The self-field is the planar field of the in-plane current, `Bz = µ0 ∫ Jy dx'`, because Elmer's out-of-plane magnetic solver does not match this current direction. Lorentz components are `Fx = Jy Bz` and `Fy = −Jx Bz`, so the force drives the arc toward the plates.

OpenFOAM uses a perfect gas with `Cp = 50 kJ/kg/K`. That effective heat capacity stands in for the dissociation and ionization enthalpy. Conductivity and the net emission coefficient come from `src/arcsim/data/air_1atm.csv`, an approximate LTE air table at 1 atm. The contact does not move during the run.

More detail on the dictionaries, the boundary conditions and the coupling is in [`cases/lvcb-2d/README.md`](cases/lvcb-2d/README.md).
