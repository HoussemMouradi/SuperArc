# Planar OpenFOAM + Elmer + preCICE breaker arc

Open contact, gap 6 mm. OpenFOAM (`buoyantPimpleFoam`) integrates the flow and the energy equation. Elmer integrates current conservation through the gas, the contacts and the floating splitter plates. preCICE passes cell-centred temperature to Elmer and passes volumetric heating and the Lorentz force back. The R–L circuit, the LTE air table and the planar self-field live in `em/em_participant.py`.

The default coupling runs to 2.6 ms: time step 0.25 µs, window 10 µs (`precice-config.xml`). OpenFOAM `endTime` is longer than that; preCICE stops the run.

## Install

See the repository [README](../../README.md). From the repository root, `uv sync`, with OpenFOAM 2606, preCICE 3.4.1, the OpenFOAM adapter 1.4.0, Elmer 26.2 and Gmsh on `PATH`.

## Run

```bash
bash cases/lvcb-2d/Allrun          # mesh, couple, plot
bash cases/lvcb-2d/Allrun mesh     # meshes only
bash cases/lvcb-2d/Allrun solve    # reuse existing meshes
bash cases/lvcb-2d/Allclean
```

`Allrun` copies `fluid/0/T.initial` over `fluid/0/T` before `setFields`, so a mesh change does not inherit the previous cell count. It also deletes `precice-run/` so a leftover connection file cannot block the next start.

Outputs: `em/history.csv`, `temperature.png`, `waveforms.png`, and the OpenFOAM time directories under `fluid/`.

ParaView reads `fluid/lvcb.foam`:

```bash
pvpython cases/lvcb-2d/paraview/render_case.py
paraview --data=cases/lvcb-2d/fluid/lvcb.foam
```

Colour `internalMesh` by cell field `T` (300–20000 K) and play the time steps. Each written frame is 0.05 ms. A threshold above 8000 K shows the column without the cold gas.

## Boundary conditions

OpenFOAM, one prism through the thickness. Front and back are `empty`.

| Patch | Velocity | Temperature | `p_rgh` |
|---|---|---|---|
| inlet (x = 0, y = 2–20 mm) | `pressureInletOutletVelocity` | `inletOutlet`, 300 K | fixed 101325 Pa |
| outlet (x = 50 mm, y = 2–20 mm) | `inletOutlet` | `inletOutlet`, 300 K | fixed 101325 Pa |
| walls | `noSlip` | fixed 300 K | `fixedFluxPressure` |
| front, back | `empty` | `empty` | `empty` |

`Qnet` and `Lorentz` are not solved. They are `zeroGradient` on the real patches and `empty` on front and back, and a `readFields` function object loads them before the preCICE adapter is constructed. The sources are coded `fvOptions`: `jouleSource` on enthalpy and `lorentzSource` on momentum, both in `codeAddSupRho`, because `buoyantPimpleFoam` calls `fvOptions(rho, psi)`.

Thermo: `heRhoThermo`, pure mixture, constant transport, `hConst`, perfect gas, sensible enthalpy, molar weight 28.96, `Cp = 50000` J/kg/K, `mu = 1.8e-5`, Pr = 0.7. Laminar. Gravity `(0, -9.81, 0)`. Radiation model `none` in OpenFOAM; net emission is subtracted on the Elmer side and arrives inside `Qnet`.

Pressure solver is PCG with DIC. Density uses a diagonal solver, including `rhoFinal`. Limits: `rhoMin` 0.001, `rhoMax` 5, `pMin` 5000 Pa, `pMax` 500000 Pa.

Elmer `StatCurrentSolver`, Cartesian, steady, BiCGStab + ILU1, tolerance `1e-8`. It writes volume current, Joule heating and the electric field. Metal conductivity is `1e5` S/m. Gas conductivity is binned from the LTE table. Gas elements that touch metal take the hottest nodal gas conductivity, which stands in for the near-electrode layer. Body ids are contiguous. The upper-runner terminal is `phi = 1`, the lower-runner terminal is `phi = 0`. Every other edge has zero normal current. The plates are floating conductors: high conductivity, no Dirichlet value.

The Elmer input is [`em/case.sif`](em/case.sif). During a run the participant writes a fresh `em/work/case.sif` every coupling window. The solver block and the two terminal conditions stay as in `em/case.sif`. The gas materials are the conductivity bins of that window. The file in the repository is the last window of the reference run.

## Coupling

Serial-explicit. Fluid is first and accepts the socket; EM connects. Window `1e-5` s, stop at `2.6e-3` s. The EM participant reads temperature at the OpenFOAM cell centres (`api-access` on the received mesh), runs Elmer on a unit voltage, and returns

- `G` from the unit-voltage Joule field times the 10 mm depth,
- ohmic voltage `clip(I/G, ±600 V)`,
- `Qnet = q_Joule(U) − q_radiation(T)`,
- Lorentz force from the planar self-field, capped at `1e8` N/m³.

The circuit update is `L dI/dt = v_s − R I − U_arc`, with the ohmic arc resistance treated implicitly. The reported arc voltage is `I/G` plus the fall. Heating uses the clipped ohmic voltage, so a collapsing conductance cannot dump an unbounded source into the energy equation.

## Scope of this case

- One cell through the thickness. Front and back are `empty`.
- The contact does not move. The gap stays at the open position, 6 mm.
- Perfect gas with constant viscosity in OpenFOAM, and an effective heat capacity of 50 kJ/kg/K so the energy equation can absorb arc heating. Dissociation and ionization sit in the conductivity and radiation table, not in the equation of state.
- The self-field is the planar field of the in-plane current, computed in the participant.
- Electrode falls are a voltage per series root, `n * 20 V * tanh(I/I_s)`, not a resolved sheath.
