# SIMPLE CFD Solver (C++)

## Overview
Laminar, incompressible SIMPLE solver on a structured Cartesian grid (finite volume, steady-state with pseudo-transient relaxation). Geometry, fluid fields, and postprocessing are all contained in this repo; no external CFD suite needed.

## Components (key files)
- `SIMPLE.h / SIMPLE.cpp`: solver parameters and main driver.
- `Utilities/iterations.cpp`: momentum + correction loop (calls helpers below).
- `Utilities/pressure_solver.cpp`: pressure correction (direct SimplicialLDLT or SOR fallback).
- `Utilities/convection.cpp`: SOU deferred corrections.
- `Utilities/masks.cpp`: fluid/solid masks and alpha lookups.
- `Utilities/time_control.cpp`: pseudo time-stepping, CFL ramp, pseudo-dt stats logging.
- `Utilities/boundaries.cpp`: velocity/pressure boundary conditions.
- `Utilities/output.cpp`: I/O for residuals, pressure drop histories, field saves.
- `Utilities/postprocessing.cpp`: pressure-drop sampling planes, reporting.
- `GeometryGenerator.py`: builds geometry and fluid params (writes to `ExportFiles/`).
- `PlotResiduals.py`: interactive plotting of residuals and pressure drops.

## Build
Requires [Eigen](https://eigen.tuxfamily.org/) headers. Example (Windows/MinGW, adjust Eigen path):
```
g++ -std=c++17 -fopenmp SIMPLE.cpp Utilities\boundaries.cpp Utilities\iterations.cpp Utilities\output.cpp Utilities\postprocessing.cpp Utilities\pressure_solver.cpp Utilities\convection.cpp Utilities\time_control.cpp Utilities\masks.cpp -I. -I"C:\Toolbox Coding\eigen-5.0.0" -O3 -o simple.exe
```
On Linux/macOS adjust slashes and Eigen include path:
```
g++ -std=c++17 -fopenmp SIMPLE.cpp Utilities/boundaries.cpp Utilities/iterations.cpp Utilities/output.cpp Utilities/postprocessing.cpp Utilities/pressure_solver.cpp Utilities/convection.cpp Utilities/time_control.cpp Utilities/masks.cpp -I. -I/path/to/eigen -O3 -o simple
```

## How to run
1) Generate geometry/params:
```
python GeometryGenerator.py
```
This writes `ExportFiles/fluid_params.txt` and geometry matrices.
2) Build (see above) and run the solver (`simple.exe` or `./simple`).
3) Postprocess/plot residuals:
```
python PlotResiduals.py
```
Results (fields, histories, VTK) land in `ExportFiles/`.

## Workflow (quick start)
- Generate geometry: `python GeometryGenerator.py`
- Build: g++ command above (adjust Eigen include path)
- Run: `./simple.exe` (or `./simple` on *nix)
- Plot: `python PlotResiduals.py`
- Inspect outputs in `ExportFiles/` (VTK + text + PNGs)

## Tuning knobs (SIMPLE.h)
- Pressure solver: `useDirectPressureSolver` (direct SimplicialLDLT) vs SOR fallback; `useSIMPLEC` toggles SIMPLEC variant.
- Relaxation: `uvAlpha`, `pAlpha`; SOR params: `maxPressureIter`, `sorOmega` (auto if 0), `pTol`.
- Pseudo time/CFL ramp: `enablePseudoTimeStepping`, `enableCflRamp`, `pseudoCFLInitial/Max`, `cflRamp*`, `useLocalPseudoTime`.
- Convection scheme: `convectionScheme = 0/1` (FOU/SOU).
- Iteration limits/convergence: `maxIterations`, `epsilon`, pressure-drop convergence options.
- Inlet ramp: `enableInletRamp`, `rampSteps`.

## Inputs & knobs
- Main parameters live in `SIMPLE.h` (relaxation, CFL ramp, max iterations, convergence eps, ramp steps, etc.).
- Geometry and mesh come from `GeometryGenerator.py` (refinement, channel layout, buffers).
- Pressure solver: toggle direct vs SOR in `SIMPLE.h` (`useDirectPressureSolver`), SIMPLE vs SIMPLEC (`useSIMPLEC`).

## Outputs
- `ExportFiles/residuals.txt`, `pressure_drop_history.txt` (plus PNGs).
- Field dumps: `u*.txt`, `v*.txt`, `p*.txt`, VTK files for visualization.
- Final console summary of residuals and pressure drops (core and full domain).

## Notes
- Steady laminar only; pseudo-transient stepping is for convergence aid, not real-time accuracy.
- All fluid properties are set in `SIMPLE.h` (density/viscosity hardcoded).
- Direct pressure solve uses Eigen SimplicialLDLT; symbolic pattern is reused across iterations for speed. SOR fallback remains available. 

## Flowchart (pipeline)
```mermaid
flowchart TD
  A[GeometryGenerator.py<br/>- set geometry/mesh<br/>- fluid params<br/>- write ExportFiles/] --> B[Build<br/>g++ SIMPLE.cpp + Utilities/*.cpp<br/>-I. -I (Eigen include) -fopenmp -O3]
  B --> C[Run solver (simple.exe)]

  subgraph SIMPLE_LOOP[SIMPLE Loop]
    C --> C1[Load params/geometry<br/>allocate fields/masks<br/>init logs]
    C1 --> C2[Iteration loop]

    subgraph ITERATION[Iteration]
      C2 --> I1[Momentum U/V<br/>FOU + optional SOU<br/>Brinkman masks<br/>solve U, V]
      I1 --> I2[Pseudo-time / CFL ramp<br/>updateCflRamp<br/>log pseudo-dt stats]
      I2 --> I3[Pressure correction<br/>assemble A_p, b_p<br/>Direct (LDLT) or SOR<br/>cache mapping/factor]
      I3 --> I4[Velocity correction<br/>apply dP<br/>BCs]
      I4 --> I5[Residuals & dP<br/>core/full planes]
      I5 --> I6[Logging/output<br/>residuals.txt<br/>dp_history.txt<br/>console row<br/>checkpoint optional]
      I6 --> I7[Convergence check<br/>(residuals or dP window)]
    end

    I7 -->|not done| C2
    I7 -->|done| C3[Finalize<br/>sample final planes<br/>print summary<br/>saveAll fields/VTK]
  end

  C3 --> D[Outputs (ExportFiles/)<br/>residuals.txt, pressure_drop_history.txt<br/>u/v/p txt, VTK, PNGs]

  D --> E[PlotResiduals.py<br/>plot residuals & dP<br/>save PNG<br/>interactive zoom if matplotlib]

  D --> F[Optional Thermal]
  subgraph THERMAL[Thermal Post]
    F --> F1[main.py (Carlos Goni Gill)<br/>load thermal params/fields<br/>call HeatTransferSolver]
    F1 --> F2[heat_solver.py (Carlos Goni Gill)<br/>conjugate two-layer solve<br/>write thermal_results.vtk]
  end
```
