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
  A[GeometryGenerator.py\n- set geometry/mesh\n- fluid params\n- write ExportFiles/] --> B[Build\n g++ SIMPLE.cpp + Utilities/*.cpp\n -I. -I<Eigen> -fopenmp -O3]
  B --> C[Run simple(.exe)]

  subgraph SIMPLE Loop
    C --> C1[Load params/geometry\nallocate fields/masks\ninit logs]
    C1 --> C2[Iteration loop]

    subgraph Iteration
      C2 --> I1[Momentum U/V\nFOU + optional SOU\nBrinkman masks\nsolve U, V]
      I1 --> I2[Pseudo-time / CFL ramp\nupdateCflRamp\nlog pseudo-dt stats]
      I2 --> I3[Pressure correction\nassemble A_p, b_p\nDirect (LDLT) or SOR\ncache mapping/factor]
      I3 --> I4[Velocity correction\napply dP\nBCs]
      I4 --> I5[Residuals & dP\ncore/full planes]
      I5 --> I6[Logging/output\nresiduals.txt\ndp_history.txt\nconsole row\ncheckpoint optional]
      I6 --> I7[Convergence check\n(residuals or dP window)]
    end

    I7 -->|not done| C2
    I7 -->|done| C3[Finalize\nsample final planes\nprint summary\nsaveAll fields/VTK]
  end

  C3 --> D[Outputs (ExportFiles/)\nresiduals.txt, pressure_drop_history.txt\nu/v/p txt, VTK, PNGs]

  D --> E[PlotResiduals.py\nplot residuals & dP\nsave PNG\ninteractive zoom if matplotlib]

  D --> F[Optional Thermal]
  subgraph Thermal Post
    F --> F1[main.py (Carlos Goni Gill)\nload thermal params/fields\ncall HeatTransferSolver]
    F1 --> F2[heat_solver.py (Carlos Goni Gill)\nconjugate two-layer solve\nwrite thermal_results.vtk]
  end
```