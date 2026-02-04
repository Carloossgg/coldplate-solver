# Thermal Solver Output Files Guide

## Overview

The thermal solver (`thermal_solver.cpp`) generates several output files for visualization, post-processing, and topology optimization. All files are saved to the specified output directory (default: `ExportFiles/`).

---

## Output Files Summary

| File | Type | Purpose |
|------|------|---------|
| `thermal_results_3d.vtk` | VTK | 3D temperature and velocity visualization |
| `thermal_gradient_3d.vtk` | VTK | 3D gradient vector field visualization |
| `thermal_metrics.txt` | Text | Scalar statistics (averages, uniformity, gradients) |
| `k_thermal.txt` | Text | 2D thermal conductivity field for MMA optimizer |
| `objective.txt` | Text | Optimization objective values |
| `dJ_dgamma.txt` | Text | Sensitivity field (requires adjoint solver) |

---

## Detailed File Descriptions

### 1. `thermal_results_3d.vtk`

**Format:** VTK Rectilinear Grid (ASCII)  
**Visualization:** ParaView, VisIt, or any VTK-compatible software

**Contains:**

| Field | Type | Description |
|-------|------|-------------|
| `Temperature` | Scalar (CELL_DATA) | Temperature at each cell [°C] |
| `Region` | Scalar (int) | 0 = solid (base/fin), 1 = fluid |
| `Density` | Scalar | Gamma field: 1 = fluid, 0 = solid, intermediate = buffer |
| `ThermalConductivity` | Scalar | k(γ) from RAMP interpolation [W/m-K] |
| `Velocity` | Vector | 3D velocity field (u, v, w) with parabolic z-profile [m/s] |

**Grid Structure:**
- Rectilinear grid with dimensions (Nx+1) × (Ny+1) × (Nz+1) vertices
- Non-uniform z-spacing: `dz_solid` for base plate, `dz_fluid` for channel region
- Cell data: Nx × Ny × Nz cells

**Usage Example (ParaView):**
1. Open `thermal_results_3d.vtk`
2. Apply "Threshold" filter on `Region` to isolate fluid/solid
3. Color by `Temperature` to visualize thermal distribution
4. Use "Glyph" filter on `Velocity` to show flow direction

---

### 2. `thermal_gradient_3d.vtk`

**Format:** VTK Rectilinear Grid (ASCII)  
**Visualization:** ParaView, VisIt

**Contains:**

| Field | Type | Description |
|-------|------|-------------|
| `TemperatureGradient` | Vector (CELL_DATA) | ∇T = (∂T/∂x, ∂T/∂y, ∂T/∂z) [°C/m] |
| `GradientMagnitude` | Scalar | \|∇T\| = √(∂T/∂x² + ∂T/∂y² + ∂T/∂z²) [°C/m] |

**Gradient Computation Method:**
- Interior cells: Central differences
- Boundary cells: One-sided (forward/backward) differences
- Z-direction: Accounts for non-uniform spacing between base plate and channel

**Usage Example (ParaView):**
1. Open `thermal_gradient_3d.vtk`
2. Color by `GradientMagnitude` to identify high thermal stress regions
3. Apply "Glyph" filter on `TemperatureGradient` to show heat flow direction
4. Use "Threshold" to filter out extreme values (outliers)

**Interpreting Results:**
- High gradient magnitude → regions of thermal stress
- Gradient vectors point from cold to hot regions
- Maximum gradient location indicates potential failure points

---

### 3. `thermal_metrics.txt`

**Format:** Plain text with labeled values  
**Purpose:** Scalar statistics for optimization monitoring and convergence tracking

**Contents:**

```
# Thermal Metrics - Volume-Weighted Statistics
# All temperatures in [°C]

# Average Temperatures
T_avg_base      XX.XXXX    # Substrate average (k < nz_solid)
T_avg_fluid     XX.XXXX    # Fluid region average (gamma > 0.01)
T_avg_solid     XX.XXXX    # Solid fins average (gamma <= 0.01, k >= nz_solid)
T_avg_global    XX.XXXX    # Entire domain average

# Standard Deviations
T_std_base      XX.XXXX
T_std_fluid     XX.XXXX
T_std_global    XX.XXXX

# Temperature Ranges (max - min)
T_range_base    XX.XXXX
T_range_fluid   XX.XXXX
T_range_global  XX.XXXX

# Uniformity Indices (1 = uniform, 0 = non-uniform)
T_uniformity_base   X.XXXX
T_uniformity_global X.XXXX

# Thermal Gradients [°C/m]
grad_max          XXXXX.XXXX    # Maximum |∇T| in domain
grad_avg          XXXXX.XXXX    # Volume-weighted average |∇T|
grad_max_base     XXXXX.XXXX    # Maximum |∇T| in base plate
grad_avg_base     XXXXX.XXXX    # Average |∇T| in base plate
grad_max_loc      i j k         # Cell indices of maximum gradient
```

**Region Definitions:**
- **Base plate:** z-layers where k < nz_solid (always solid substrate)
- **Fluid:** Channel region cells where gamma > 0.01
- **Solid fins:** Channel region cells where gamma ≤ 0.01

**Uniformity Index Formula:**
```
uniformity = 1 - (std_dev / avg)
```
- Value of 1.0 = perfectly uniform temperature
- Value approaching 0 = highly non-uniform

**Parsing Example (Python):**
```python
def parse_thermal_metrics(filepath):
    metrics = {}
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith('#') or not line.strip():
                continue
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1] if len(parts) == 2 else parts[1:]
                try:
                    metrics[key] = float(value) if isinstance(value, str) else [float(v) for v in value]
                except ValueError:
                    metrics[key] = value
    return metrics
```

---

### 4. `k_thermal.txt`

**Format:** Tab-separated values (TSV)  
**Dimensions:** Ny rows × Nx columns  
**Purpose:** 2D thermal conductivity field for MMA optimizer verification

**Grid Layout:**
- Same layout as `geometry_thermal.txt`
- Row j, Column i corresponds to cell (i, j) in the channel region
- Values computed from gamma using RAMP interpolation

**RAMP Interpolation Formula:**
```
k(γ) = k_f + (k_s - k_f) * (1 - γ) / (1 + q_k * γ)

Where:
  k_f = fluid thermal conductivity (default: 0.6 W/m-K for water)
  k_s = solid thermal conductivity (default: 400 W/m-K for copper)
  q_k = RAMP convexity parameter (continuation: 1 → 3 → 10 → 30)
  γ   = design variable (1 = fluid, 0 = solid)
```

**Example Values:**
| γ (gamma) | q_k = 1 | q_k = 10 | q_k = 30 |
|-----------|---------|----------|----------|
| 0.00 | 400.0 | 400.0 | 400.0 |
| 0.25 | 240.1 | 137.1 | 85.5 |
| 0.50 | 133.5 | 36.3 | 13.5 |
| 0.75 | 66.8 | 9.8 | 3.6 |
| 1.00 | 0.6 | 0.6 | 0.6 |

**Loading Example (Python):**
```python
import numpy as np
k_field = np.loadtxt('ExportFiles/k_thermal.txt', delimiter='\t')
# k_field.shape = (Ny, Nx)
```

**Note:** Base plate always uses k_solid and is not included in this 2D export.

---

### 5. `objective.txt`

**Format:** Plain text, one value per line  
**Purpose:** Optimization objective values for MMA

**Contents:**
```
Line 1: Tb_max      # True maximum base temperature [°C]
Line 2: Tb_pnorm    # P-norm approximation of max temperature [°C]
Line 3: p           # P-norm exponent used (typically 10)
Line 4: imbalance   # Energy balance error [%]
```

**P-Norm Formula:**
```
Tb_pnorm = (1/N * Σ Tb_i^p)^(1/p)

Where:
  N = number of substrate cells
  Tb_i = temperature at substrate cell i
  p = norm exponent (higher p → closer to true max)
```

**Why P-Norm?**
- True max() is non-differentiable → problematic for gradient-based optimization
- P-norm is smooth and differentiable everywhere
- As p → ∞, p-norm → true max
- p = 10 provides good balance between smoothness and accuracy

---

### 6. `dJ_dgamma.txt`

**Format:** Tab-separated values (TSV)  
**Dimensions:** Ny rows × Nx columns  
**Purpose:** Sensitivity field for MMA optimizer (gradient of objective w.r.t. design variables)

**Note:** This file is generated by the adjoint solver. If the adjoint solver is not implemented or called, this file may contain zeros or not be generated.

**Contents:**
- Each value represents ∂J/∂γ at that cell
- Negative values → increasing γ (more fluid) decreases objective (good)
- Positive values → increasing γ (more fluid) increases objective (bad)

**Usage in MMA:**
```python
import numpy as np
sensitivity = np.loadtxt('ExportFiles/dJ_dgamma.txt', delimiter='\t')
# Feed to MMA optimizer along with current gamma and objective
```

---

## Console Output Summary

When the thermal solver runs, it prints:

```
=== 3D Conjugate Heat Transfer Solver ===
Reference: Yan et al. (2019) Two-Layer Model
Data directory: ../ExportFiles

Interpolation: RAMP
  q_k = X (continuation: 1→3→10→30)

Loaded geometry_thermal.txt (XXX buffer cells)
Grid: Nx x Ny x Nz = XXXXXXX DOFs

  Assembling matrix (Multicore)... done (X.XXs)
  Building Parallel Preconditioner (once)... X.XXs
  Solving Single Pass (IDR(s), s=8, All Cores)... XXX iters, res=X.XXe-XX (X.XXs)

Results:
  T_base:  [XX.XX, XX.XX] °C
  T_fin:   [XX.XX, XX.XX] °C
  T_fluid: [XX.XX, XX.XX] °C

[Energy Balance]
  Q_in  = XXXX.XX W
  T_out = XX.XX °C (mass-avg)
  Q_out = XXXX.XX W
  Imbalance = X.XX%

Saved: .../thermal_results_3d.vtk

[Average Temperatures]
  T_avg_base   = XX.XX °C (substrate)
  T_avg_fluid  = XX.XX °C (channel fluid)
  T_avg_solid  = XX.XX °C (solid fins)
  T_avg_global = XX.XX °C (entire domain)

[Temperature Uniformity]
  Base plate:
    std_dev = X.XX °C
    range   = XX.XX °C (max-min)
    uniformity = X.XXXX (1=uniform)
  Global:
    std_dev = XX.XX °C
    range   = XXXX.XX °C (max-min)
    uniformity = X.XXXX (1=uniform)

[Thermal Gradients]
  grad_max      = XXXXX.XX °C/m at cell (i, j, k)
  grad_avg      = XXXXX.XX °C/m
  grad_max_base = XXXXX.XX °C/m (substrate)
  grad_avg_base = XXXXX.XX °C/m

[Thermal Conductivity Field]
  k_min = X.XX W/m-K (at gamma=0: XXX.XX)
  k_max = XXX.XX W/m-K (at gamma=1: X.XX)
  k_avg = XXX.XX W/m-K
  Cells: XXXXX fluid, XXXXX solid, XXX buffer
  RAMP parameter q_k = X

Saved: .../thermal_metrics.txt
Saved: .../k_thermal.txt
Saved: .../thermal_gradient_3d.vtk
```

---

## File Dependencies

```
                    ┌─────────────────────┐
                    │  geometry_thermal.txt│ (input)
                    │  thermal_params.txt  │ (input)
                    │  u_thermal.txt       │ (input)
                    │  v_thermal.txt       │ (input)
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  thermal_solver.cpp  │
                    │  thermal_output.cpp  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          │                    │                    │
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│thermal_results  │  │thermal_gradient │  │thermal_metrics  │
│    _3d.vtk      │  │    _3d.vtk      │  │    .txt         │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │  k_thermal.txt  │   │  objective.txt  │
          └─────────────────┘   └─────────────────┘
                    │
                    ▼
          ┌─────────────────┐
          │  dJ_dgamma.txt  │ (requires adjoint)
          └─────────────────┘
```

---

## Common Issues & Troubleshooting

### Extreme Temperature Values in T_fluid
**Symptom:** T_fluid shows unrealistic values (e.g., 4000°C)  
**Cause:** Geometry generator creating isolated or poorly connected fluid cells  
**Impact:** Affects global range, uniformity, and gradient max statistics  
**Workaround:** Focus on T_avg values and base plate metrics; outlier filtering planned

### High Energy Imbalance (>5%)
**Symptom:** Q_in - Q_out differs significantly  
**Possible Causes:**
- Mesh too coarse for the geometry
- Poor convergence (check residual)
- Boundary condition issues at outlet

### Gradient Max is Unrealistically High
**Symptom:** grad_max is millions of °C/m  
**Cause:** Sharp temperature discontinuities from outlier cells  
**Workaround:** Use grad_avg and grad_avg_base for meaningful comparisons

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2025-02 | Initial thermal solver outputs |
| 1.1 | 2025-02 | Added thermal_metrics.txt with averages |
| 1.2 | 2025-02 | Added uniformity metrics (std_dev, range, uniformity index) |
| 1.3 | 2025-02 | Added gradient statistics and thermal_gradient_3d.vtk |
| 1.4 | 2025-02 | Added k_thermal.txt export |

---

## References

1. Yan, S., Wang, F., Hong, J., & Sigmund, O. (2019). *Topology optimization of microchannel heat sinks using a two-layer model.* International Journal of Heat and Mass Transfer, 143, 118462.

2. VTK File Formats: https://vtk.org/wp-content/uploads/2015/04/file-formats.pdf
