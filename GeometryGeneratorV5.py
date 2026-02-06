"""
GeometryGeneratorV5.py

Recreates the V5 straight-channel benchmark geometry:
- 60 mm long, 0.6 mm tall
- 20 cells across channel height
- all cells fluid (no solid cells in geometry file)
"""

import os
import numpy as np
import matplotlib.pyplot as plt


# Physical geometry
Lx_total = 0.060     # [m]
Ly_total = 0.0006    # [m]
Ht_channel = 0.004     # [m], pure 2D export

# Flow/thermal metadata for solver input files
U_inlet_phys = 1.0   # [m/s]
HeatFlux = 0.0       # [W/cm^2]

# Resolution target: 20 cells across height
cells_per_width = 20
dx = Ly_total / cells_per_width
dy = dx

M_total = int(round(Ly_total / dy))   # Ny = 20
N_total = int(round(Lx_total / dx))   # Nx = 2000

print("-" * 60)
print("GEOMETRY GENERATION V5 (Straight Channel, No Solid Cells)")
print("-" * 60)
print(f"Lx: {Lx_total*1000:.1f} mm, Ly: {Ly_total*1000:.1f} mm")
print(f"Cells: Nx={N_total}, Ny={M_total}")
print(f"dx=dy={dx*1e6:.1f} microns")
print(f"Total cells: {N_total * M_total}")

# 1 = solid, 0 = fluid -> all fluid for this benchmark
geometry = np.zeros((M_total, N_total), dtype=int)

# Solver convention:
# - geometry_fluid.txt stores cellType (0=fluid, 1=solid)
# - solver density field is gamma = 1 - cellType
density_gamma = 1.0 - geometry.astype(float)

# Hard guard: V5 benchmark must be fully fluid (gamma == 1 everywhere)
if not np.all(geometry == 0):
    raise RuntimeError("V5 generation error: geometry contains non-fluid cells.")
if not np.allclose(density_gamma, 1.0):
    raise RuntimeError("V5 generation error: density gamma is not 1 everywhere.")

output_dir = "ExportFiles"
os.makedirs(output_dir, exist_ok=True)

# fluid_params.txt columns:
# M N dy dx U_inlet N_inlet N_outlet Ht_channel
with open(os.path.join(output_dir, "fluid_params.txt"), "w") as f:
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {U_inlet_phys} 0 0 {Ht_channel:.9f}")

np.savetxt(os.path.join(output_dir, "geometry_fluid.txt"), geometry, fmt="%d", delimiter="\t")
np.savetxt(os.path.join(output_dir, "density_gamma_v5.txt"), density_gamma, fmt="%.1f", delimiter="\t")

q_flux_wm2 = HeatFlux * 10000.0
with open(os.path.join(output_dir, "thermal_params.txt"), "w") as f:
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {q_flux_wm2} {Ht_channel:.9f}")

np.savetxt(os.path.join(output_dir, "geometry_thermal.txt"), geometry, fmt="%d", delimiter="\t")

print(f"Files saved to {output_dir}/")
print("Density check (gamma): min=1.0, max=1.0 (all fluid)")

# Quick preview
plt.figure(figsize=(12, 3))
plt.imshow(geometry, cmap="gray_r", origin="lower", interpolation="nearest", aspect="auto")
plt.title(f"V5 Geometry ({N_total}x{M_total}) - All Fluid")
plt.xlabel("X cell index")
plt.ylabel("Y cell index")
plt.colorbar(label="0=Fluid, 1=Solid")
plt.tight_layout()
plt.savefig("geometry_v5_preview.png")
print("Preview saved: geometry_v5_preview.png")
