"""
File: GeometryGeneratorV5.py
Author: Antigravity (Assistant) for Peter Tcherkezian
Description: 
    Generates a single straight channel geometry (all-fluid cells) with:
    - Length (X): 60 mm (0.060 m)
    - Height (Y): 0.6 mm (0.0006 m)
    - Depth (Z): 0 mm (Pure 2D)
    
    Grid Specs (aligned with STAR/Gmsh setup):
    - Cells across height (Ny): 20
    - Element size (dx, dy): 30 microns (0.00003 m)
    - Streamwise cells (Nx): 2000
    - Total Cells: 40,000 (2000 x 20)
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. PHYSICAL GEOMETRY DEFINITIONS
# =============================================================================
Lx_total = 0.060    # 60 mm Length
Ly_total = 0.0006   # 0.6 mm Height
Ht_channel = 0.0    # Pure 2D (Depth Z = 0)

# Flow settings
U_inlet_phys = 1.0  # m/s (Standard inlet condition)
HeatFlux = 0.0      # W/cm^2 (Default to adiabatic unless specified)

# No buffer zones for this benchmark
Lx_inlet = 0.0
Lx_outlet = 0.0

# =============================================================================
# 2. DISCRETIZATION (EXACT SPECIFICATION)
# =============================================================================
# Match STAR/Gmsh reference:
# cells_per_width = 20 -> dx = dy = Ly / 20 = 30 microns
cells_per_width = 20
dx = Ly_total / cells_per_width
dy = dx

M_total = int(round(Ly_total / dy))   # 20 cells across channel height
N_total = int(round(Lx_total / dx))   # 2000 streamwise cells

Ly_total_domain = M_total * dy

print("-" * 60)
print(f"GEOMETRY GENERATION V5 (Single Channel - All Fluid)")
print("-" * 60)
print(f"Fluid Channel Length (X): {Lx_total*1000:.1f} mm")
print(f"Fluid Channel Height (Y): {Ly_total*1000:.1f} mm ({M_total} cells)")
print(f"Cell size: {dx*1e6:.1f} microns")
print(f"Grid Size: {N_total} x {M_total} (all fluid cells)")
print(f"Total Domain Height: {Ly_total_domain*1000:.3f} mm")
print(f"Total Cells: {N_total * M_total}")

# =============================================================================
# 3. BUILD GEOMETRY MATRIX
# =============================================================================
# 1 = Solid, 0 = Fluid
# Initializing as fluid (0)
geometry = np.zeros((M_total, N_total), dtype=int)

# =============================================================================
# 4. EXPORT FILES
# =============================================================================
OUTPUT_DIR = "ExportFiles"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. fluid_params.txt
# Columns: M N dy dx U_inlet N_inlet N_outlet Ht_channel
# Set N_inlet/N_outlet to 0 as we have no buffers
with open(os.path.join(OUTPUT_DIR, "fluid_params.txt"), "w") as f:
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {U_inlet_phys} 0 0 {Ht_channel:.9f}")

# 2. geometry_fluid.txt
np.savetxt(os.path.join(OUTPUT_DIR, "geometry_fluid.txt"), geometry, fmt='%d', delimiter='\t')

# 3. thermal_params.txt & geometry_thermal.txt (Full domain as heatsink)
q_flux_wm2 = HeatFlux * 10000.0 
with open(os.path.join(OUTPUT_DIR, "thermal_params.txt"), "w") as f:
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {q_flux_wm2} {Ht_channel:.9f}")

np.savetxt(os.path.join(OUTPUT_DIR, "geometry_thermal.txt"), geometry, fmt='%d', delimiter='\t')

print(f"Files saved to {OUTPUT_DIR}/")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
plt.figure(figsize=(12, 4))
plt.imshow(geometry, cmap='gray_r', origin='lower', interpolation='nearest', aspect='auto')
plt.title(f"Single Channel Geometry ({N_total}x{M_total})\nAll cells are Fluid (0)")
plt.xlabel("Streamwise Cells (X)")
plt.ylabel("Cross-stream Cells (Y)")
plt.colorbar(label="0=Fluid, 1=Solid")
plt.tight_layout()
plt.savefig('geometry_v5_preview.png')
print("Preview saved as geometry_v5_preview.png")
# plt.show()
