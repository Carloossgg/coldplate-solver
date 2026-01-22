"""
File: GeometryGeneratorV3.py
Author: Peter Tcherkezian
Description: Generates structured channel geometry with DENSITY-BASED topology optimization support.
  Instead of binary solid/fluid (0/1), outputs a continuous density field (gamma) where:
    - gamma = 1.0 → pure fluid
    - gamma = 0.0 → pure solid
    - 0 < gamma < 1 → transition/buffer zone
  
  The buffer zone creates a smooth transition between solid and fluid regions,
  which is essential for gradient-based topology optimization (MMA) and prevents
  numerical issues with abrupt material property changes.
  
  Based on: "Topology optimization of microchannel heat sinks using a two-layer model"
  Brinkman penalization: F(gamma) = -alpha * u * I_alpha(gamma)
  where I_alpha(gamma) = (1 - gamma) / (1 + b_alpha * gamma)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import distance_transform_edt
import os

# =============================================================================
# 1. PHYSICAL GEOMETRY DEFINITIONS (FIXED CONSTANTS)
# =============================================================================
# Units: meters

# Overall Domain
Lx_inlet    = 0.0       # No inlet buffer (set to 0.010 for 10mm buffer)
Lx_heatsink = 0.060     # 60 mm Heatsink Length
Lx_outlet   = 0.0       # No outlet buffer (set to 0.010 for 10mm buffer)
Ly_total    = 0.030     # 30 mm Total Height

# Out-of-plane channel height for 2.5D model (Ht in governing equations)
Ht_channel  = 4/1000    # 4 mm fin height

# Channel Features
w_channel    = 0.6/1000   # 0.6 mm Channel Width
num_channels = 20         # Number of channels

# Pitch (Distance between channel centers)
pitch = Ly_total / num_channels 

# Inlet/outlet channel extension
inlet_channel_extension_frac = 1.0
inlet_plenum_fraction = max(0.0, min(1.0, 1.0 - inlet_channel_extension_frac))

# Flow settings
U_inlet_phys = 1.0      # m/s
HeatFlux     = 100.0    # W/cm^2

# =============================================================================
# 2. DENSITY-BASED TOPOLOGY OPTIMIZATION SETTINGS
# =============================================================================
# Number of buffer cells for the transition zone between solid and fluid
# This creates a smooth gradient instead of an abrupt 0→1 jump
# Recommended: 2-4 cells for good numerical stability without excessive diffusion

NUM_BUFFER_CELLS = 2    # <--- ADJUSTABLE: Number of transition cells on each side

# Brinkman penalization parameters (from paper)
# I_alpha(gamma) = (1 - gamma) / (1 + b_alpha * gamma)
# Higher b_alpha → sharper transition (more nonlinear)
BRINKMAN_CONVEXITY = 0.1  # b_alpha parameter (start low, increase during optimization)

# =============================================================================
# 3. MESH REFINEMENT SETTINGS
# =============================================================================
REFINEMENT_FACTOR = 4   # <--- CHANGE THIS TO 1, 2, 4, 8
BASE_CELLS_PER_CHANNEL = 5 

# Calculate cell size
target_cells_channel = BASE_CELLS_PER_CHANNEL * REFINEMENT_FACTOR
dx = w_channel / target_cells_channel
dy = dx  # Square cells

print("-" * 60)
print(f"GEOMETRY GENERATION V3 (Density-Based)")
print(f"Refinement: {REFINEMENT_FACTOR}x, Buffer Cells: {NUM_BUFFER_CELLS}")
print("-" * 60)
print(f"Physical Channel Width: {w_channel*1000:.3f} mm")
print(f"Target Cells per Width: {target_cells_channel}")
print(f"Computed Cell Size (dx): {dx*1e6:.1f} microns")
print(f"Buffer Zone Width: {NUM_BUFFER_CELLS * dx * 1e6:.1f} microns")

# Grid Dimensions
N_inlet    = int(round(Lx_inlet / dx))
N_heatsink = int(round(Lx_heatsink / dx))
N_outlet   = int(round(Lx_outlet / dx))
N_total    = N_inlet + N_heatsink + N_outlet

M_total    = int(round(Ly_total / dy))

N_inlet_plenum = int(round(N_inlet * inlet_plenum_fraction))

print(f"Grid Size: {N_total} x {M_total}")

# Thermal cropping indices
x_start_fin = N_inlet
x_end_fin   = N_inlet + N_heatsink

# =============================================================================
# 4. BUILD BINARY GEOMETRY MATRIX (Base geometry before density smoothing)
# =============================================================================
# 1 = Solid, 0 = Fluid (binary)
geometry_binary = np.ones((M_total, N_total), dtype=int)

# Create Inlet Plenum (Fully Open)
if N_inlet_plenum > 0:
    geometry_binary[:, 0:N_inlet_plenum] = 0

# Carve Channels
for k in range(num_channels):
    y_center_phys = (k + 0.5) * pitch 
    y_bot_phys = y_center_phys - (w_channel / 2.0)
    y_top_phys = y_center_phys + (w_channel / 2.0)
    
    j_bot = int(round(y_bot_phys / dy))
    j_top = int(round(y_top_phys / dy))
    
    # Clamp to domain (keep top/bottom walls solid)
    j_bot = max(1, j_bot)            
    j_top = min(M_total - 1, j_top)  
    
    # Carve Fluid (Set to 0)
    geometry_binary[j_bot:j_top, N_inlet_plenum : N_total] = 0

# Ensure Top/Bottom Walls are Solid (boundary condition - always binary)
geometry_binary[0, :] = 1
geometry_binary[-1, :] = 1

# =============================================================================
# 5. CREATE DENSITY FIELD WITH BUFFER ZONES
# =============================================================================
# gamma = 1.0 → fluid, gamma = 0.0 → solid
# Use distance transform to create smooth transitions

def create_density_field(binary_geom, num_buffer_cells):
    """
    Convert binary geometry (0=fluid, 1=solid) to continuous density field.
    gamma = 1.0 → fluid, gamma = 0.0 → solid
    
    Uses signed distance transform to create smooth buffer zones.
    """
    M, N = binary_geom.shape
    
    # Invert: fluid=1, solid=0 for distance transform
    fluid_mask = (binary_geom == 0).astype(float)
    solid_mask = (binary_geom == 1).astype(float)
    
    # Distance from each cell to nearest fluid cell
    dist_to_fluid = distance_transform_edt(solid_mask)
    # Distance from each cell to nearest solid cell
    dist_to_solid = distance_transform_edt(fluid_mask)
    
    # Signed distance: positive in solid, negative in fluid
    signed_dist = dist_to_fluid - dist_to_solid
    
    # Create gamma field with smooth transition
    # gamma = 1 (fluid) when signed_dist <= -buffer
    # gamma = 0 (solid) when signed_dist >= +buffer
    # Linear interpolation in between
    
    buffer = float(num_buffer_cells)
    
    if buffer > 0:
        # Normalize to [-1, 1] range within buffer zone
        gamma = 0.5 - 0.5 * np.clip(signed_dist / buffer, -1.0, 1.0)
    else:
        # No buffer: binary
        gamma = fluid_mask
    
    # Enforce strict boundaries (top/bottom walls always solid)
    gamma[0, :] = 0.0
    gamma[-1, :] = 0.0
    
    # Enforce inlet/outlet are pure fluid (no buffer at domain boundaries)
    # Left boundary (inlet): if it was fluid, keep it pure fluid
    gamma[:, 0] = np.where(binary_geom[:, 0] == 0, 1.0, 0.0)
    # Right boundary (outlet): if it was fluid, keep it pure fluid  
    gamma[:, -1] = np.where(binary_geom[:, -1] == 0, 1.0, 0.0)
    
    return gamma

# Create the density field
gamma = create_density_field(geometry_binary, NUM_BUFFER_CELLS)

# Statistics
num_pure_fluid = np.sum(gamma > 0.99)
num_pure_solid = np.sum(gamma < 0.01)
num_buffer = np.sum((gamma >= 0.01) & (gamma <= 0.99))
print(f"\nDensity Field Statistics:")
print(f"  Pure Fluid (gamma > 0.99): {num_pure_fluid} cells ({100*num_pure_fluid/(M_total*N_total):.1f}%)")
print(f"  Pure Solid (gamma < 0.01): {num_pure_solid} cells ({100*num_pure_solid/(M_total*N_total):.1f}%)")
print(f"  Buffer Zone (0.01 <= gamma <= 0.99): {num_buffer} cells ({100*num_buffer/(M_total*N_total):.1f}%)")

# =============================================================================
# 6. EXPORT FILES
# =============================================================================
OUTPUT_DIR = "ExportFiles"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. fluid_params.txt
with open(os.path.join(OUTPUT_DIR, "fluid_params.txt"), "w") as f:
    # Columns: M N dy dx U_inlet N_inlet N_outlet Ht_channel
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {U_inlet_phys} {N_inlet} {N_outlet} {Ht_channel:.9f}")

# 2. geometry_fluid.txt (continuous gamma field: 0=fluid, 1=solid, intermediate=buffer)
# Convert from gamma (1=fluid, 0=solid) to geometry convention (0=fluid, 1=solid)
geometry_continuous = 1.0 - gamma
np.savetxt(os.path.join(OUTPUT_DIR, "geometry_fluid.txt"), geometry_continuous, fmt='%.6f', delimiter='\t')

# 3. Thermal exports
gamma_therm = gamma[:, x_start_fin : x_end_fin]
geometry_therm_continuous = 1.0 - gamma_therm  # Convert to geometry convention
q_flux_wm2 = HeatFlux * 10000.0 

with open(os.path.join(OUTPUT_DIR, "thermal_params.txt"), "w") as f:
    f.write(f"{M_total} {N_heatsink} {dy:.9f} {dx:.9f} {q_flux_wm2} {Ht_channel:.9f}")

np.savetxt(os.path.join(OUTPUT_DIR, "geometry_thermal.txt"), geometry_therm_continuous, fmt='%.6f', delimiter='\t')

print(f"\nFiles saved to {OUTPUT_DIR}/")
print(f"  - fluid_params.txt")
print(f"  - geometry_fluid.txt (continuous: 0=fluid, 1=solid, {100*num_buffer/(M_total*N_total):.1f}% buffer)")
print(f"  - thermal_params.txt, geometry_thermal.txt")

# =============================================================================
# 7. VISUALIZATION
# =============================================================================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Binary geometry
ax1 = axes[0]
im1 = ax1.imshow(geometry_binary, cmap='gray_r', origin='lower', interpolation='nearest')
ax1.set_title(f"Binary Geometry (Refinement {REFINEMENT_FACTOR}x)")
ax1.set_xlabel("X (Cells)")
ax1.set_ylabel("Y (Cells)")
plt.colorbar(im1, ax=ax1, label="0=Fluid, 1=Solid")

# Density field
ax2 = axes[1]
im2 = ax2.imshow(gamma, cmap='RdYlBu', origin='lower', interpolation='nearest', vmin=0, vmax=1)
ax2.set_title(f"Density Field (Buffer={NUM_BUFFER_CELLS} cells)")
ax2.set_xlabel("X (Cells)")
ax2.set_ylabel("Y (Cells)")
plt.colorbar(im2, ax=ax2, label="gamma (0=Solid, 1=Fluid)")

# Add markers
for ax in axes:
    if N_inlet_plenum > 0:
        ax.axvline(x=N_inlet_plenum, color='r', linestyle='--', alpha=0.7)
    if N_inlet > 0:
        ax.axvline(x=N_inlet, color='g', linestyle='--', alpha=0.7)
    if N_inlet + N_heatsink < N_total:
        ax.axvline(x=N_inlet + N_heatsink, color='b', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig('geometry_preview_v3.png', dpi=150)
plt.show()

print("\nDensity-based geometry generation complete!")
print(f"Use density_field.txt for topology optimization (MMA).")
print(f"Brinkman interpolation: I_alpha(gamma) = (1-gamma)/(1 + {BRINKMAN_CONVEXITY}*gamma)")
