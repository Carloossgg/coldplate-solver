"""
File: GeometryGenerator.py
Author: Peter Tcherkezian
Description: Generates structured channel geometry and fluid parameter exports for the SIMPLE CFD solver:
  defines physical dimensions and channel layout, discretizes to a structured grid, and writes geometry/fluid property
  text files to ExportFiles/ for the C++ solver and downstream plotting.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. PHYSICAL GEOMETRY DEFINITIONS (FIXED CONSTANTS)
# =============================================================================
# These values define your "Real World" object. They NEVER change.
# Units: Meters

# Overall Domain
Lx_inlet    =0 #0.010      # 10 mm Inlet Buffer
Lx_heatsink = 0.060      # 40 mm Heatsink Length
Lx_outlet   = 0 #0.010      # 10 mm Outlet Buffer
Ly_total    = 0.030      # 30 mm Total Height

# Out-of-plane channel height for 2.5D model (Ht in governing equations)
# This is exported so the C++ solver can pick it up automatically (no manual edits).
Ht_channel  = 4/1000     # 4 mm fin height

# Channel Features
w_channel    = 0.6/1000    # 0.8 mm Channel Width (Exact)
num_channels = 20        # Number of channels

# Pitch (Distance between channel centers)
# Evenly spaced in Y
pitch = Ly_total / num_channels 

# --- INLET/OUTLET CHANNEL LOGIC (YOUR SPECIFIC LOGIC) ---
# Fraction of inlet length that extends the channels (vs open plenum)
# 0.2 means 20% of the inlet is channels, 80% is open plenum.
inlet_channel_extension_frac = 1 
inlet_plenum_fraction = max(0.0, min(1.0, 1.0 - inlet_channel_extension_frac))

# Flow settings (rho/eta fixed in SIMPLE.h; not exported from here)
U_inlet_phys = 1       # m/s
# Reference fluid properties (for documentation): rho=997 kg/m^3, eta=0.00089 Pa.s
HeatFlux     = 100     # W/cm^2

# =============================================================================
# 2. MESH REFINEMENT SETTINGS
# =============================================================================
# This is the ONLY knob you turn to change mesh quality.
# Factor 1 = Base Resolution (e.g., 5 cells across channel)
# Factor 4 = High Resolution (e.g., 20 cells across channel)

REFINEMENT_FACTOR = 4  # <--- CHANGE THIS TO 1, 2, 4, 8

# Base resolution: How many cells minimum to resolve 1.0 mm channel?
BASE_CELLS_PER_CHANNEL = 5 

# =============================================================================
# 3. DISCRETIZATION LOGIC
# =============================================================================

# Calculate exact cell size (dx = dy) to ensure channel width is an integer
target_cells_channel = BASE_CELLS_PER_CHANNEL * REFINEMENT_FACTOR
dx = w_channel / target_cells_channel
dy = dx # Square cells

print("-" * 60)
print(f"GEOMETRY GENERATION (Refinement: {REFINEMENT_FACTOR}x)")
print("-" * 60)
print(f"Physical Channel Width: {w_channel*1000:.3f} mm")
print(f"Target Cells per Width: {target_cells_channel}")
print(f"Computed Cell Size (dx): {dx*1e6:.1f} microns")

# Calculate Grid Dimensions (Rounding to nearest integer)
N_inlet    = int(round(Lx_inlet / dx))
N_heatsink = int(round(Lx_heatsink / dx))
N_outlet   = int(round(Lx_outlet / dx))
N_total    = N_inlet + N_heatsink + N_outlet

M_total    = int(round(Ly_total / dy))

# --- Calculate Split Inlet Regions based on your logic ---
N_inlet_plenum = int(round(N_inlet * inlet_plenum_fraction))
# The rest of the inlet is "Channel Extension"
# Channels start at N_inlet_plenum

print(f"Grid Size: {N_total} x {M_total}")
print(f"Inlet Plenum Ends at Cell: {N_inlet_plenum}")

# =============================================================================
# 4. BUILD GEOMETRY MATRIX
# =============================================================================
# 1 = Solid, 0 = Fluid
geometry = np.ones((M_total, N_total), dtype=int)

# --- A. Create Inlet Plenum (Fully Open) ---
# From x=0 to x=N_inlet_plenum, the whole height is fluid
if N_inlet_plenum > 0:
    geometry[:, 0:N_inlet_plenum] = 0

# --- B. Carve Channels ---
# Channels start where the plenum ends (N_inlet_plenum)
# And they go ALL THE WAY to the end (N_total), covering:
# 1. The Inlet Extension
# 2. The Heatsink
# 3. The Outlet Buffer (which acts as extended channels in your logic)

# Fluid Region X-indices for thermal solver cropping (just for reference)
x_start_fin = N_inlet
x_end_fin   = N_inlet + N_heatsink

for k in range(num_channels):
    # 1. Calculate Physical Center of Channel k
    y_center_phys = (k + 0.5) * pitch 
    
    # 2. Calculate Top and Bottom Edges (Physical)
    y_bot_phys = y_center_phys - (w_channel / 2.0)
    y_top_phys = y_center_phys + (w_channel / 2.0)
    
    # 3. Convert to Grid Indices
    j_bot = int(round(y_bot_phys / dy))
    j_top = int(round(y_top_phys / dy))
    
    # 4. Clamp to domain
    j_bot = max(1, j_bot)            
    j_top = min(M_total - 1, j_top)  
    
    # 5. Carve Fluid (Set to 0)
    # Start at N_inlet_plenum, go to N_total
    geometry[j_bot:j_top, N_inlet_plenum : N_total] = 0

# Ensure Top/Bottom Walls are Solid
geometry[0, :] = 1
geometry[-1, :] = 1

# =============================================================================
# 5. EXPORT FILES
# =============================================================================
OUTPUT_DIR = "ExportFiles"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# 1. fluid_params.txt
with open(os.path.join(OUTPUT_DIR, "fluid_params.txt"), "w") as f:
    # Columns: M N dy dx U_inlet N_inlet N_outlet Ht_channel (rho/eta are hardcoded in SIMPLE.h)
    f.write(f"{M_total} {N_total} {dy:.9f} {dx:.9f} {U_inlet_phys} {N_inlet} {N_outlet} {Ht_channel:.9f}")



# 2. geometry_fluid.txt
np.savetxt(os.path.join(OUTPUT_DIR, "geometry_fluid.txt"), geometry, fmt='%d', delimiter='\t')

# 3. thermal_params.txt & geometry_thermal.txt
# We define the thermal domain as the Heatsink Region Only
# This CROPS the geometry to exclude Inlet and Outlet buffers.
geo_therm = geometry[:, x_start_fin : x_end_fin]
q_flux_wm2 = HeatFlux * 10000.0 

with open(os.path.join(OUTPUT_DIR, "thermal_params.txt"), "w") as f:
    # Note: We write N_heatsink here because we are cropping the geometry
    f.write(f"{M_total} {N_heatsink} {dy:.9f} {dx:.9f} {q_flux_wm2} {Ht_channel:.9f}")

np.savetxt(os.path.join(OUTPUT_DIR, "geometry_thermal.txt"), geo_therm, fmt='%d', delimiter='\t')

print(f"Files saved to {OUTPUT_DIR}/")

# =============================================================================
# 6. VISUALIZATION
# =============================================================================
plt.figure(figsize=(10, 6))
plt.imshow(geometry, cmap='gray_r', origin='lower', interpolation='nearest')
plt.title(f"Geometry (Refinement {REFINEMENT_FACTOR})\nInlet Frac: {inlet_channel_extension_frac}, Outlet: Extended")
plt.xlabel("X (Cells)")
plt.ylabel("Y (Cells)")
# Markers
plt.axvline(x=N_inlet_plenum, color='r', linestyle='--', label='Plenum End')
plt.axvline(x=N_inlet, color='g', linestyle='--', label='Heatsink Start (Thermal)')
plt.axvline(x=N_inlet+N_heatsink, color='b', linestyle='--', label='Heatsink End (Thermal)')
plt.colorbar(label="0=Fluid, 1=Solid")
plt.legend()
plt.tight_layout()
plt.show()