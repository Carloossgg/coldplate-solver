import numpy as np
import scipy.ndimage
import matplotlib.pyplot as plt
import os

# =============================================================================
# 1. PHYSICAL GEOMETRY DEFINITIONS
# =============================================================================
# Units: meters

# Overall Domain
Lx_heatsink = 0.0355      # Heatsink/Power map Length
Ly_total    = 0.0475      # Total Height

# =============================================================================
# 2. MESH REFINEMENT SETTINGS
# =============================================================================
REFINEMENT_FACTOR = 2
BASE_CELLS_PER_CHANNEL = 5  # base cells per nominal width (used to set dx)

# Use a nominal width to set dx
w_nominal = 0.0010  # 1.0 mm nominal for resolution choice

target_cells_channel = BASE_CELLS_PER_CHANNEL * REFINEMENT_FACTOR
dx = w_nominal / target_cells_channel
dy = dx

# Grid Dimensions
N_heatsink = int(round(Lx_heatsink / dx))
M_total    = int(round(Ly_total / dy))


print(f"Refinement: {REFINEMENT_FACTOR}x")
print(f"Computed Cell Size (dx): {dx*1e6:.1f} microns")
print(f"Grid Size: {N_heatsink} x {M_total}")

# =============================================================================
# 3. OUTPUT DIRECTORY
# =============================================================================
OUTPUT_DIR = "ExportFiles"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# =============================================================================
# 4. POWER MAP GENERATION
# =============================================================================

# Dimensions: M_total (Height) x N_heatsink (Width)
power_map = np.full((M_total, N_heatsink), 20.0) # Background 20 W/cm2

print("\nSelect Power Map Mode:")
print("1: Hotspots (Default)")
print("2: Uniform")
mode_choice = input("Enter choice (1 or 2): ").strip()
USE_UNIFORM = (mode_choice == '2')

if USE_UNIFORM:
    flux_str = input("Enter Heat Flux [W/cm^2] (default 100): ").strip()
    flux_val = float(flux_str) if flux_str else 100.0
    print(f">> MODE SELECTED: UNIFORM ({flux_val} W/cm^2)")
    power_map.fill(flux_val)
else:
    print(">> MODE SELECTED: HOTSPOTS")
    
    # Hotspot Parameters
    # 6 Hotspots: 2 rows x 3 columns
    rows = 2
    cols = 3
    max_flux = 120.0
    min_flux = 20.0
    
    # Grid for Gaussian (not used for squares, but kept for indices)
    y_indices, x_indices = np.indices((M_total, N_heatsink))
    
    # Centers - adjusted to create small gap while staying close to middle
    row_centers = np.array([M_total * 0.37, M_total * 0.63])
    
    # Spaced out more in X
    col_centers = np.array([N_heatsink * 0.16, N_heatsink * 0.50, N_heatsink * 0.84])
    
    # Hotspot Size (Square Chips) - adjusted height
    h_spot = int((M_total / rows) * 0.40)
    
    # Variable widths - Middle wider, outers narrower
    base_w = N_heatsink / 3
    w_spot_mid = int(base_w * 0.8)  # Middle wider
    w_spot_out = int(base_w * 0.35) # Outers smaller
    w_spots = [w_spot_out, w_spot_mid, w_spot_out]
    
    # --- ADD WARM ZONE ---
    # "Area in between the hotspots including inner edge strips"
    flux_warm = 40.0
    
    # Calculate bounds based on MAIN HOTSPOT centers
    min_r = min(row_centers)
    max_r = max(row_centers)
    
    # Use first and last column centers and their respective widths
    x_warm_0 = max(0, int(col_centers[0] - w_spots[0]/2))
    x_warm_1 = min(N_heatsink, int(col_centers[-1] + w_spots[-1]/2))
    
    # Y bounds - extended to include inner edge strips (at 0.25 and 0.75)
    y_warm_0 = max(0, int(M_total * 0.25 - h_spot/2))
    y_warm_1 = min(M_total, int(M_total * 0.75 + h_spot/2))
    
    power_map[y_warm_0:y_warm_1, x_warm_0:x_warm_1] = np.maximum(power_map[y_warm_0:y_warm_1, x_warm_0:x_warm_1], flux_warm)
    
    # Draw Main Hotspots
    for rc in row_centers:
        for i, cc in enumerate(col_centers):
            # Define Square bounds
            r_int = int(rc)
            c_int = int(cc)
            w_this = w_spots[i]
            
            y0 = max(0, r_int - h_spot // 2)
            y1 = min(M_total, r_int + h_spot // 2)
            x0 = max(0, c_int - w_this // 2)
            x1 = min(N_heatsink, c_int + w_this // 2)
            
            # Set Square to Max Flux
            power_map[y0:y1, x0:x1] = max_flux
    
    # --- ADD TOP/BOTTOM EDGE STRIPS ---
    # Thin rectangles aligned with the columns, matching variable widths
    h_spot_small = int(h_spot * 0.25) # 25% of the main hotspot height
    flux_edge = 75.0
    
    # Y-centers for edge rows (near top and bottom) - 4 rows total (2 top, 2 bottom)
    # Positioned with larger gap between pairs and margin from edges
    y_centers_edge = [int(M_total * 0.08), int(M_total * 0.2), int(M_total * 0.8), int(M_total * 0.92)] 
    
    for rc in y_centers_edge:
        for i, cc in enumerate(col_centers):
            r_int = int(rc)
            c_int = int(cc)
            w_this = w_spots[i] # MATCHING WIDTH of the main hotspot in that column
            
            y0 = max(0, r_int - h_spot_small // 2)
            y1 = min(M_total, r_int + h_spot_small // 2)
            x0 = max(0, c_int - w_this // 2)
            x1 = min(N_heatsink, c_int + w_this // 2)
            
            power_map[y0:y1, x0:x1] = np.maximum(power_map[y0:y1, x0:x1], flux_edge)
    
    
    # Ensure everything is at least min_flux
    power_map = np.maximum(power_map, min_flux)
    
    # Smooth the edges to create a transition (Gaussian Blur)
    sigma_phys = 0.001
    sigma_smooth = sigma_phys / dx
    power_map = scipy.ndimage.gaussian_filter(power_map, sigma=sigma_smooth)

    # Re-clamp to ensure boundaries
    power_map = np.clip(power_map, min_flux, max_flux)

# --- CALCULATE TOTAL POWER ---
# Flux is in W/cm^2
# Cell area in cm^2 = (dx * 100) * (dy * 100)
cell_area_cm2 = (dx * 100) * (dy * 100)
total_power_watts = np.sum(power_map) * cell_area_cm2

print(f"\n{'='*30}")
print(f"TOTAL ESTIMATED POWER: {total_power_watts:.2f} Watts")
print(f"{'='*30}\n")

np.savetxt(os.path.join(OUTPUT_DIR, "power_map.txt"), power_map, fmt='%.2f', delimiter='\t')

print(f"Power map saved to {OUTPUT_DIR}/power_map.txt")

# =============================================================================
# 5. VISUALIZATION
# =============================================================================
plt.figure(figsize=(10, 5))

plt.imshow(power_map, cmap='inferno', origin='lower', interpolation='nearest', 
           extent=[0, Lx_heatsink*1000, 0, Ly_total*1000])
plt.title(f"Power Map (Heat Flux W/cm²) - Max: {np.max(power_map):.1f} W/cm² | Total: {total_power_watts:.0f} W")
plt.xlabel("X (mm)")
plt.ylabel("Y (mm)")
plt.colorbar(label="Heat Flux (W/cm²)")

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'power_map.png'))
plt.show()
