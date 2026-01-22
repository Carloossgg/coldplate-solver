"""
File: ChannelWidthField.py
Author: Peter Tcherkezian (+ChatGPT)
Updated: To support Variable Width Channels (Local Propagation) with Robustness Improvements

Description:
  Reads the structured-grid geometry exported by GeometryGenerator.py.
  Computes a "Channel Width Field" that captures LOCAL variations (e.g., bulges, tapers).

  Algorithm Improvements ("Foolproof"):
    1. Pre-processing: Cleans geometry using binary opening to remove single-pixel noise/spurs.
    2. Ridge Detection: Samples the local maximum of the Distance Transform near the skeleton
       to correct for grid-alignment errors.
    3. Propagation: Uses Nearest-Neighbor (Voronoi) propagation for physical accuracy.

  Exports:
    - ExportFiles/channel_width.vtk (Legacy VTK for ParaView)

Dependencies:
  pip install numpy scipy scikit-image
"""

import os
import numpy as np

# --- Required deps ---
try:
    from scipy.ndimage import distance_transform_edt, maximum_filter
except Exception as e:
    raise ImportError("scipy is required: pip install scipy") from e

try:
    from skimage.morphology import medial_axis, binary_opening, disk
except Exception as e:
    raise ImportError("scikit-image is required: pip install scikit-image") from e


# =============================================================================
# Settings
# =============================================================================
EXPORT_DIR = "ExportFiles"
GEOM_FILE = os.path.join(EXPORT_DIR, "geometry_fluid.txt")
FLUID_PARAMS_FILE = os.path.join(EXPORT_DIR, "fluid_params.txt")
VTK_OUT = os.path.join(EXPORT_DIR, "channel_width.vtk")

SOLID_THRESHOLD = 0.5
CLEANUP_RADIUS = 1  # Radius for noise removal (pixels). 1 is safe for most grids.

# =============================================================================
# I/O helpers
# =============================================================================

def read_fluid_params(path: str):
    with open(path, "r") as f:
        parts = f.read().strip().split()
    if len(parts) < 4:
        raise ValueError(f"fluid_params.txt invalid. Got: {parts}")
    M = int(parts[0])
    N = int(parts[1])
    dy = float(parts[2])
    dx = float(parts[3])
    return M, N, dy, dx

def write_vtk_structured_points(path: str, nx: int, ny: int, dx: float, dy: float, point_data: dict):
    """Writes robust Legacy VTK format using np.savetxt for speed."""
    n_points = nx * ny
    print(f"Writing VTK to {path}...")
    
    with open(path, "w") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Channel Width Field (Local Propagation)\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} 1\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx:.9f} {dy:.9f} 1\n")
        f.write(f"POINT_DATA {n_points}\n")

        for name, arr in point_data.items():
            if arr.shape != (ny, nx):
                raise ValueError(f"Shape mismatch for {name}: {arr.shape}")
            
            f.write(f"SCALARS {name} double 1\n")
            f.write("LOOKUP_TABLE default\n")
            
            # Flatten and handle NaNs for ParaView compatibility
            flat = arr.astype(np.float64).ravel(order="C")
            flat = np.nan_to_num(flat, nan=-1.0)
            
            np.savetxt(f, flat, fmt='%.6e')
            
    print("VTK write complete.")

# =============================================================================
# Core computation
# =============================================================================

def compute_local_channel_width(geom: np.ndarray, dx: float, dy: float, solid_threshold: float):
    """
    Computes local channel width by propagating skeleton width values 
    to the surrounding fluid via nearest-neighbor interpolation.
    
    Includes robust pre-processing to handle noisy geometries.
    """
    M, N = geom.shape
    
    # 1. Identify Fluid vs Solid
    solid_frac = np.clip(geom.astype(np.float64), 0.0, 1.0)
    solid = solid_frac >= solid_threshold
    fluid = ~solid

    # --- Robustness Step A: Geometry Cleanup ---
    # Removes single-pixel noise and smooths jagged edges that create fake skeleton branches.
    # We use a small disk (radius 1). If channels are extremely thin (1-2 pixels), 
    # this might be aggressive, but for standard CFD grids, it's beneficial.
    # We apply it only if the grid is large enough to support it.
    if min(M, N) > 10:
        fluid_clean = binary_opening(fluid, footprint=disk(CLEANUP_RADIUS))
        # Restore connectivity if opening broke something critical? 
        # Usually opening is safe for noise. We use the cleaned version for calculating structure.
    else:
        fluid_clean = fluid

    # 2. Distance Transform (Distance to nearest wall)
    # Computed on the CLEANED fluid to get smooth gradients
    dist_m = distance_transform_edt(fluid_clean, sampling=(dy, dx))
    
    # 3. Skeletonize (Medial Axis)
    # The skeleton represents the centerlines of the channels.
    skel = medial_axis(fluid_clean)
    
    # --- Robustness Step B: Ridge Detection ---
    # The topological skeleton might sit 1 pixel off the exact geometric center 
    # due to discrete grid logic.
    # We sample the MAX distance in a 3x3 neighborhood around each skeleton pixel 
    # to ensure we capture the true peak width (the "Ridge").
    dist_peaks = maximum_filter(dist_m, size=3)
    
    # 4. Define Width on Skeleton
    width_on_skel = np.zeros_like(dist_m)
    # Width = 2 * Radius
    width_on_skel[skel] = 2.0 * dist_peaks[skel]
    
    # 5. Propagate Skeleton Width to the rest of the fluid
    # We find, for every pixel, the index of the NEAREST skeleton pixel.
    
    # Invert skeleton: 0 where skeleton is, 1 elsewhere
    feature_mask = ~skel 
    
    # return_indices=True gives us the [y, x] indices of the nearest feature (skeleton point)
    # indices has shape (2, M, N)
    _, indices = distance_transform_edt(feature_mask, sampling=(dy, dx), return_indices=True)
    
    # Look up the width value at those nearest skeleton coordinates
    propagated_width_m = width_on_skel[indices[0], indices[1]]
    
    # 6. Apply to Original Fluid Mask
    # We use the ORIGINAL 'fluid' mask for the final assignment, so we don't lose 
    # pixels that were trimmed by the 'binary_opening' cleanup step. 
    # Those edge pixels just inherit the width of their nearest skeleton neighbor.
    final_width_field = np.zeros_like(propagated_width_m)
    final_width_field[fluid] = propagated_width_m[fluid]
    
    return final_width_field, dist_m, skel

# =============================================================================
# Main
# =============================================================================

def main():
    if not os.path.exists(GEOM_FILE):
        raise FileNotFoundError(f"Missing {GEOM_FILE}")
    
    M, N, dy, dx = read_fluid_params(FLUID_PARAMS_FILE)
    geom = np.loadtxt(GEOM_FILE, dtype=float, delimiter="\t")

    if geom.shape != (M, N):
        raise ValueError(f"Shape mismatch: Params=({M},{N}), Geom={geom.shape}")

    print("Computing Local Channel Width (Robust Propagation Method)...")
    
    width_m, dist_m, skel = compute_local_channel_width(
        geom, dx, dy, SOLID_THRESHOLD
    )

    # Convert to mm
    cell_data = {
        "channel_width_mm": width_m * 1e3,
        "dist_to_wall_mm": dist_m * 1e3,
        "skeleton": skel.astype(np.float64),
    }

    write_vtk_structured_points(VTK_OUT, N, M, dx, dy, cell_data)

    print("-" * 60)
    print(f"Output: {VTK_OUT}")
    print("Improvements applied:")
    print("  1. Binary Opening: Removes pixel noise to prevent fake branches.")
    print("  2. Ridge Detection: Samples local max distance for better accuracy.")
    print("  3. Voronoi Propagation: Assigns centerline width to full channel.")
    print("-" * 60)

if __name__ == "__main__":
    main()