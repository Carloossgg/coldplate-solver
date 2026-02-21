"""
Run adjoint sensitivity using coldplate-solver export data.
Reads T_bottom.txt, pressure_thermal.txt, k_thermal.txt from ExportFiles,
computes sensitivity via the adjoint method, and writes sensitivity_field.txt.

For MMA route optimization, also converts dQ/dk -> dJ/dgamma when geometry_thermal.txt
is available.

Usage:
    python run_coldplate_adjoint.py [export_dir] [q_k] [k_s] [k_f]
    
Arguments:
    export_dir: Path to ExportFiles directory (default: from env or hardcoded)
    q_k:        RAMP parameter (default: 1.0, continuation: 1→3→10→30)
    k_s:        Solid thermal conductivity [W/m-K] (default: 400.0)
    k_f:        Fluid thermal conductivity [W/m-K] (default: 0.6)
"""
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from thermal_adjoint import solve_thermal_adjoint

# Default paths to coldplate ExportFiles (override with env or args)
EXPORT_DIR = os.environ.get(
    "COLDPLATE_EXPORT_DIR",
    r"c:\Thermal_and_Optimization\coldplate-solver-main\coldplate-solver-main\ExportFiles",
)
PATHS = {
    "T": os.path.join(EXPORT_DIR, "T_bottom.txt"),
    "P": os.path.join(EXPORT_DIR, "pressure_thermal.txt"),
    "k": os.path.join(EXPORT_DIR, "k_thermal.txt"),
}
# Output sensitivity field to ExportFiles directory
SENSITIVITY_OUTPUT = os.path.join(EXPORT_DIR, "sensitivity_field.txt")

# Default RAMP parameters (match thermal_solver.cpp defaults)
DEFAULT_K_S = 400.0   # Solid thermal conductivity [W/m-K] (copper)
DEFAULT_K_F = 0.6     # Fluid thermal conductivity [W/m-K] (water)
DEFAULT_Q_K = 1.0     # RAMP parameter (continuation: 1→3→10→30)


def load_export(path, delimiter="\t"):
    """Load a 2D matrix from tab-delimited export file."""
    return np.loadtxt(path, delimiter=delimiter, dtype=np.float64)


def compute_dk_dgamma(gamma, k_s, k_f, q_k):
    """
    Compute RAMP derivative dk/dgamma for thermal conductivity interpolation.
    
    RAMP formula: k(gamma) = k_f + (k_s - k_f) * (1 - gamma) / (1 + q_k * gamma)
    Derivative:  dk/dgamma = -(k_s - k_f) * (1 + q_k) / (1 + q_k*gamma)^2
    
    Args:
        gamma: Density field (1 = fluid, 0 = solid)
        k_s:   Solid thermal conductivity [W/m-K]
        k_f:   Fluid thermal conductivity [W/m-K]
        q_k:   RAMP convexity parameter
    
    Returns:
        dk_dgamma: Derivative field (same shape as gamma)
    """
    gamma_clipped = np.clip(gamma, 0.0, 1.0)
    den = 1.0 + q_k * gamma_clipped
    dk_dgamma = -(k_s - k_f) * (1.0 + q_k) / (den * den)
    return dk_dgamma


def main():
    # Parse command-line arguments
    export_dir = PATHS["T"].rsplit(os.sep, 1)[0] if len(sys.argv) < 2 else sys.argv[1]
    q_k = float(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_Q_K
    k_s = float(sys.argv[3]) if len(sys.argv) > 3 else DEFAULT_K_S
    k_f = float(sys.argv[4]) if len(sys.argv) > 4 else DEFAULT_K_F
    
    paths = {
        "T": os.path.join(export_dir, "T_bottom.txt"),
        "P": os.path.join(export_dir, "pressure_thermal.txt"),
        "k": os.path.join(export_dir, "k_thermal.txt"),
    }
    sensitivity_output = os.path.join(export_dir, "sensitivity_field.txt")
    
    print("=== Adjoint Sensitivity Analysis for Cold Plate Route Optimization ===")
    print(f"Export directory: {export_dir}")
    print(f"RAMP parameters: k_s={k_s:.1f} W/m-K, k_f={k_f:.3f} W/m-K, q_k={q_k:.1f}")
    print()

    print("Loading coldplate export data...")
    for key, p in paths.items():
        if not os.path.isfile(p):
            print(f"Error: file not found: {p}")
            sys.exit(1)
        print(f"  {key}: {p}")

    T_bottom = load_export(paths["T"])
    pressure_thermal = load_export(paths["P"])
    k_thermal = load_export(paths["k"])

    ny, nx = T_bottom.shape
    if pressure_thermal.shape != (ny, nx) or k_thermal.shape != (ny, nx):
        print(
            f"Error: shape mismatch T {T_bottom.shape}, P {pressure_thermal.shape}, k {k_thermal.shape}"
        )
        sys.exit(1)
    print(f"Grid shape: ny={ny}, nx={nx}")

    print("\nRunning adjoint method...")
    sens = solve_thermal_adjoint(
        T_in=T_bottom,
        k_in=k_thermal,
        P_in=pressure_thermal,
        sensitivity_output_path=sensitivity_output,
    )

    print(f"\nDone. Sensitivity field (dQ/dk) written to: {sensitivity_output}")
    print(f"  Shape: {sens.shape}, min={sens.min():.6e}, max={sens.max():.6e}")
    
    # Convert dQ/dk -> dJ/dgamma for MMA route optimization
    geom_path = os.path.join(export_dir, "geometry_thermal.txt")
    if os.path.isfile(geom_path):
        try:
            print(f"\nConverting dQ/dk -> dJ/dgamma for MMA optimization...")
            geo = np.loadtxt(geom_path, delimiter="\t", dtype=np.float64)
            
            if geo.shape != (ny, nx):
                print(f"  ERROR: geometry_thermal shape {geo.shape} != grid ({ny},{nx})")
                print(f"  Skipping dJ/dgamma conversion.")
            else:
                # Solver convention: file 0 = fluid, 1 = solid => gamma = 1 - cellType
                # geometry_thermal.txt: 0 = fluid, 1 = solid
                # gamma convention: 1 = fluid, 0 = solid
                gamma = 1.0 - np.clip(geo, 0.0, 1.0)
                
                # Compute dk/dgamma using RAMP derivative
                dk_dgamma = compute_dk_dgamma(gamma, k_s, k_f, q_k)
                
                # Chain rule: dJ/dgamma = (dQ/dk) * (dk/dgamma)
                dJ_dgamma = sens * dk_dgamma
                
                # Validate results
                fluid_mask = gamma > 0.99
                solid_mask = gamma < 0.01
                buffer_mask = ~(fluid_mask | solid_mask)
                
                print(f"  Geometry: {np.sum(fluid_mask)} fluid, {np.sum(solid_mask)} solid, {np.sum(buffer_mask)} buffer cells")
                print(f"  dk/dgamma range: [{dk_dgamma.min():.6e}, {dk_dgamma.max():.6e}] W/m-K")
                print(f"  dJ/dgamma range: [{dJ_dgamma.min():.6e}, {dJ_dgamma.max():.6e}]")
                
                # Check sign convention: negative dJ/dgamma means increasing gamma (more fluid) decreases objective (good)
                if np.any(dJ_dgamma < 0):
                    print(f"  Note: Negative dJ/dgamma values indicate regions where increasing fluid (gamma up) improves cooling")
                if np.any(dJ_dgamma > 0):
                    print(f"  Note: Positive dJ/dgamma values indicate regions where increasing fluid (gamma up) worsens cooling")
                
                # Write dJ_dgamma.txt for MMA optimizer
                dJ_path = os.path.join(export_dir, "dJ_dgamma.txt")
                np.savetxt(dJ_path, dJ_dgamma, delimiter="\t", fmt="%.10e")
                print(f"\n  [OK] dJ/dgamma (for MMA) written to: {dJ_path}")
                print(f"    Format: Ny x Nx tab-separated, ready for MMA optimizer")
                
        except Exception as e:
            print(f"\n  ERROR converting dQ/dk -> dJ/dgamma: {e}")
            print(f"  Only dQ/dk written. See ADJOINT_README.md for manual conversion.")
            import traceback
            traceback.print_exc()
    else:
        print(f"\n  No geometry_thermal.txt found at: {geom_path}")
        print(f"  Only dQ/dk written. To get dJ/dgamma for MMA:")
        print(f"    1. Ensure geometry_thermal.txt exists in ExportFiles")
        print(f"    2. Re-run this script (it will auto-convert)")
        print(f"  See ADJOINT_README.md for details.")

    print("\n=== Adjoint analysis complete ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
