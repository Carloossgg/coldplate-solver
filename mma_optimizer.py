"""
mma_optimizer.py — MMA Topology Optimization for Cold Plate Channel Layout

Density-based topology optimization using MMA (Method of Moving Asymptotes,
Svanberg 1987/2002) to find optimal cold plate channel geometry.

Design variables:  gamma field (0=solid, 1=fluid) — continuous relaxation
Objective:         Weighted combination of peak base temperature + non-uniformity
Constraints:       Volume fraction (fraction of domain that is fluid)
Gradients:         Thermal adjoint via run_coldplate_adjoint.py (dJ/dγ)

Architecture:
    mma_optimizer.py         (this file — main driver)
    ├── run_solvers.py        (runs simple.exe + thermal_solver.exe per iteration)
    └── run_coldplate_adjoint.py (thermal adjoint for dJ/dγ)

Usage:
    python mma_optimizer.py
    python mma_optimizer.py --max-iter 50 --vol-frac 0.4
    python mma_optimizer.py --resume  (continue from last saved state)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time

import numpy as np
from pathlib import Path

from run_solvers import run_iteration, EXPORT_DIR, PROJECT_ROOT, parse_thermal_metrics


# =========================================================================
# CHECKPOINT SAVE / LOAD
# =========================================================================
def save_checkpoint(
    path: str | Path,
    gamma: np.ndarray,
    history: dict,
    mma_state: "MMAState",
    config: "OptConfig",
    current_iter: int,
) -> None:
    """
    Save a full optimization checkpoint to a .npz file.

    The file contains:
        - gamma array (heatsink design variables)
        - history dict (serialised as JSON string)
        - MMA state arrays (xold1, xold2, low, upp, iter_count)
        - config values (all numeric OptConfig fields)
        - current_iter (the last completed iteration index)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Pack config fields into a dict
    cfg_dict = {
        k: v for k, v in vars(config).items()
        if isinstance(v, (int, float, list)) and k not in ("inlet_geometry", "outlet_geometry")
    }

    arrays = {
        "gamma": gamma,
        "current_iter": np.array(current_iter, dtype=int),
        "config_json": np.array(json.dumps(cfg_dict)),
        "history_json": np.array(json.dumps(history)),
        "mma_iter": np.array(mma_state.iter, dtype=int),
    }
    if mma_state.xold1 is not None:
        arrays["mma_xold1"] = mma_state.xold1
    if mma_state.xold2 is not None:
        arrays["mma_xold2"] = mma_state.xold2
    if mma_state.low is not None:
        arrays["mma_low"] = mma_state.low
    if mma_state.upp is not None:
        arrays["mma_upp"] = mma_state.upp
    if config.inlet_geometry is not None:
        arrays["inlet_geometry"] = config.inlet_geometry
    if config.outlet_geometry is not None:
        arrays["outlet_geometry"] = config.outlet_geometry

    np.savez_compressed(str(path), **arrays)
    print(f"  [Checkpoint] Saved to {path}")


def load_checkpoint(path: str | Path) -> tuple[np.ndarray, dict, "MMAState", "OptConfig", int]:
    """
    Load a checkpoint and return (gamma, history, mma_state, config, start_iter).

    start_iter = current_iter + 1 (the next iteration to run).
    """
    data = np.load(str(path), allow_pickle=False)

    gamma = data["gamma"]
    current_iter = int(data["current_iter"])
    history = json.loads(str(data["history_json"]))
    cfg_dict = json.loads(str(data["config_json"]))

    config = OptConfig()
    for k, v in cfg_dict.items():
        setattr(config, k, v)

    mma_iter = int(data["mma_iter"])
    mma_state = MMAState(gamma.size)
    mma_state.iter = mma_iter
    if "mma_xold1" in data:
        mma_state.xold1 = data["mma_xold1"]
    if "mma_xold2" in data:
        mma_state.xold2 = data["mma_xold2"]
    if "mma_low" in data:
        mma_state.low = data["mma_low"]
    if "mma_upp" in data:
        mma_state.upp = data["mma_upp"]
    if "inlet_geometry" in data:
        config.inlet_geometry = data["inlet_geometry"]
    if "outlet_geometry" in data:
        config.outlet_geometry = data["outlet_geometry"]

    start_iter = current_iter + 1
    print(f"  [Checkpoint] Loaded from {path}")
    print(f"  Resuming from iteration {start_iter}")
    return gamma, history, mma_state, config, start_iter


# =========================================================================
# CONFIGURATION
# =========================================================================
class OptConfig:
    """All tunable optimization parameters."""

    def __init__(self):
        # --- Optimization loop ---
        self.max_iter = 100

        # --- Volume fraction constraint ---
        # Fraction of design domain that should be fluid (gamma=1)
        self.vol_frac_target = 0.4

        # --- Objective weights ---
        # J = w_peak * T_max_base  +  w_uniformity * (1 - uniformity_base)
        self.w_peak = 1.0
        self.w_uniformity = 0.0  # Set > 0 to penalize non-uniformity

        # --- MMA parameters ---
        self.move_limit = 0.2        # Max change per iteration (absolute)
        self.asyinit = 0.5           # Initial asymptote distance (× bound range)
        self.asydecr = 0.7           # Shrink factor for oscillating variables
        self.asyincr = 1.2           # Growth factor for monotone variables

        # --- RAMP continuation schedule (thermal solver qk) ---
        # Gradually increase qk to push intermediate densities toward 0/1
        self.qk_schedule   = [1.0, 3.0, 10.0, 30.0]
        self.qk_switch_iter = [0,   25,  50,   75]

        # --- Density filter ---
        # Smooths the design to avoid checkerboard patterns
        # Radius in number of cells (set 0 to disable)
        self.filter_radius = 3.0

        # --- Convergence ---
        self.change_tol = 0.01  # Stop if max|x_new - x_old| < tol

        # --- Boundary handling ---
        # Physical thickness of the solid wall at top/bottom (metres).
        # n_wall_rows is derived at runtime from wall_thickness / dy.
        self.wall_thickness = 0.002   # 2 mm
        self.n_wall_rows = 1          # placeholder — overwritten in run_optimization

        # --- Grid info (populated at runtime) ---
        self.M = 0              # Number of rows
        self.N_fluid = 0        # Total columns in fluid domain
        self.N_heatsink = 0     # Columns in heatsink (thermal domain)
        self.N_in_buffer = 0    # Inlet buffer columns
        self.N_out_buffer = 0   # Outlet buffer columns
        self.dx = 0.0
        self.dy = 0.0
        self.Ht = 0.0
        
        # --- Static Geometry Buffers ---
        # Stores the initial inlet/outlet geometry (fluid=0, solid=1)
        # to preserve them during optimization loop.
        self.inlet_geometry = None  # (M, N_in_buffer)
        self.outlet_geometry = None # (M, N_out_buffer)


# =========================================================================
# GRID PARAMETER LOADING
# =========================================================================
def load_grid_params(config: OptConfig):
    """Read grid dimensions from ExportFiles/fluid_params.txt."""
    fp_path = EXPORT_DIR / "fluid_params.txt"
    if not fp_path.exists():
        raise FileNotFoundError(
            f"fluid_params.txt not found at {fp_path}.\n"
            "Run a GeometryGenerator first to create initial geometry."
        )

    with open(fp_path, "r") as f:
        vals = f.readline().strip().split()

    config.M = int(vals[0])
    config.N_fluid = int(vals[1])
    config.dy = float(vals[2])
    config.dx = float(vals[3])
    config.N_in_buffer = int(vals[5])
    config.N_out_buffer = int(vals[6])
    config.Ht = float(vals[7])
    config.N_heatsink = config.N_fluid - config.N_in_buffer - config.N_out_buffer

    print(f"Grid: {config.M} x {config.N_fluid}  "
          f"(heatsink: {config.N_heatsink} cols, "
          f"buffers: in={config.N_in_buffer} out={config.N_out_buffer})")
    print(f"Cell: {config.dx*1e6:.0f} x {config.dy*1e6:.0f} µm  "
          f"Ht={config.Ht*1000:.1f} mm")

    # --- Load Initial Geometry (to preserve inlet/outlet) ---
    geom_path = EXPORT_DIR / "geometry_fluid.txt"
    if geom_path.exists():
        # Load full geometry: 0=fluid, 1=solid
        full_geom = np.loadtxt(str(geom_path))
        
        if full_geom.shape != (config.M, config.N_fluid):
            print(f"WARNING: geometry_fluid.txt shape {full_geom.shape} mismatch with params {(config.M, config.N_fluid)}")
        else:
            # Extract and store static regions
            config.inlet_geometry = full_geom[:, :config.N_in_buffer].copy()
            config.outlet_geometry = full_geom[:, config.N_in_buffer + config.N_heatsink:].copy()

            # Ensure buffer interiors are fluid (0).
            # A uniform-value geometry (e.g. 0.8) would otherwise block the
            # inlet/outlet, producing zero flow and instant solver convergence.
            # Keep the top/bottom wall rows solid.
            n_wall = max(1, int(np.ceil(config.wall_thickness / config.dy))) if hasattr(config, 'wall_thickness') else 1
            config.inlet_geometry[n_wall:-n_wall, :] = 0.0
            config.outlet_geometry[n_wall:-n_wall, :] = 0.0
            # Wall rows stay solid
            config.inlet_geometry[:n_wall, :] = 1.0
            config.inlet_geometry[-n_wall:, :] = 1.0
            config.outlet_geometry[:n_wall, :] = 1.0
            config.outlet_geometry[-n_wall:, :] = 1.0

            print("  Loaded and preserved initial inlet/outlet geometry (buffers forced fluid).")
    else:
        print("WARNING: geometry_fluid.txt not found. Inlet/Outlet will be empty.")


# =========================================================================
# GEOMETRY I/O
# =========================================================================
def write_geometry(gamma_design: np.ndarray, config: OptConfig):
    """
    Write the design gamma field to ExportFiles geometry files.

    Parameters
    ----------
    gamma_design : (M, N_heatsink) array, values in [0, 1]
        gamma=1 → fluid, gamma=0 → solid.
    config : OptConfig with grid dimensions.

    The file format stores cellType = 1 - gamma:
        cellType=0 → fluid, cellType=1 → solid
    """
    M = config.M
    N_in = config.N_in_buffer
    N_out = config.N_out_buffer
    N_hs = config.N_heatsink
    N_total = config.N_fluid

    # --- Build full-domain cellType (0=fluid, 1=solid) ---
    celltype_full = np.zeros((M, N_total), dtype=float)

    # Inlet buffer: Restore static geometry if available
    if config.inlet_geometry is not None:
        celltype_full[:, :N_in] = config.inlet_geometry
    else:
        # Fallback: all fluid (0)
        celltype_full[:, :N_in] = 0.0

    # Outlet buffer: Restore static geometry if available
    if config.outlet_geometry is not None:
        celltype_full[:, N_in + N_hs:] = config.outlet_geometry
    else:
        # Fallback: all fluid (0)
        celltype_full[:, N_in + N_hs:] = 0.0

    # Heatsink region: cellType = 1 - gamma
    celltype_full[:, N_in : N_in + N_hs] = 1.0 - gamma_design

    # --- Enforce solid walls at top/bottom (Safety Override) ---
    celltype_full[0, :] = 1.0
    celltype_full[-1, :] = 1.0

    # --- Write fluid geometry (full domain) ---
    np.savetxt(
        str(EXPORT_DIR / "geometry_fluid.txt"),
        celltype_full, fmt="%.6f", delimiter="\t",
    )

    # --- Write thermal geometry (heatsink crop only) ---
    celltype_thermal = celltype_full[:, N_in : N_in + N_hs]
    np.savetxt(
        str(EXPORT_DIR / "geometry_thermal.txt"),
        celltype_thermal, fmt="%.6f", delimiter="\t",
    )


def read_current_gamma(config: OptConfig) -> np.ndarray:
    """
    Read the current geometry and convert to gamma (design variable space).

    Returns (M, N_heatsink) array with gamma=1 (fluid), gamma=0 (solid).
    """
    geom_path = EXPORT_DIR / "geometry_thermal.txt"
    if not geom_path.exists():
        raise FileNotFoundError(f"No geometry at {geom_path}")

    celltype = np.loadtxt(str(geom_path))
    gamma = 1.0 - celltype  # Invert: cellType 0→gamma 1, cellType 1→gamma 0
    return gamma.clip(0.0, 1.0)


# =========================================================================
# DENSITY FILTER (Cone filter / Bruns & Tortorelli 2001)
# =========================================================================
def density_filter(x: np.ndarray, radius: float, dx: float, dy: float) -> np.ndarray:
    """
    Apply a cone-shaped density filter for regularization.

    Prevents checkerboard patterns and mesh-dependent solutions.
    Each cell's filtered value is a weighted average of its neighborhood,
    with weights linearly decreasing with distance.
    """
    if radius <= 0:
        return x.copy()

    M, N = x.shape
    x_filt = np.zeros_like(x)

    # Radius in cell units
    ri = int(np.ceil(radius))

    for i in range(M):
        i_lo = max(0, i - ri)
        i_hi = min(M, i + ri + 1)
        for j in range(N):
            j_lo = max(0, j - ri)
            j_hi = min(N, j + ri + 1)

            # Compute weights (linear cone)
            w_sum = 0.0
            wx_sum = 0.0
            for ii in range(i_lo, i_hi):
                for jj in range(j_lo, j_hi):
                    dist = np.sqrt(((ii - i) * dy) ** 2 + ((jj - j) * dx) ** 2)
                    w = max(0.0, radius * max(dx, dy) - dist)
                    w_sum += w
                    wx_sum += w * x[ii, jj]

            x_filt[i, j] = wx_sum / max(w_sum, 1e-30)

    return x_filt


def density_filter_fast(x: np.ndarray, radius: float) -> np.ndarray:
    """
    Fast approximate density filter using scipy convolution.

    Falls back to the exact version if scipy is unavailable.
    """
    if radius <= 0:
        return x.copy()

    try:
        from scipy.ndimage import uniform_filter
        # Approximate cone filter with uniform (box) filter
        size = int(2 * np.ceil(radius) + 1)
        return uniform_filter(x, size=size, mode="reflect")
    except ImportError:
        # Fallback to exact (slow) implementation
        return density_filter(x, radius, 1.0, 1.0)


# =========================================================================
# GEOMETRY PREVIEW
# =========================================================================
def save_geometry_preview(
    gamma: np.ndarray,
    output_path: str,
    iteration: int,
    f0: float | None = None,
    vol_frac: float | None = None,
) -> None:
    """
    Save a PNG visualisation of the design gamma field.

    White = fluid (gamma=1), black = solid (gamma=0).
    Intermediate grey values represent the continuous density field.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend — safe in any environment
        import matplotlib.pyplot as plt
    except ImportError:
        print("  WARNING: matplotlib not available — skipping PNG preview.")
        return

    fig, ax = plt.subplots(figsize=(12, 4), dpi=120)

    # gamma=1 is fluid (white), gamma=0 is solid (black)
    im = ax.imshow(
        gamma,
        cmap="gray",
        vmin=0.0, vmax=1.0,
        aspect="auto",
        origin="upper",
    )
    plt.colorbar(im, ax=ax, label="gamma  (0=solid, 1=fluid)", fraction=0.02, pad=0.02)

    title = f"MMA Iteration {iteration}"
    if f0 is not None:
        title += f"  |  f0 = {f0:.4f}"
    if vol_frac is not None:
        title += f"  |  vol = {vol_frac:.3f}"
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("Column (x)")
    ax.set_ylabel("Row (y)")

    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


# =========================================================================
# OBJECTIVE EVALUATION
# =========================================================================
def evaluate_objective(
    gamma: np.ndarray,
    config: OptConfig,
    iteration: int,
    qk: float,
) -> tuple[float, dict]:
    """
    Write geometry, run solvers, compute objective.

    Returns
    -------
    f0 : float
        Objective value.
    metrics : dict
        Full thermal metrics from thermal_metrics.txt.
    """
    # 1. Write geometry to files (continuous gamma, no rounding).
    write_geometry(gamma, config)

    # 2. Run both solvers
    result = run_iteration(
        iteration=iteration,
        skip_fluid=False,
        qk=qk,
        k_interp_mode=1,  # RAMP
        verbose=True,
    )

    if not result["success"]:
        print("WARNING: Solver failed. Returning large objective.")
        return 1e10, {}

    metrics = result["metrics"]

    # 3. Compute weighted objective
    T_peak = metrics.get("T_avg_base", 100.0)  # fallback
    # Try to get max base temperature
    # thermal_metrics.txt has T_avg_base and T_range_base
    # Peak ≈ T_avg + 0.5 * T_range (rough estimate from stats)
    T_range = metrics.get("T_range_base", 0.0)
    T_peak_est = T_peak + 0.5 * T_range

    uniformity = metrics.get("T_uniformity_base", 1.0)

    f0 = (config.w_peak * T_peak_est
          + config.w_uniformity * (1.0 - uniformity))

    print(f"  Objective: f0 = {f0:.4f}  "
          f"(T_peak≈{T_peak_est:.2f}°C, uniformity={uniformity:.4f})")

    return f0, metrics


# =========================================================================
# SENSITIVITY
# =========================================================================
def compute_sensitivities(
    gamma: np.ndarray,
    config: OptConfig,
    metrics: dict,
    qk: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute objective and constraint sensitivities (gradients).

    Calls run_coldplate_adjoint.main(), which reads the solver outputs from
    ExportFiles, runs the adjoint method, and writes dJ_dgamma.txt there.
    We then read that file back as df0dx.

    Returns
    -------
    df0dx : (M, N_heatsink) array — dJ/dgamma for objective
    dgdx  : (M, N_heatsink) array — dg/dgamma for volume fraction constraint
    """
    # Ensure AdjointMethod/ is on the path so run_coldplate_adjoint can be found
    _adjoint_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AdjointMethod")
    if _adjoint_dir not in sys.path:
        sys.path.insert(0, _adjoint_dir)
    import run_coldplate_adjoint

    M, N = gamma.shape

    # ---------------------------------------------------------------
    # OBJECTIVE GRADIENT — via run_coldplate_adjoint
    # It reads T_bottom.txt, pressure_thermal.txt, k_thermal.txt from
    # ExportFiles, computes the adjoint, and writes:
    #   sensitivity_field.txt  (dQ/dk)
    #   dJ_dgamma.txt          (dJ/dgamma, ready for MMA)
    # ---------------------------------------------------------------
    print("  Running adjoint sensitivity (run_coldplate_adjoint)...")
    try:
        # Pass qk so RAMP derivative uses the same parameter as the solver
        import sys as _sys
        _argv_backup = _sys.argv[:]
        _sys.argv = [run_coldplate_adjoint.__file__, str(EXPORT_DIR), str(qk)]
        run_coldplate_adjoint.main()
        _sys.argv = _argv_backup
    except SystemExit:
        pass  # main() calls sys.exit(0) on success — that's fine

    # Read the dJ/dgamma output written by the adjoint script
    dJ_path = EXPORT_DIR / "dJ_dgamma.txt"
    if not dJ_path.exists():
        print("  WARNING: dJ_dgamma.txt not found after adjoint run. Returning zero gradient.")
        return np.zeros((M, N)), np.ones((M, N)) / (M * N)

    df0dx_full = np.loadtxt(str(dJ_path), delimiter="\t", dtype=np.float64)

    # dJ_dgamma.txt covers the thermal (heatsink) domain only — same shape as gamma
    if df0dx_full.shape != (M, N):
        print(f"  WARNING: dJ_dgamma shape {df0dx_full.shape} != gamma shape {(M, N)}. "
              "Returning zero gradient.")
        return np.zeros((M, N)), np.ones((M, N)) / (M * N)

    df0dx = df0dx_full
    print(f"  Adjoint df0dx range: [{df0dx.min():.6e}, {df0dx.max():.6e}]")
    # ---------------------------------------------------------------

    # ---------------------------------------------------------------
    # CONSTRAINT GRADIENT — Volume fraction (exact, always correct)
    # ---------------------------------------------------------------
    # g = mean(gamma) - vol_frac_target <= 0
    # dg/dgamma_i = 1 / (M * N)
    dgdx = np.ones((M, N)) / (M * N)

    return df0dx, dgdx


# =========================================================================
# MMA UPDATE (Svanberg 1987)
# =========================================================================
class MMAState:
    """Stores MMA iteration history (asymptotes, previous designs)."""

    def __init__(self, n: int):
        self.n = n
        self.xold1 = None  # x from iteration k-1
        self.xold2 = None  # x from iteration k-2
        self.low = None    # Lower asymptotes
        self.upp = None    # Upper asymptotes
        self.iter = 0


def mma_update(
    x: np.ndarray,
    f0: float,
    df0dx: np.ndarray,
    g: float,
    dgdx: np.ndarray,
    xmin: np.ndarray,
    xmax: np.ndarray,
    state: MMAState,
    move_limit: float = 0.2,
    asyinit: float = 0.5,
    asydecr: float = 0.7,
    asyincr: float = 1.2,
) -> np.ndarray:
    """
    One iteration of the MMA algorithm (Method of Moving Asymptotes).

    Solves a single-constraint MMA subproblem via dual bisection.

    Parameters
    ----------
    x       : (n,) current design variables
    f0      : scalar objective value
    df0dx   : (n,) objective gradient
    g       : scalar constraint value (g <= 0 is feasible)
    dgdx    : (n,) constraint gradient
    xmin    : (n,) lower bounds
    xmax    : (n,) upper bounds
    state   : MMAState carrying history
    move_limit, asyinit, asydecr, asyincr : MMA parameters

    Returns
    -------
    xnew    : (n,) updated design variables
    """
    n = x.size
    xval = x.ravel()
    df0 = df0dx.ravel()
    dg = dgdx.ravel()

    xmin_f = xmin.ravel()
    xmax_f = xmax.ravel()

    delta = xmax_f - xmin_f
    delta = np.maximum(delta, 1e-10)

    # --- 1. Update asymptotes ---
    if state.iter < 2 or state.low is None:
        # First two iterations: use initial asymptote distance
        low = xval - asyinit * delta
        upp = xval + asyinit * delta
    else:
        xold1 = state.xold1
        xold2 = state.xold2
        low = state.low.copy()
        upp = state.upp.copy()

        # Check for oscillation: sign change in consecutive moves
        s = (xval - xold1) * (xold1 - xold2)

        # Oscillating variables: shrink asymptotes
        osc = s < 0
        low[osc] = xval[osc] - asydecr * (xold1[osc] - low[osc])
        upp[osc] = xval[osc] + asydecr * (upp[osc] - xold1[osc])

        # Monotone variables: expand asymptotes
        mon = s > 0
        low[mon] = xval[mon] - asyincr * (xold1[mon] - low[mon])
        upp[mon] = xval[mon] + asyincr * (upp[mon] - xold1[mon])

        # Stationary variables: keep initial distance
        stat = s == 0
        low[stat] = xval[stat] - asyinit * delta[stat]
        upp[stat] = xval[stat] + asyinit * delta[stat]

    # Clamp asymptotes to reasonable range
    low = np.maximum(low, xval - 10.0 * delta)
    upp = np.minimum(upp, xval + 10.0 * delta)
    low = np.minimum(low, xval - 0.01 * delta)
    upp = np.maximum(upp, xval + 0.01 * delta)

    # --- 2. Move limits ---
    alpha = np.maximum(xmin_f, np.maximum(low + 0.1 * (xval - low), xval - move_limit * delta))
    beta  = np.minimum(xmax_f, np.minimum(upp - 0.1 * (upp - xval), xval + move_limit * delta))

    # --- 3. Compute MMA approximation coefficients ---
    ux = upp - xval
    xl = xval - low
    ux2 = ux * ux
    xl2 = xl * xl

    # Objective
    p0 = np.maximum(df0, 0.0) * ux2 + 1e-6 * delta
    q0 = np.maximum(-df0, 0.0) * xl2 + 1e-6 * delta

    # Constraint
    p1 = np.maximum(dg, 0.0) * ux2
    q1 = np.maximum(-dg, 0.0) * xl2

    # --- 4. Solve dual problem via bisection ---
    # For one constraint, the dual is 1D: find lam >= 0 that satisfies
    # the constraint g(x(lam)) = 0 (or lam = 0 if constraint is inactive)

    def get_x_from_lam(lam):
        """Compute x* given Lagrange multiplier lam."""
        pp = p0 + lam * p1
        qq = q0 + lam * q1
        # Optimal x_j = (sqrt(pp_j) * L_j + sqrt(qq_j) * U_j) / (sqrt(pp_j) + sqrt(qq_j))
        sp = np.sqrt(pp)
        sq = np.sqrt(qq)
        denom = sp + sq
        denom = np.maximum(denom, 1e-20)
        xnew = (sp * low + sq * upp) / denom
        xnew = np.clip(xnew, alpha, beta)
        return xnew

    def constraint_residual(lam):
        """Compute g(x*(lam))."""
        xnew = get_x_from_lam(lam)
        return np.dot(dg, xnew) + (g - np.dot(dg, xval))

    # Bisection on lam
    lam_lo = 0.0
    lam_hi = 1e6

    # Check if constraint is active
    if constraint_residual(0.0) <= 0.0:
        # Constraint already satisfied at lam=0
        lam_star = 0.0
    else:
        # Find upper bound for lam
        while constraint_residual(lam_hi) > 0 and lam_hi < 1e12:
            lam_hi *= 10.0

        # Bisection
        for _ in range(100):
            lam_mid = 0.5 * (lam_lo + lam_hi)
            if constraint_residual(lam_mid) > 0:
                lam_lo = lam_mid
            else:
                lam_hi = lam_mid
            if lam_hi - lam_lo < 1e-10:
                break
        lam_star = 0.5 * (lam_lo + lam_hi)

    xnew = get_x_from_lam(lam_star)

    # --- 5. Update state ---
    state.xold2 = state.xold1.copy() if state.xold1 is not None else xval.copy()
    state.xold1 = xval.copy()
    state.low = low
    state.upp = upp
    state.iter += 1

    return xnew


# =========================================================================
# MAIN OPTIMIZATION LOOP
# =========================================================================
def run_optimization(
    config: OptConfig,
    start_iter: int = 0,
    gamma_init: np.ndarray | None = None,
    history_init: dict | None = None,
    mma_state_init: "MMAState | None" = None,
    stop_event: threading.Event | None = None,
    on_progress=None,
    checkpoint_dir: Path | None = None,
):
    """Run the full MMA topology optimization."""

    # --- Load grid parameters ---
    load_grid_params(config)

    # --- Derive wall rows from physical thickness ---
    config.n_wall_rows = max(1, int(np.ceil(config.wall_thickness / config.dy)))
    print(f"Wall thickness: {config.wall_thickness*1000:.1f} mm  →  {config.n_wall_rows} rows")
    M = config.M
    N = config.N_heatsink

    print(f"\nDesign domain: {M} x {N} = {M*N} variables")
    print(f"Volume fraction target: {config.vol_frac_target}")
    print(f"Objective weights: w_peak={config.w_peak}, w_unif={config.w_uniformity}")
    print(f"MMA move limit: {config.move_limit}")
    print(f"Filter radius: {config.filter_radius} cells")

    # --- Initialize design variables ---
    if gamma_init is not None:
        gamma = gamma_init.copy()
        print(f"  Using provided gamma (checkpoint resume): {gamma.shape}")
        print(f"  Current vol. fraction: {gamma.mean():.4f}")
    else:
        # Start from the EXISTING geometry (generated by GeometryGenerator)
        # This ensures the initial design has real channels with actual flow,
        # rather than a uniform gray field that Brinkman penalization kills.
        try:
            gamma = read_current_gamma(config)
            print(f"  Loaded existing geometry: {gamma.shape}")
            print(f"  Current vol. fraction: {gamma.mean():.4f}")
        except FileNotFoundError:
            print("  WARNING: No existing geometry found, using uniform initialization.")
            print("  Run a GeometryGenerator first for better results!")
            gamma = np.full((M, N), config.vol_frac_target)

    # Enforce wall boundaries (top/bottom rows = solid)
    gamma[:config.n_wall_rows, :] = 0.0
    gamma[-config.n_wall_rows:, :] = 0.0

    # Variable bounds
    xmin = np.zeros((M, N))
    xmax = np.ones((M, N))

    # Fix wall rows (passive elements)
    xmax[:config.n_wall_rows, :] = 0.0
    xmax[-config.n_wall_rows:, :] = 0.0

    # --- MMA state ---
    mma_state = mma_state_init if mma_state_init is not None else MMAState(M * N)

    # --- History tracking ---
    history = history_init if history_init is not None else {
        "iterations": [],
        "f0": [],
        "vol_frac": [],
        "T_peak": [],
        "T_uniformity": [],
        "max_change": [],
    }

    # --- Output directory (base) ---
    # Per-iteration files go into ExportFiles/iter{N} (same as run_solvers archives).
    # Top-level EXPORT_DIR is also used for final outputs.
    opt_output_dir = checkpoint_dir if checkpoint_dir is not None else EXPORT_DIR
    opt_output_dir.mkdir(exist_ok=True)

    print(f"\n{'='*70}")
    if start_iter > 0:
        print(f"  RESUMING MMA TOPOLOGY OPTIMIZATION FROM ITERATION {start_iter}")
    else:
        print(f"  STARTING MMA TOPOLOGY OPTIMIZATION")
    print(f"{'='*70}\n")

    for it in range(start_iter, config.max_iter):
        t0 = time.time()

        # --- Determine qk from continuation schedule ---
        qk = config.qk_schedule[0]
        for q, switch_it in zip(config.qk_schedule, config.qk_switch_iter):
            if it >= switch_it:
                qk = q

        print(f"\n{'-'*70}")
        print(f"  MMA Iteration {it}  |  qk = {qk}")
        print(f"{'-'*70}")

        # --- Apply density filter ---
        gamma_phys = density_filter_fast(gamma, config.filter_radius)
        gamma_phys = gamma_phys.clip(0.0, 1.0)

        # Enforce walls on physical density too
        gamma_phys[:config.n_wall_rows, :] = 0.0
        gamma_phys[-config.n_wall_rows:, :] = 0.0

        # --- Evaluate objective ---
        f0, metrics = evaluate_objective(gamma_phys, config, it, qk)

        # --- Volume fraction constraint ---
        # g = mean(gamma_phys) - vol_target <= 0
        vol_frac = gamma_phys.mean()
        g_vol = vol_frac - config.vol_frac_target

        # --- Compute sensitivities ---
        df0dx, dgdx = compute_sensitivities(gamma_phys, config, metrics, qk=qk)

        # --- MMA update ---
        gamma_old = gamma.copy()
        gamma_new_flat = mma_update(
            x=gamma.ravel(),
            f0=f0,
            df0dx=df0dx.ravel(),
            g=g_vol,
            dgdx=dgdx.ravel(),
            xmin=xmin.ravel(),
            xmax=xmax.ravel(),
            state=mma_state,
            move_limit=config.move_limit,
            asyinit=config.asyinit,
            asydecr=config.asydecr,
            asyincr=config.asyincr,
        )
        gamma = gamma_new_flat.reshape(M, N)

        # Enforce walls
        gamma[:config.n_wall_rows, :] = 0.0
        gamma[-config.n_wall_rows:, :] = 0.0

        # --- Convergence check ---
        max_change = np.max(np.abs(gamma - gamma_old))
        elapsed = time.time() - t0

        # --- Track history ---
        T_peak_val = metrics.get("T_avg_base", 0) + 0.5 * metrics.get("T_range_base", 0)
        T_unif_val = metrics.get("T_uniformity_base", 0)

        history["iterations"].append(it)
        history["f0"].append(float(f0))
        history["vol_frac"].append(float(vol_frac))
        history["T_peak"].append(float(T_peak_val))
        history["T_uniformity"].append(float(T_unif_val))
        history["max_change"].append(float(max_change))

        print(f"\n  Summary iter {it}:")
        print(f"    f0 = {f0:.4f}")
        print(f"    Vol frac = {vol_frac:.4f}  (target: {config.vol_frac_target})")
        print(f"    g_vol = {g_vol:.4f}  ({'feasible' if g_vol <= 0 else 'INFEASIBLE'})")
        print(f"    Max change = {max_change:.6f}")
        print(f"    Elapsed = {elapsed:.1f}s")

        # --- Progress callback (for GUI) ---
        if on_progress is not None:
            try:
                on_progress(
                    iteration=it,
                    f0=float(f0),
                    vol_frac=float(vol_frac),
                    max_change=float(max_change),
                    gamma=gamma.copy(),
                    history=history,
                )
            except Exception as _cb_err:
                print(f"  WARNING: on_progress callback error: {_cb_err}")

        # --- Save gamma + preview every iteration (for animation) ---
        iter_dir = opt_output_dir / "iterations" / f"iter{it + 1}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        np.savetxt(str(iter_dir / f"gamma_iter_{it:04d}.txt"),
                   gamma, fmt="%.6f", delimiter="\t")
        save_geometry_preview(
            gamma,
            output_path=str(iter_dir / f"preview_iter_{it:04d}.png"),
            iteration=it,
            f0=f0,
            vol_frac=vol_frac,
        )

        # --- Heavier saves every 5 iterations (checkpoint + history) ---
        if it % 5 == 0 or it == config.max_iter - 1:
            # Save history
            with open(str(iter_dir / "history.json"), "w") as f:
                json.dump(history, f, indent=2)
            # Auto-checkpoint
            save_checkpoint(
                iter_dir / f"checkpoint_iter_{it:04d}.npz",
                gamma, history, mma_state, config, it,
            )


        # --- Stop signal (from GUI) ---
        if stop_event is not None and stop_event.is_set():
            print(f"\n  [STOP] Stop requested after iteration {it}.")
            iter_dir = opt_output_dir / "iterations" / f"iter{it + 1}"
            iter_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(
                iter_dir / f"checkpoint_iter_{it:04d}.npz",
                gamma, history, mma_state, config, it,
            )
            break

        # --- Convergence check ---
        if max_change < config.change_tol and it > 5:
            print(f"\n  [OK] CONVERGED at iteration {it} (max_change={max_change:.6f} < {config.change_tol})")
            break

    # --- Final save (into ExportFiles root) ---
    np.savetxt(str(opt_output_dir / "gamma_final.txt"),
               gamma, fmt="%.6f", delimiter="\t")
    save_geometry_preview(
        gamma,
        output_path=str(opt_output_dir / "preview_final.png"),
        iteration=len(history["iterations"]) - 1,
        f0=history["f0"][-1] if history["f0"] else None,
        vol_frac=history["vol_frac"][-1] if history["vol_frac"] else None,
    )
    with open(str(opt_output_dir / "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  OPTIMIZATION COMPLETE — {len(history['iterations'])} iterations")
    print(f"  Final f0 = {history['f0'][-1]:.4f}")
    print(f"  Results saved to {opt_output_dir}")
    print(f"{'='*70}")

    return gamma, history


# =========================================================================
# CLI ENTRY POINT
# =========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="MMA topology optimization for cold plate channels"
    )
    ap.add_argument("--max-iter", type=int, default=100)
    ap.add_argument("--vol-frac", type=float, default=0.4,
                    help="Target fluid volume fraction (0-1)")
    ap.add_argument("--w-peak", type=float, default=1.0,
                    help="Weight on peak base temperature")
    ap.add_argument("--w-uniformity", type=float, default=0.0,
                    help="Weight on temperature non-uniformity")
    ap.add_argument("--move-limit", type=float, default=0.2,
                    help="MMA move limit per iteration")
    ap.add_argument("--filter-radius", type=float, default=3.0,
                    help="Density filter radius in cells (0=off)")
    ap.add_argument("--change-tol", type=float, default=0.01,
                    help="Convergence tolerance on max design change")
    args = ap.parse_args()

    config = OptConfig()
    config.max_iter = args.max_iter
    config.vol_frac_target = args.vol_frac
    config.w_peak = args.w_peak
    config.w_uniformity = args.w_uniformity
    config.move_limit = args.move_limit
    config.filter_radius = args.filter_radius
    config.change_tol = args.change_tol

    run_optimization(config)


if __name__ == "__main__":
    main()
