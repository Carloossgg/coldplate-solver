"""
main.py

Single entrypoint (keep the project simple):
- Loads `ExportFiles/*_thermal.txt`
- Optional downsampling (`--fx`, `--fy`)
- Runs the 3D thermal solver in `heat_solver_3d.py`
- Writes VTK + `ExportFiles/thermal_balance.txt`

Run:
  .\\venv312\\Scripts\\Activate.ps1
  python main.py --input ExportFiles --fx 4 --fy 4 --nz-solid 3 --nz-fluid 5 --threads 16
"""

from __future__ import annotations

import argparse
import os


def enforce_velocity_mask(u, v, gamma, solid_threshold: float = 0.5):
    """Zero velocity inside solid (gamma>=threshold)."""
    import numpy as np

    mask = np.asarray(gamma) < solid_threshold
    return np.where(mask, u, 0.0), np.where(mask, v, 0.0)


def block_mean(a, fy: int, fx: int):
    """Downsample by integer block averaging."""
    import numpy as np

    a = np.asarray(a)
    M, N = a.shape
    M2 = (M // fy) * fy
    N2 = (N // fx) * fx
    a = a[:M2, :N2]
    return a.reshape(M2 // fy, fy, N2 // fx, fx).mean(axis=(1, 3))


def _read_field(path: str, M: int, N: int, label: str):
    import numpy as np

    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {label}: {path}")
    arr = np.loadtxt(path, dtype=float)
    arr = np.asarray(arr, dtype=float)
    if arr.size != M * N:
        raise ValueError(f"{label} in {path} has size {arr.size}, expected {M*N}")
    return arr.reshape((M, N))


def load_fvm_exports(input_dir: str, fx: int = 1, fy: int = 1):
    """
    Load FVM (SIMPLE) outputs on their native grid.
    No refinement or synthetic fields are generated here.
    """

    params_path = os.path.join(input_dir, "thermal_params.txt")
    geom_path = os.path.join(input_dir, "geometry_thermal.txt")
    u_path = os.path.join(input_dir, "u_thermal.txt")
    v_path = os.path.join(input_dir, "v_thermal.txt")
    pressure_candidates = ["p_thermal.txt", "pressure_thermal.txt", "p.txt"]

    if not os.path.exists(params_path):
        raise FileNotFoundError(f"Missing thermal params: {params_path}")

    with open(params_path, "r") as f:
        line = f.readline().strip().split()
        if len(line) < 6:
            raise ValueError(f"Expected 6 columns in {params_path}, got {len(line)}")
        M_orig, N_orig = int(line[0]), int(line[1])
        dy_orig, dx_orig = float(line[2]), float(line[3])
        q_flux_val, Ht_channel = float(line[4]), float(line[5])

    gamma_raw = _read_field(geom_path, M_orig, N_orig, "gamma")
    u_raw = _read_field(u_path, M_orig, N_orig, "u")
    v_raw = _read_field(v_path, M_orig, N_orig, "v")

    p_raw = None
    p_path_used = None
    for cand in pressure_candidates:
        cand_path = os.path.join(input_dir, cand)
        if os.path.exists(cand_path):
            p_raw = _read_field(cand_path, M_orig, N_orig, "p")
            p_path_used = cand_path
            break

    # Mask velocities on fine grid first (important before averaging)
    u_raw, v_raw = enforce_velocity_mask(u_raw, v_raw, gamma_raw)

    # Optional downsampling
    fx = max(1, int(fx))
    fy = max(1, int(fy))
    if fx > 1 or fy > 1:
        gamma_c = block_mean(gamma_raw, fy, fx)
        u_c = block_mean(u_raw, fy, fx)
        v_c = block_mean(v_raw, fy, fx)
        gamma_field = (gamma_c > 0.5).astype(float)
        u_field, v_field = enforce_velocity_mask(u_c, v_c, gamma_field)
        dx_val = dx_orig * fx
        dy_val = dy_orig * fy
        target_shape = gamma_field.shape
    else:
        gamma_field = gamma_raw
        u_field, v_field = enforce_velocity_mask(u_raw, v_raw, gamma_raw)
        dx_val = dx_orig
        dy_val = dy_orig
        target_shape = gamma_raw.shape

    import numpy as np
    p_field = p_raw
    q0_field = np.full(target_shape, q_flux_val, dtype=float)

    return {
        "M": target_shape[0],
        "N": target_shape[1],
        "dx": dx_val,
        "dy": dy_val,
        "Ht": Ht_channel,
        "q_flux": q_flux_val,
        "gamma": gamma_field,
        "u": u_field,
        "v": v_field,
        "p": p_field,
        "q0": q0_field,
        "base_shape": (M_orig, N_orig),
        "pressure_path": p_path_used,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="ExportFiles")
    ap.add_argument("--fx", type=int, default=1)
    ap.add_argument("--fy", type=int, default=1)
    ap.add_argument("--nz-solid", type=int, default=3)
    ap.add_argument("--nz-fluid", type=int, default=5)
    ap.add_argument("--threads", type=int, default=0, help="Set OMP/MKL/OPENBLAS thread counts (best-effort)")
    ap.add_argument("--out", default="", help="VTK output path (default: ExportFiles/thermal_results_3d.vtk)")
    ap.add_argument("--rtol", type=float, default=1e-6, help="Relative tolerance for BiCGSTAB solver")
    ap.add_argument("--maxiter", type=int, default=500, help="Max iterations for BiCGSTAB solver")
    ap.add_argument("--ilu-fill", type=int, default=10, help="ILU fill factor (higher=more accurate, slower)")
    args = ap.parse_args()

    if args.threads and args.threads > 0:
        os.environ["OMP_NUM_THREADS"] = str(args.threads)
        os.environ["MKL_NUM_THREADS"] = str(args.threads)
        os.environ["OPENBLAS_NUM_THREADS"] = str(args.threads)
        os.environ["NUMEXPR_NUM_THREADS"] = str(args.threads)

    # Heavy imports after thread env vars
    from heat_solver_3d import HeatTransferSolver3D, SolverParams3D, SolverInputs3D

    INPUT_DIR = args.input
    H_B_SUBSTRATE = 0.0005
    CHANNEL_WIDTH = 0.0006

    if not os.path.exists(INPUT_DIR):
        raise FileNotFoundError(f"Input directory not found: {INPUT_DIR}")

    print("--- Starting Thermal Simulation (3D) ---")
    data = load_fvm_exports(INPUT_DIR, fx=args.fx, fy=args.fy)
    print(
        f"Loaded ExportFiles {data['base_shape'][0]}x{data['base_shape'][1]} "
        f"-> {data['M']}x{data['N']} (fx={max(1,args.fx)}, fy={max(1,args.fy)})"
    )
    if data["pressure_path"]:
        print(f"Pressure field loaded from {data['pressure_path']}")

    params = SolverParams3D(
        M=data["M"],
        N=data["N"],
        L_x=data["N"] * data["dx"],
        L_y=data["M"] * data["dy"],
        H_t=data["Ht"],
        H_b=H_B_SUBSTRATE,
        channel_width=CHANNEL_WIDTH,
        nz_solid=int(args.nz_solid),
        nz_fluid=int(args.nz_fluid),
        rtol=args.rtol,
        maxiter=args.maxiter,
        ilu_fill_factor=args.ilu_fill,
    )

    inputs = SolverInputs3D(gamma=data["gamma"], u=data["u"], v=data["v"], q0=data["q0"])

    solver = HeatTransferSolver3D(params)
    outputs = solver.solve(inputs)

    # Write energy balance to file (solver also prints it)
    bal = solver.energy_balance(inputs, outputs)
    bal_path = os.path.join(INPUT_DIR, "thermal_balance.txt")
    with open(bal_path, "w", encoding="utf-8") as f:
        for k, v in bal.items():
            f.write(f"{k} {v}\n")
    print(f"Saved energy balance: {bal_path}")

    output_path = args.out or os.path.join(INPUT_DIR, "thermal_results_3d.vtk")
    solver.save_vtk(outputs, output_path)
    print(f"Saved VTK: {output_path}")


if __name__ == "__main__":
    main()