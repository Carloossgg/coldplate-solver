"""
run_solvers.py

Automates running the fluid solver (simple.exe) and thermal solver
(thermal_solver.exe) in sequence, then archives results to a per-iteration
subfolder under ExportFiles/iterations/.

Designed to be called per-iteration during MMA topology optimization.

Usage (standalone test):
    python run_solvers.py                     # Run once, archive as iter_0
    python run_solvers.py --iteration 5       # Archive as iter_5
    python run_solvers.py --skip-fluid        # Skip fluid solver (thermal only)
    python run_solvers.py --qk 10 --mode 1    # Pass RAMP params to thermal solver
"""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path


# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
def _get_project_root() -> Path:
    """Return the project root, handling PyInstaller frozen bundles."""
    if getattr(sys, 'frozen', False):
        # When frozen, the exe lives at dist/MMA_Optimizer/MMA_Optimizer.exe
        # ExportFiles and user data should be next to the exe.
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent


def _get_bundle_dir() -> Path:
    """Return the directory containing bundled data files (solvers, .py scripts).
    
    In frozen mode this is sys._MEIPASS (_internal/ folder).
    In normal mode it's the same as project root.
    """
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent


PROJECT_ROOT = _get_project_root()
_BUNDLE_DIR  = _get_bundle_dir()
EXPORT_DIR = PROJECT_ROOT / "ExportFiles"
ITERATIONS_DIR = EXPORT_DIR / "iterations"

SIMPLE_EXE = _BUNDLE_DIR / "simple.exe"
THERMAL_EXE = _BUNDLE_DIR / "ThermalSolver" / "thermal_solver.exe"


# Files to archive after each iteration
FILES_TO_ARCHIVE = [
    # Thermal results (optimization objectives)
    "thermal_metrics.txt",
    "T_bottom.txt",
    # Input geometry for this iteration
    "geometry_fluid.txt",
    "geometry_thermal.txt",
    # Fluid params
    "fluid_params.txt",
    "thermal_params.txt",
    # Convergence info
    "residuals.txt",
    "pressure_drop_history.txt",
    # Velocity / pressure fields (needed for sensitivity analysis later)
    "u_thermal.txt",
    "v_thermal.txt",
    "pressure_thermal.txt",
    # Thermal conductivity field
    "k_thermal.txt",
]


# ---------------------------------------------------------------------------
# Active-process handle  (lets the GUI kill the solver on window close)
# ---------------------------------------------------------------------------
_active_proc: "subprocess.Popen | None" = None


def kill_active_process() -> None:
    """Terminate the currently running solver subprocess (if any).
    
    Uses `taskkill /F /T` on Windows to kill the entire process tree
    (catches OpenMP workers and any children simple.exe spawns).
    Falls back to proc.kill() on other platforms.
    """
    global _active_proc
    proc = _active_proc
    if proc is None:
        return
    try:
        import sys as _sys
        if _sys.platform == "win32":
            # /F = force, /T = include child process tree
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                capture_output=True,
            )
        else:
            import os, signal
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
        proc.wait(timeout=3)
    except Exception:
        pass
    _active_proc = None


# ---------------------------------------------------------------------------
# Helper: run a subprocess with real-time stdout streaming
# ---------------------------------------------------------------------------
def _run_with_streaming(cmd: list[str], cwd: str, label: str) -> tuple[int, str]:
    """
    Run a command, streaming its stdout/stderr to the terminal in real-time.

    Returns (return_code, captured_log_string).
    """
    global _active_proc
    log_lines = []
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,   # merge stderr into stdout
        text=True,
        bufsize=1,                  # line-buffered
    )
    _active_proc = proc
    try:
        for line in proc.stdout:
            print(f"  [{label}] {line}", end="")
            log_lines.append(line)
    except KeyboardInterrupt:
        proc.terminate()
        proc.wait()
        raise
    proc.wait()
    _active_proc = None   # clear only after process has fully exited
    return proc.returncode, "".join(log_lines)


# ---------------------------------------------------------------------------
# Thermal metrics parser
# ---------------------------------------------------------------------------
def parse_thermal_metrics(metrics_path: str | Path) -> dict[str, float]:
    """
    Parse ExportFiles/thermal_metrics.txt into a dict of metric_name -> value.

    Example file content:
        T_avg_base    74.0062
        T_uniformity_base   0.8627
        ...

    Returns dict like:
        {"T_avg_base": 74.0062, "T_uniformity_base": 0.8627, ...}
    """
    metrics = {}
    path = Path(metrics_path)
    if not path.exists():
        print(f"  WARNING: {path} not found — cannot parse thermal metrics.")
        return metrics

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            # Match lines like: "T_avg_base    74.0062"
            match = re.match(r"^(\S+)\s+([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
            if match:
                key = match.group(1)
                val = float(match.group(2))
                metrics[key] = val
    return metrics


# ---------------------------------------------------------------------------
# Core solver runner
# ---------------------------------------------------------------------------
def run_iteration(
    iteration: int = 0,
    skip_fluid: bool = False,
    qk: float = 1.0,
    k_interp_mode: int = 1,
    verbose: bool = True,
) -> dict:
    """
    Run the fluid + thermal solvers and archive results.

    Parameters
    ----------
    iteration : int
        Iteration index (used for naming the archive subfolder).
    skip_fluid : bool
        If True, skip simple.exe and only run the thermal solver.
        Useful when the fluid field hasn't changed (e.g., same geometry).
    qk : float
        RAMP convexity parameter for the thermal solver (continuation: 1→3→10→30).
    k_interp_mode : int
        Thermal conductivity interpolation mode (0=Linear, 1=RAMP, 2=Harmonic).
    verbose : bool
        Print progress to console.

    Returns
    -------
    dict with keys:
        "success" : bool
        "metrics" : dict of thermal metric name -> value
        "iter_dir": str path to the iteration archive folder
        "fluid_log": str (stdout from simple.exe)
        "thermal_log": str (stdout from thermal_solver.exe)
    """
    result = {
        "success": False,
        "metrics": {},
        "iter_dir": "",
        "fluid_log": "",
        "thermal_log": "",
    }

    # ------------------------------------------------------------------
    # 1. Run fluid solver (simple.exe)
    # ------------------------------------------------------------------
    if not skip_fluid:
        if not SIMPLE_EXE.exists():
            print(f"ERROR: Fluid solver not found at {SIMPLE_EXE}")
            return result

        if verbose:
            print(f"\n{'='*60}")
            print(f"  ITERATION {iteration} — Running fluid solver (simple.exe)")
            print(f"{'='*60}")

        t0 = time.time()
        retcode, log = _run_with_streaming(
            [str(SIMPLE_EXE)], str(PROJECT_ROOT), "FLUID"
        )
        elapsed = time.time() - t0
        result["fluid_log"] = log

        if retcode != 0:
            print(f"ERROR: simple.exe exited with code {retcode}")
            return result

        if verbose:
            print(f"\n  Fluid solver completed in {elapsed:.1f}s (exit code 0)")
    else:
        if verbose:
            print(f"\n  ITERATION {iteration} — Skipping fluid solver (--skip-fluid)")

    # ------------------------------------------------------------------
    # 2. Run thermal solver (thermal_solver.exe)
    # ------------------------------------------------------------------
    if not THERMAL_EXE.exists():
        print(f"ERROR: Thermal solver not found at {THERMAL_EXE}")
        return result

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ITERATION {iteration} — Running thermal solver")
        print(f"  qk={qk}, mode={k_interp_mode}")
        print(f"{'='*60}")

    thermal_cmd = [
        str(THERMAL_EXE),
        str(EXPORT_DIR),       # directory argument
        str(qk),               # RAMP parameter
        str(k_interp_mode),    # interpolation mode
    ]

    t0 = time.time()
    retcode, log = _run_with_streaming(
        thermal_cmd, str(PROJECT_ROOT), "THERMAL"
    )
    elapsed = time.time() - t0
    result["thermal_log"] = log

    if retcode != 0:
        print(f"ERROR: thermal_solver.exe exited with code {retcode}")
        return result

    if verbose:
        print(f"\n  Thermal solver completed in {elapsed:.1f}s (exit code 0)")

    # ------------------------------------------------------------------
    # 3. Parse thermal metrics
    # ------------------------------------------------------------------
    metrics_path = EXPORT_DIR / "thermal_metrics.txt"
    result["metrics"] = parse_thermal_metrics(metrics_path)

    if verbose and result["metrics"]:
        print(f"\n  --- Thermal Metrics (iter {iteration}) ---")
        for k, v in result["metrics"].items():
            print(f"    {k:25s} = {v}")

    # ------------------------------------------------------------------
    # 4. Archive results to iteration subfolder
    # ------------------------------------------------------------------
    # 1-indexed folder name: iteration 0 → iter1, iteration 1 → iter2, ...
    iter_dir = ITERATIONS_DIR / f"iter{iteration + 1}"
    iter_dir.mkdir(parents=True, exist_ok=True)
    result["iter_dir"] = str(iter_dir)

    archived = 0
    for fname in FILES_TO_ARCHIVE:
        src = EXPORT_DIR / fname
        if src.exists():
            shutil.copy2(str(src), str(iter_dir / fname))
            archived += 1

    # Save solver logs
    log_path = iter_dir / "solver_log.txt"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write(f"=== Iteration {iteration} ===\n")
        f.write(f"qk={qk}, k_interp_mode={k_interp_mode}, skip_fluid={skip_fluid}\n\n")
        if result["fluid_log"]:
            f.write("=== FLUID SOLVER (simple.exe) ===\n")
            f.write(result["fluid_log"])
            f.write("\n\n")
        f.write("=== THERMAL SOLVER (thermal_solver.exe) ===\n")
        f.write(result["thermal_log"])

    if verbose:
        print(f"\n  Archived {archived} files to {iter_dir}")

    result["success"] = True
    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Run fluid + thermal solvers and archive results."
    )
    ap.add_argument(
        "--iteration", type=int, default=0,
        help="Iteration index for archiving (default: 0)",
    )
    ap.add_argument(
        "--skip-fluid", action="store_true",
        help="Skip the fluid solver (run thermal only)",
    )
    ap.add_argument(
        "--qk", type=float, default=1.0,
        help="RAMP convexity parameter for thermal solver (default: 1.0)",
    )
    ap.add_argument(
        "--mode", type=int, default=1,
        help="Thermal conductivity interpolation mode (0=Linear, 1=RAMP, 2=Harmonic)",
    )
    args = ap.parse_args()

    result = run_iteration(
        iteration=args.iteration,
        skip_fluid=args.skip_fluid,
        qk=args.qk,
        k_interp_mode=args.mode,
    )

    if result["success"]:
        print(f"\n  [OK] Iteration {args.iteration} completed successfully.")
        print(f"    Results archived to: {result['iter_dir']}")
    else:
        print(f"\n  [FAIL] Iteration {args.iteration} FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
