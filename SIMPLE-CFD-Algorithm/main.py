"""
File: main.py
Author: Carlos Goni Gill
Description: Driver for thermal post-processing; loads flow/geometry exports from ExportFiles/ and calls the
  conjugate heat transfer solver to produce thermal VTK outputs (fluid/substrate temperatures, pressure, velocity
  magnitude, vectors).
"""

import numpy as np
import sys
import os
from heat_solver import HeatTransferSolver, SolverParams, SolverInputs

def main():
    print("--- Starting Thermal Simulation ---")
    
    INPUT_DIR = "ExportFiles"
    
    # 1. Load Parameters
    params_file = os.path.join(INPUT_DIR, "thermal_params.txt")
    if not os.path.exists(params_file):
        print(f"Error: {params_file} not found. Run generate_geometry.py first.")
        return

    print(f"Loading Parameters from {params_file}...")
    with open(params_file, 'r') as f:
        line = f.readline().strip().split()
        # Format: M  N  dy  dx  HeatFlux(W/m^2)
        M_grid = int(line[0])
        N_grid = int(line[1])
        dy_val = float(line[2])
        dx_val = float(line[3])
        q_flux_val = float(line[4])

    print(f"Thermal Grid: {M_grid} (Y) x {N_grid} (X)")
    print(f"Cell Size: {dx_val:.6f} m")
    print(f"Heat Flux: {q_flux_val} W/m^2")

    # 2. Load Matrices
    try:
        # Load Files
        gamma_raw = np.loadtxt(os.path.join(INPUT_DIR, "geometry_thermal.txt"))
        u_raw = np.loadtxt(os.path.join(INPUT_DIR, "u_thermal.txt"))
        v_raw = np.loadtxt(os.path.join(INPUT_DIR, "v_thermal.txt"))
        
        # Load Pressure (If available, otherwise None)
        p_path = os.path.join(INPUT_DIR, "pressure_thermal.txt")
        if os.path.exists(p_path):
            p_raw = np.loadtxt(p_path)
            # Transpose Pressure
            p_field = p_raw.T
            print("Pressure data loaded.")
        else:
            p_field = None
            print("Warning: pressure_thermal.txt not found. Pressure will be missing in VTK.")
        
        # TRANSPOSE Logic: Aligning C++ (Rows, Cols) to Solver (X, Y)
        gamma_field = gamma_raw.T
        u_field = u_raw.T
        v_field = v_raw.T
        
        # Threshold geometry
        gamma_field = np.where(gamma_field > 0.5, 1.0, 0.0)
        
    except Exception as e:
        print(f"Error loading data matrices: {e}")
        return

    # 3. Setup Solver Parameters
    real_M, real_N = gamma_field.shape
    
    L_x = real_M * dx_val
    L_y = real_N * dy_val

    params = SolverParams(
        M = real_M,      
        N = real_N,
        L_x = L_x,
        L_y = L_y,
        k_f = 0.6, rho = 998.0, C_p = 4180.0, k_s = 400.0
    )
    
    # 4. Setup Inputs (Now including Pressure)
    q0_field = np.full((real_M, real_N), q_flux_val)
    
    inputs = SolverInputs(
        gamma = gamma_field,
        u = u_field,
        v = v_field,
        q0 = q0_field,
        p = p_field
    )
    
    # 5. Run Solver
    solver = HeatTransferSolver(params)
    
    outputs = solver.solve(inputs)
    T_fluid = outputs.T_t
    T_solid = outputs.T_b
    
    print("Optimization/Simulation Complete.")
    
    # 6. Save VTK (Detailed)
    vtk_path = os.path.join(INPUT_DIR, "thermal_results.vtk")
    solver.save_vtk(inputs, T_fluid, T_solid, vtk_path)
    print(f"Results saved to {vtk_path}")
    print("VTK includes: T_Fluid, T_Substrate, Pressure, Velocity_Magnitude, Heat_Source, Geometry, Vectors.")

if __name__ == "__main__":
    main()