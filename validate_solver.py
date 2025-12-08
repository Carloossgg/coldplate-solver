"""
Validation Script for Improved Heat Solver (Mandel Model)
Checks global energy balance: Q_in (Source) vs Q_out (Convection)
"""

import numpy as np
from heat_solver import SolverParams, SolverInputs, HeatTransferSolver

def run_validation():
    print("Running Energy Balance Validation...")
    
    # 1. Setup Parameters - FINER MESH
    # Increased M/N from 20 to 50 to capture leading edge effect
    p = SolverParams(
        M=50, N=50,
        L_x=0.01, L_y=0.01,
        H_t=0.5e-3, # 1mm channel height
        q0=50000.0   # 5 W/cm2 input
    )
    
    solver = HeatTransferSolver(p)
    
    # 2. Create Geometry (Single Channel)
    gamma = np.zeros((p.M, p.N))
    # Channel in middle 50%
    j_start, j_end = p.N//4, 3*p.N//4
    gamma[:, j_start:j_end] = 1.0 
    
    # 3. Create Velocity Field (Poiseuille in channel, 0 elsewhere)
    u = np.zeros((p.M + 1, p.N + 1))
    v = np.zeros((p.M + 1, p.N + 1))
    
    # Channel Y coords
    y = np.linspace(0, p.L_y, p.N + 1)
    y_start = y[j_start]
    y_end = y[j_end]
    w_ch = y_end - y_start
    
    u_mean = 0.5 # m/s
    
    for j in range(p.N + 1):
        if y_start <= y[j] <= y_end:
            # Parabolic profile
            rel_y = (y[j] - y_start) / w_ch
            u[:, j] = 6.0 * u_mean * rel_y * (1.0 - rel_y)
            
    inputs = SolverInputs(gamma=gamma, u=u, v=v)
    
    # 4. Solve
    out = solver.solve(inputs)
    
    # 5. Calculate Energy Balance
    area_total = p.L_x * p.L_y
    Q_in = p.q0 * area_total
    
    # Q_out integration at outlet
    T_out = solver.reshape_to_grid(out.T_t)[-1, :]
    u_out = u[-1, :]
    
    Q_out_accum = 0.0
    dy = p.L_y / p.N
    for j in range(p.N + 1):
        weight = 0.5 if (j==0 or j==p.N) else 1.0
        flux = p.rho * p.C_p * u_out[j] * (T_out[j] - p.T_inlet)
        Q_out_accum += flux * (2.0 * p.H_t) * (dy * weight)
        
    print("\n" + "="*30)
    print("RESULTS (Fine Mesh 50x50)")
    print("="*30)
    print(f"Total Heat Input  (Q_in) : {Q_in:.4f} W")
    print(f"Total Heat Removed(Q_out): {Q_out_accum:.4f} W")
    
    error = abs(Q_in - Q_out_accum) / Q_in * 100
    print(f"Energy Imbalance         : {error:.2f}%")
    print("="*30)
    
    if error < 10.0:
        print("PASS: Energy balance is excellent (<10%).")
        print("The solver is now accurately tracking energy.")
    else:
        print("FAIL: Energy balance error is still high.")

if __name__ == "__main__":
    run_validation()