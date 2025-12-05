"""
Two-Layer Heat Transfer Solver for Microchannel Heat Sink Topology Optimization
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
from dataclasses import dataclass
from typing import Tuple, Optional
import time

@dataclass
class SolverParams:
    """Physical and numerical parameters for the heat transfer solver"""
    M: int = 300          
    N: int = 300          
    L_x: float = 0.03     
    L_y: float = 0.03     
    
    H_t: float = 250e-6   
    H_b: float = 100e-6   
    
    rho: float = 998.0    
    C_p: float = 4180.0   
    k_f: float = 0.6      
    k_s: float = 400.0    
    
    @property
    def dx(self) -> float:
        return self.L_x / self.M
        
    @property
    def dy(self) -> float:
        return self.L_y / self.N

@dataclass
class SolverInputs:
    """Input fields for the solver"""
    gamma: np.ndarray     
    u: np.ndarray         
    v: np.ndarray         
    q0: np.ndarray        
    p: Optional[np.ndarray] = None # Added Pressure field (optional)

@dataclass
class SolverOutputs:
    """Output fields from the solver"""
    T_t: np.ndarray       
    T_b: np.ndarray       
    compliance: float     

class HeatTransferSolver:
    
    def __init__(self, params: SolverParams):
        self.params = params
        self.nnodes = (params.M + 1) * (params.N + 1)
        self.nelems = params.M * params.N
        
    def _get_element_material_properties(self, gamma_e: float) -> Tuple[float, float]:
        k_e = gamma_e * self.params.k_f + (1.0 - gamma_e) * self.params.k_s
        rho_cp_e = self.params.rho * self.params.C_p * gamma_e 
        return k_e, rho_cp_e

    def solve(self, inputs: SolverInputs) -> SolverOutputs:
        t_start = time.time()
        print(f"Setting up system ({self.nelems * 2} elements)...")
        
        N_dof = 2 * self.nelems
        rows = []
        cols = []
        vals = []
        rhs = np.zeros(N_dof)
        
        dist = 0.5 * (self.params.H_t + self.params.H_b)
        h_int = self.params.k_s / dist
        
        dx = self.params.dx
        dy = self.params.dy
        Vol = dx * dy
        
        # Simplified loop for brevity, same physics as before
        for i in range(self.params.M):
            for j in range(self.params.N):
                idx = i * self.params.N + j  
                dof_t = idx              
                dof_b = idx + self.nelems 
                
                g = inputs.gamma[i, j]
                u = inputs.u[i, j]
                v = inputs.v[i, j]
                
                k_eff, rho_cp = self._get_element_material_properties(g)
                
                # --- 1. TOP LAYER ---
                # Convection X
                F_x = rho_cp * u * (self.params.H_t * dy)
                if i < self.params.M - 1: 
                    idx_E = (i + 1) * self.params.N + j
                    rows.append(dof_t); cols.append(dof_t); vals.append(max(F_x, 0))
                    rows.append(dof_t); cols.append(idx_E); vals.append(min(F_x, 0))
                else: 
                    rows.append(dof_t); cols.append(dof_t); vals.append(max(F_x, 0))
                    
                if i > 0: 
                    idx_W = (i - 1) * self.params.N + j
                    rows.append(dof_t); cols.append(idx_W); vals.append(-max(F_x, 0))
                    rows.append(dof_t); cols.append(dof_t); vals.append(-min(F_x, 0))
                else: 
                    if F_x < 0:
                        rows.append(dof_t); cols.append(dof_t); vals.append(-F_x)

                # Convection Y
                F_y = rho_cp * v * (self.params.H_t * dx)
                if j < self.params.N - 1: 
                    idx_N = i * self.params.N + (j + 1)
                    rows.append(dof_t); cols.append(dof_t); vals.append(max(F_y, 0))
                    rows.append(dof_t); cols.append(idx_N); vals.append(min(F_y, 0))
                else: 
                    rows.append(dof_t); cols.append(dof_t); vals.append(max(F_y, 0))
                    
                if j > 0: 
                    idx_S = i * self.params.N + (j - 1)
                    rows.append(dof_t); cols.append(idx_S); vals.append(-max(F_y, 0))
                    rows.append(dof_t); cols.append(dof_t); vals.append(-min(F_y, 0))
                else: 
                    rows.append(dof_t); cols.append(dof_t); vals.append(-min(F_y, 0))

                # Diffusion (Simplified 5-point)
                D_x = k_eff * (self.params.H_t * dy) / dx
                D_y = k_eff * (self.params.H_t * dx) / dy
                
                diag_t = 0
                if i < self.params.M - 1:
                    idx_E = (i + 1) * self.params.N + j
                    rows.append(dof_t); cols.append(idx_E); vals.append(-D_x)
                    diag_t += D_x
                if i > 0:
                    idx_W = (i - 1) * self.params.N + j
                    rows.append(dof_t); cols.append(idx_W); vals.append(-D_x)
                    diag_t += D_x
                if j < self.params.N - 1:
                    idx_N = i * self.params.N + (j + 1)
                    rows.append(dof_t); cols.append(idx_N); vals.append(-D_y)
                    diag_t += D_y
                if j > 0:
                    idx_S = i * self.params.N + (j - 1)
                    rows.append(dof_t); cols.append(idx_S); vals.append(-D_y)
                    diag_t += D_y
                rows.append(dof_t); cols.append(dof_t); vals.append(diag_t)

                # Coupling
                C = h_int * Vol
                rows.append(dof_t); cols.append(dof_t); vals.append(C)
                rows.append(dof_t); cols.append(dof_b); vals.append(-C)
                
                # --- 2. BOTTOM LAYER ---
                k_sub = self.params.k_s
                D_x_b = k_sub * (self.params.H_b * dy) / dx
                D_y_b = k_sub * (self.params.H_b * dx) / dy
                
                diag_b = 0
                if i < self.params.M - 1:
                    idx_E = (i + 1) * self.params.N + j + self.nelems
                    rows.append(dof_b); cols.append(idx_E); vals.append(-D_x_b)
                    diag_b += D_x_b
                if i > 0:
                    idx_W = (i - 1) * self.params.N + j + self.nelems
                    rows.append(dof_b); cols.append(idx_W); vals.append(-D_x_b)
                    diag_b += D_x_b
                if j < self.params.N - 1:
                    idx_N = i * self.params.N + (j + 1) + self.nelems
                    rows.append(dof_b); cols.append(idx_N); vals.append(-D_y_b)
                    diag_b += D_y_b
                if j > 0:
                    idx_S = i * self.params.N + (j - 1) + self.nelems
                    rows.append(dof_b); cols.append(idx_S); vals.append(-D_y_b)
                    diag_b += D_y_b
                    
                rows.append(dof_b); cols.append(dof_b); vals.append(diag_b)

                # Coupling
                rows.append(dof_b); cols.append(dof_b); vals.append(C)
                rows.append(dof_b); cols.append(dof_t); vals.append(-C)
                
                # Heat Source
                rhs[dof_b] += inputs.q0[i, j] * Vol

        print("Assembling sparse matrix...")
        A_mat = sparse.coo_matrix((vals, (rows, cols)), shape=(N_dof, N_dof))
        A_csr = A_mat.tocsr()
        
        print("Solving linear system...")
        T_vec = spsolve(A_csr, rhs)
        
        T_t = T_vec[:self.nelems]
        T_b = T_vec[self.nelems:]
        
        print(f"Solved in {time.time() - t_start:.2f}s")
        return SolverOutputs(T_t, T_b, 0.0)

    def reshape_to_grid(self, T_flat):
        return T_flat.reshape((self.params.M, self.params.N))

    def save_vtk(self, inputs: SolverInputs, T_t, T_b, filename="thermal_results.vtk"):
        print(f"Writing detailed results to {filename}...")
        
        # Reshape Temperatures
        T_top = self.reshape_to_grid(T_t)
        T_bot = self.reshape_to_grid(T_b)
        
        nx = self.params.M
        ny = self.params.N
        
        with open(filename, 'w') as f:
            f.write("# vtk DataFile Version 3.0\n")
            f.write("Two-Layer Thermal Solver Results\n")
            f.write("ASCII\n")
            f.write("DATASET STRUCTURED_POINTS\n")
            f.write(f"DIMENSIONS {nx} {ny} 1\n")
            f.write(f"ORIGIN 0 0 0\n")
            f.write(f"SPACING {self.params.dx} {self.params.dy} 1\n")
            f.write(f"POINT_DATA {nx * ny}\n")
            
            # 1. SCALAR: Fluid Temperature (Top)
            f.write("SCALARS T_Fluid double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{T_top[i, j]}\n")
            
            # 2. SCALAR: Substrate Temperature (Bottom)
            f.write("SCALARS T_Substrate double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{T_bot[i, j]}\n")
            
            # 3. SCALAR: Velocity Magnitude
            f.write("SCALARS Velocity_Magnitude double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    u_val = inputs.u[i, j]
                    v_val = inputs.v[i, j]
                    mag = np.sqrt(u_val**2 + v_val**2)
                    f.write(f"{mag}\n")

            # 4. SCALAR: Pressure (if available)
            if inputs.p is not None:
                f.write("SCALARS Pressure double 1\n")
                f.write("LOOKUP_TABLE default\n")
                for j in range(ny):
                    for i in range(nx):
                        f.write(f"{inputs.p[i, j]}\n")

            # 5. SCALAR: Heat Source (q0)
            f.write("SCALARS Heat_Source_q0 double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{inputs.q0[i, j]}\n")

            # 6. SCALAR: Geometry
            f.write("SCALARS Gamma double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{inputs.gamma[i, j]}\n")
            
            # 7. VECTORS: Velocity
            f.write("VECTORS Velocity double\n")
            for j in range(ny):
                for i in range(nx):
                    f.write(f"{inputs.u[i, j]} {inputs.v[i, j]} 0.0\n")
            
            # 8. VECTORS: Conductive Heat Flux (Bottom Layer) q = -k grad T
            f.write("VECTORS HeatFlux_Vector_Bottom double\n")
            for j in range(ny):
                for i in range(nx):
                    # Compute local gradient (simple central difference)
                    if i == 0: dTdx = (T_bot[i+1,j] - T_bot[i,j]) / self.params.dx
                    elif i == nx-1: dTdx = (T_bot[i,j] - T_bot[i-1,j]) / self.params.dx
                    else: dTdx = (T_bot[i+1,j] - T_bot[i-1,j]) / (2*self.params.dx)
                    
                    if j == 0: dTdy = (T_bot[i,j+1] - T_bot[i,j]) / self.params.dy
                    elif j == ny-1: dTdy = (T_bot[i,j] - T_bot[i,j-1]) / self.params.dy
                    else: dTdy = (T_bot[i,j+1] - T_bot[i,j-1]) / (2*self.params.dy)
                    
                    k = self.params.k_s
                    f.write(f"{-k*dTdx} {-k*dTdy} 0.0\n")