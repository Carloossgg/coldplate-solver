"""
Two-Layer Heat Transfer Solver for Microchannel Heat Sink Topology Optimization

Based on: Yan et al. (2019) "Topology optimization of microchannel heat sinks 
using a two-layer model" - International Journal of Heat and Mass Transfer

This solver implements:
- Coupled thermal-fluid layer and substrate heat transfer
- SUPG stabilization for convection-dominated transport
- Design-dependent material interpolation
- Adjoint sensitivity computation for optimization

Author: Carlos Goni
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
    # Grid dimensions
    M: int = 40          # elements in x-direction
    N: int = 40          # elements in y-direction
    L_x: float = 0.01    # domain length in x (m)
    L_y: float = 0.01    # domain length in y (m)
    
    # Layer heights (half-heights as in paper)
    H_t: float = 0.25e-3  # half-height of thermal-fluid layer (m)
    H_b: float = 0.1e-3   # half-height of substrate (m)
    
    # Fluid properties (water at ~20°C)
    rho: float = 998.0    # density (kg/m³)
    C_p: float = 4180.0   # heat capacity (J/kg·K)
    k_f: float = 0.598    # thermal conductivity (W/m·K)
    
    # Solid properties (silicon)
    k_s: float = 149.0    # thermal conductivity (W/m·K)
    
    # Boundary conditions
    T_inlet: float = 0.0  # inlet temperature (°C)
    q0: float = 6e4       # heat flux at substrate bottom (W/m²)
    
    # Optimization parameters
    q_k: float = 1.0      # penalization parameter for conductivity
    p_norm: int = 10      # p-norm exponent for max temperature approx
    
    @property
    def dx(self) -> float:
        return self.L_x / self.M
    
    @property
    def dy(self) -> float:
        return self.L_y / self.N
    
    @property
    def n_nodes(self) -> int:
        return (self.M + 1) * (self.N + 1)
    
    @property
    def n_elements(self) -> int:
        return self.M * self.N


@dataclass
class SolverInputs:
    """Input fields for the heat transfer solver"""
    gamma: np.ndarray      # Design field (M x N), 0=solid, 1=fluid
    u: np.ndarray          # x-velocity at nodes ((M+1) x (N+1))
    v: np.ndarray          # y-velocity at nodes ((M+1) x (N+1))
    q0_field: Optional[np.ndarray] = None  # Optional spatially-varying heat flux (M x N)


@dataclass 
class SolverOutputs:
    """Output fields from the heat transfer solver"""
    T_t: np.ndarray        # Thermal-fluid layer temperature (n_nodes,)
    T_b: np.ndarray        # Substrate temperature (n_nodes,)
    objective: float       # p-norm approximation of max temperature
    dJ_dgamma: np.ndarray  # Sensitivities (n_elements,)
    solve_time: float      # Computation time (s)


class HeatTransferSolver:
    """
    Two-layer heat transfer solver for microchannel heat sinks.
    
    Solves the coupled system:
    
    Design layer (thermal-fluid):
        (2/3)*rho*C*(u·∇T_t) - (49/52)*∇·(k_t*∇T_t) - (h/2H_t)*(T_b - T_t) = 0
    
    Substrate:
        -(k_b/2)*∇²T_b + (h/2H_b)*(T_b - T_t) - q0/(2H_b) = 0
    """
    
    def __init__(self, params: SolverParams):
        self.p = params
        self._setup_mesh()
        self._precompute_element_matrices()
        
    def _setup_mesh(self):
        """Initialize mesh data structures"""
        p = self.p
        
        # Node coordinates
        x = np.linspace(0, p.L_x, p.M + 1)
        y = np.linspace(0, p.L_y, p.N + 1)
        self.X, self.Y = np.meshgrid(x, y, indexing='ij')
        
        # Element connectivity (4-node quads)
        # Node numbering: row-major, (i,j) -> i*(N+1) + j
        self.elem_nodes = np.zeros((p.n_elements, 4), dtype=int)
        for i in range(p.M):
            for j in range(p.N):
                e = i * p.N + j  # element index
                n0 = i * (p.N + 1) + j
                n1 = (i + 1) * (p.N + 1) + j
                n2 = (i + 1) * (p.N + 1) + (j + 1)
                n3 = i * (p.N + 1) + (j + 1)
                self.elem_nodes[e] = [n0, n1, n2, n3]
        
        # Identify inlet nodes (left boundary, x=0)
        self.inlet_nodes = np.array([j for j in range(p.N + 1)])
        
        # Element centroids (for material properties)
        self.elem_x = np.zeros(p.n_elements)
        self.elem_y = np.zeros(p.n_elements)
        for e in range(p.n_elements):
            nodes = self.elem_nodes[e]
            self.elem_x[e] = np.mean(self.X.flatten()[nodes])
            self.elem_y[e] = np.mean(self.Y.flatten()[nodes])
    
    def _precompute_element_matrices(self):
        """Precompute reference element matrices using Gauss quadrature"""
        p = self.p
        
        # 2x2 Gauss quadrature points and weights
        gp = 1.0 / np.sqrt(3.0)
        self.gauss_pts = np.array([[-gp, -gp], [gp, -gp], [gp, gp], [-gp, gp]])
        self.gauss_wts = np.array([1.0, 1.0, 1.0, 1.0])
        
        # Reference element shape functions and derivatives
        # N_i(xi, eta) for 4-node quad
        # N = [(1-xi)(1-eta), (1+xi)(1-eta), (1+xi)(1+eta), (1-xi)(1+eta)] / 4
        
        # Jacobian for rectangular elements
        self.detJ = (p.dx / 2) * (p.dy / 2)
        self.dxi_dx = 2.0 / p.dx
        self.deta_dy = 2.0 / p.dy
        
        # Precompute shape function values and derivatives at Gauss points
        n_gp = len(self.gauss_pts)
        self.N_gp = np.zeros((n_gp, 4))      # shape functions
        self.dNdx_gp = np.zeros((n_gp, 4))   # dN/dx
        self.dNdy_gp = np.zeros((n_gp, 4))   # dN/dy
        
        for g, (xi, eta) in enumerate(self.gauss_pts):
            # Shape functions
            self.N_gp[g] = 0.25 * np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])
            
            # Derivatives w.r.t. physical coordinates
            dNdxi = 0.25 * np.array([-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)])
            dNdeta = 0.25 * np.array([-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)])
            
            self.dNdx_gp[g] = dNdxi * self.dxi_dx
            self.dNdy_gp[g] = dNdeta * self.deta_dy
    
    def compute_material_properties(self, gamma: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute design-dependent material properties using RAMP interpolation.
        
        k_t(γ) = k_f + (k_s - k_f) * (1 - γ) / (1 + q_k * γ)
        
        h = h_t * h_b / (h_t + h_b)  where h_t = 35*k_t/(26*H_t), h_b = k_b/H_b
        """
        p = self.p
        gamma_flat = gamma.flatten()
        
        # Effective thermal conductivity (Eq. 13 from paper)
        k_t = p.k_f + (p.k_s - p.k_f) * (1 - gamma_flat) / (1 + p.q_k * gamma_flat)
        
        # Inter-layer heat transfer coefficient
        h_t = 35.0 * k_t / (26.0 * p.H_t) #Eq. 32
        h_b = p.k_s / p.H_b  # substrate is always solid
        h = (h_t * h_b) / (h_t + h_b)
        
        return k_t, h
    
    def compute_supg_parameter(self, u_e: np.ndarray, v_e: np.ndarray, 
                                k_t_e: float) -> float:
        """
        Compute SUPG stabilization parameter for an element.
        
        τ = h_e / (2|u|) with modification for diffusion
        """
        p = self.p
        
        # Element velocity magnitude (average over element)
        u_mag = np.sqrt(np.mean(u_e)**2 + np.mean(v_e)**2)
        
        if u_mag < 1e-10:
            return 0.0
        
        # Element length in flow direction
        h_e = np.sqrt(p.dx**2 + p.dy**2)
        
        # Effective diffusivity
        alpha_eff = (49.0/52.0) * k_t_e / ((2.0/3.0) * p.rho * p.C_p)
        
        # Peclet number
        Pe = u_mag * h_e / (2.0 * alpha_eff) if alpha_eff > 1e-15 else 1e10
        
        # SUPG parameter with coth correction for low Pe
        if Pe > 1e-3:
            xi = 1.0 / np.tanh(Pe) - 1.0 / Pe  # coth(Pe) - 1/Pe
        else:
            xi = Pe / 3.0  # Taylor expansion for small Pe
        
        tau = xi * h_e / (2.0 * u_mag)
        
        return tau
    
    def assemble_system(self, inputs: SolverInputs) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """
        Assemble the coupled system matrices for T_t and T_b.
        
        System structure:
        [A_tt   A_tb] [T_t]   [b_t]
        [A_bt   A_bb] [T_b] = [b_b]
        """
        p = self.p
        n = p.n_nodes
        
        # Get material properties
        k_t, h = self.compute_material_properties(inputs.gamma)
        
        # Use uniform or spatially-varying heat flux
        if inputs.q0_field is not None:
            q0_elem = inputs.q0_field.flatten()
        else:
            q0_elem = np.full(p.n_elements, p.q0)
        
        # Triplet lists for sparse assembly
        rows, cols, vals = [], [], []
        b = np.zeros(2 * n)
        
        # Flatten velocity arrays
        u_flat = inputs.u.flatten()
        v_flat = inputs.v.flatten()
        
        # Loop over elements
        for e in range(p.n_elements):
            nodes = self.elem_nodes[e]
            
            # Element velocities
            u_e = u_flat[nodes]
            v_e = v_flat[nodes]
            
            # Element material properties
            k_t_e = k_t[e]
            h_e = h[e]
            q0_e = q0_elem[e]
            
            # SUPG parameter
            tau = self.compute_supg_parameter(u_e, v_e, k_t_e)
            
            # Initialize element matrices (4x4 each block)
            A_tt_e = np.zeros((4, 4))
            A_tb_e = np.zeros((4, 4))
            A_bt_e = np.zeros((4, 4))
            A_bb_e = np.zeros((4, 4))
            b_t_e = np.zeros(4)
            b_b_e = np.zeros(4)
            
            # Gauss quadrature
            for g in range(len(self.gauss_pts)):
                N = self.N_gp[g]
                dNdx = self.dNdx_gp[g]
                dNdy = self.dNdy_gp[g]
                w = self.gauss_wts[g] * self.detJ
                
                # Velocity at Gauss point
                u_gp = np.dot(N, u_e)
                v_gp = np.dot(N, v_e)
                
                # ===== T_t equation (thermal-fluid layer) =====
                
                # Convection: (2/3)*rho*C * N_i * (u·∇N_j)
                conv_coeff = (2.0/3.0) * p.rho * p.C_p
                u_dot_grad_N = u_gp * dNdx + v_gp * dNdy
                A_tt_e += conv_coeff * np.outer(N, u_dot_grad_N) * w
                
                # Diffusion: (49/52)*k_t * ∇N_i·∇N_j
                diff_coeff = (49.0/52.0) * k_t_e
                A_tt_e += diff_coeff * (np.outer(dNdx, dNdx) + np.outer(dNdy, dNdy)) * w
                
                # Coupling T_t: (h/2H_t) * N_i * N_j
                coup_t = h_e / (2.0 * p.H_t)
                A_tt_e += coup_t * np.outer(N, N) * w
                
                # Coupling T_b in T_t equation: -(h/2H_t) * N_i * N_j
                A_tb_e -= coup_t * np.outer(N, N) * w
                
                # SUPG stabilization for T_t equation
                if tau > 1e-15:
                    # Test function modification: tau * (u·∇N_i)
                    supg_test = tau * u_dot_grad_N
                    
                    # SUPG convection term
                    A_tt_e += conv_coeff * np.outer(supg_test, u_dot_grad_N) * w
                    
                    # SUPG coupling terms
                    A_tt_e += coup_t * np.outer(supg_test, N) * w
                    A_tb_e -= coup_t * np.outer(supg_test, N) * w
                
                # ===== T_b equation (substrate) =====
                
                # Diffusion: (k_b/2) * ∇N_i·∇N_j
                diff_b = p.k_s / 2.0
                A_bb_e += diff_b * (np.outer(dNdx, dNdx) + np.outer(dNdy, dNdy)) * w
                
                # Coupling T_b: (h/2H_b) * N_i * N_j
                coup_b = h_e / (2.0 * p.H_b)
                A_bb_e += coup_b * np.outer(N, N) * w
                
                # Coupling T_t in T_b equation: -(h/2H_b) * N_i * N_j
                A_bt_e -= coup_b * np.outer(N, N) * w
                
                # Source term: (q0/2H_b) * N_i
                b_b_e += (q0_e / (2.0 * p.H_b)) * N * w
            
            # Assemble into global system
            for i_loc in range(4):
                i_glob_t = nodes[i_loc]      # T_t DOF
                i_glob_b = nodes[i_loc] + n  # T_b DOF
                
                for j_loc in range(4):
                    j_glob_t = nodes[j_loc]
                    j_glob_b = nodes[j_loc] + n
                    
                    # A_tt block
                    rows.append(i_glob_t)
                    cols.append(j_glob_t)
                    vals.append(A_tt_e[i_loc, j_loc])
                    
                    # A_tb block
                    rows.append(i_glob_t)
                    cols.append(j_glob_b)
                    vals.append(A_tb_e[i_loc, j_loc])
                    
                    # A_bt block
                    rows.append(i_glob_b)
                    cols.append(j_glob_t)
                    vals.append(A_bt_e[i_loc, j_loc])
                    
                    # A_bb block
                    rows.append(i_glob_b)
                    cols.append(j_glob_b)
                    vals.append(A_bb_e[i_loc, j_loc])
                
                # RHS
                b[i_glob_t] += b_t_e[i_loc]
                b[i_glob_b] += b_b_e[i_loc]
        
        # Build sparse matrix
        A = sparse.csr_matrix((vals, (rows, cols)), shape=(2*n, 2*n))
        
        # Apply Dirichlet BC: T_t = T_inlet at inlet nodes
        A, b = self._apply_dirichlet_bc(A, b, self.inlet_nodes, p.T_inlet)
        
        return A, b
    
    def _apply_dirichlet_bc(self, A: sparse.csr_matrix, b: np.ndarray,
                            bc_nodes: np.ndarray, bc_value: float) -> Tuple[sparse.csr_matrix, np.ndarray]:
        """Apply Dirichlet boundary conditions using row elimination"""
        A = A.tolil()  # Convert to LIL for efficient row modification
        
        for node in bc_nodes:
            # Zero out row except diagonal
            A[node, :] = 0
            A[node, node] = 1.0
            b[node] = bc_value
        
        return A.tocsr(), b
    
    def solve(self, inputs: SolverInputs) -> SolverOutputs:
        """
        Main solve routine: assemble and solve the coupled thermal system.
        """
        start_time = time.time()
        p = self.p
        n = p.n_nodes
        
        # Assemble system
        A, b = self.assemble_system(inputs)
        
        # Solve
        T = spsolve(A, b)
        
        # Extract temperature fields
        T_t = T[:n]
        T_b = T[n:]
        
        # Compute objective (p-norm of T_b)
        objective = self.compute_objective(T_b)
        
        # Compute sensitivities
        dJ_dgamma = self.compute_sensitivities(inputs, T_t, T_b, A)
        
        solve_time = time.time() - start_time
        
        return SolverOutputs(
            T_t=T_t,
            T_b=T_b,
            objective=objective,
            dJ_dgamma=dJ_dgamma,
            solve_time=solve_time
        )
    
    def compute_objective(self, T_b: np.ndarray) -> float:
        """
        Compute p-norm approximation to maximum temperature.
        
        J = (1/N * Σ T_b^p)^(1/p)
        """
        p_exp = self.p.p_norm
        N = len(T_b)
        return (np.sum(T_b**p_exp) / N)**(1.0 / p_exp)
    
    def compute_sensitivities(self, inputs: SolverInputs, 
                               T_t: np.ndarray, T_b: np.ndarray,
                               A: sparse.csr_matrix) -> np.ndarray:
        """
        Compute sensitivities dJ/dγ using adjoint method.
        
        1. Compute ∂J/∂T
        2. Solve adjoint: A^T λ = ∂J/∂T
        3. Compute dJ/dγ = -λ^T (∂A/∂γ T - ∂b/∂γ)
        """
        p = self.p
        n = p.n_nodes
        
        # Derivative of objective w.r.t. T_b
        # J = (sum(T_b^p)/N)^(1/p)
        # ∂J/∂T_b = (1/N) * (sum(T_b^p)/N)^(1/p - 1) * T_b^(p-1)
        p_exp = p.p_norm
        N = n
        sum_Tp = np.sum(T_b**p_exp)
        
        dJ_dTb = (1.0 / N) * (sum_Tp / N)**((1.0 - p_exp) / p_exp) * T_b**(p_exp - 1)
        
        # Adjoint RHS: [0, ∂J/∂T_b]
        adjoint_rhs = np.zeros(2 * n)
        adjoint_rhs[n:] = dJ_dTb
        
        # Solve adjoint system: A^T λ = adjoint_rhs
        lam = spsolve(A.T.tocsr(), adjoint_rhs)
        
        # Compute sensitivities for each element
        dJ_dgamma = np.zeros(p.n_elements)
        
        # Get material property derivatives
        gamma_flat = inputs.gamma.flatten()
        
        # dk_t/dγ from RAMP interpolation
        # k_t = k_f + (k_s - k_f) * (1-γ)/(1 + q_k*γ)
        # dk_t/dγ = (k_s - k_f) * [-(1 + q_k*γ) - (1-γ)*q_k] / (1 + q_k*γ)²
        #         = (k_s - k_f) * (-1 - q_k) / (1 + q_k*γ)²
        denom = (1 + p.q_k * gamma_flat)**2
        dk_t_dgamma = (p.k_s - p.k_f) * (-1 - p.q_k) / denom
        
        # Current material properties
        k_t, h = self.compute_material_properties(inputs.gamma)
        
        # dh/dk_t
        h_b = p.k_s / p.H_b
        h_t = 35.0 * k_t / (26.0 * p.H_t)
        dht_dkt = 35.0 / (26.0 * p.H_t)
        dh_dht = h_b**2 / (h_t + h_b)**2
        dh_dgamma = dh_dht * dht_dkt * dk_t_dgamma
        
        # Flatten velocity and temperature
        u_flat = inputs.u.flatten()
        v_flat = inputs.v.flatten()
        T_full = np.concatenate([T_t, T_b])
        
        # Compute element-wise sensitivities
        for e in range(p.n_elements):
            nodes = self.elem_nodes[e]
            
            # Element temperatures and adjoints
            T_t_e = T_t[nodes]
            T_b_e = T_b[nodes]
            lam_t_e = lam[nodes]
            lam_b_e = lam[nodes + n]
            
            # Material property derivatives for this element
            dk_t_e = dk_t_dgamma[e]
            dh_e = dh_dgamma[e]
            h_e = h[e]
            k_t_e = k_t[e]
            
            # Compute ∂A/∂γ * T contribution
            dA_T = 0.0
            
            for g in range(len(self.gauss_pts)):
                N = self.N_gp[g]
                dNdx = self.dNdx_gp[g]
                dNdy = self.dNdy_gp[g]
                w = self.gauss_wts[g] * self.detJ
                
                # Temperature values at Gauss point
                T_t_gp = np.dot(N, T_t_e)
                T_b_gp = np.dot(N, T_b_e)
                grad_T_t = np.array([np.dot(dNdx, T_t_e), np.dot(dNdy, T_t_e)])
                
                # Adjoint values at Gauss point
                lam_t_gp = np.dot(N, lam_t_e)
                lam_b_gp = np.dot(N, lam_b_e)
                grad_lam_t = np.array([np.dot(dNdx, lam_t_e), np.dot(dNdy, lam_t_e)])
                
                # ∂A_tt/∂γ contributions (from k_t and h)
                # Diffusion term: (49/52) * dk_t/dγ * ∇N·∇T_t
                dA_T += (49.0/52.0) * dk_t_e * np.dot(grad_lam_t, grad_T_t) * w
                
                # Coupling terms: dh/dγ * (T_t - T_b) / (2H_t)
                dA_T += (dh_e / (2.0 * p.H_t)) * lam_t_gp * (T_t_gp - T_b_gp) * w
                
                # ∂A_bb/∂γ = 0 (substrate conductivity doesn't depend on γ)
                
                # ∂A_bt/∂γ contributions
                dA_T += (dh_e / (2.0 * p.H_b)) * lam_b_gp * (T_b_gp - T_t_gp) * w
            
            # Sensitivity: dJ/dγ = -λ^T * ∂A/∂γ * T
            dJ_dgamma[e] = -dA_T
        
        return dJ_dgamma
    
    def get_node_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return flattened node coordinates for visualization"""
        return self.X.flatten(), self.Y.flatten()
    
    def reshape_to_grid(self, field: np.ndarray, node_based: bool = True) -> np.ndarray:
        """Reshape a field array to grid format for visualization"""
        if node_based:
            return field.reshape((self.p.M + 1, self.p.N + 1))
        else:
            return field.reshape((self.p.M, self.p.N))


def create_dummy_velocity_field(params: SolverParams, 
                                 mean_velocity: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a simple velocity field for testing.
    Parabolic profile in y, uniform in x (like Poiseuille flow).
    """
    M, N = params.M, params.N
    L_y = params.L_y
    
    # Node coordinates
    y = np.linspace(0, L_y, N + 1)
    
    # Parabolic profile: u = u_max * 4 * y/L * (1 - y/L)
    u_max = 1.5 * mean_velocity  # peak velocity
    u_profile = u_max * 4 * (y / L_y) * (1 - y / L_y)
    
    # Broadcast to full grid
    u = np.tile(u_profile, (M + 1, 1))  # (M+1) x (N+1)
    v = np.zeros((M + 1, N + 1))
    
    return u, v


def create_dummy_geometry(M: int, N: int, pattern: str = 'channels') -> np.ndarray:
    """
    Create dummy heat sink geometries for testing.
    
    Patterns:
    - 'channels': Parallel channels
    - 'pins': Pin fin array
    - 'tree': Tree-like structure
    - 'random': Random distribution
    """
    gamma = np.ones((M, N))  # Start with all fluid
    
    if pattern == 'channels':
        # Horizontal parallel channels
        n_fins = 5
        fin_width = max(1, N // (2 * n_fins))
        for i in range(n_fins):
            y_start = int((2 * i + 1) * N / (2 * n_fins)) - fin_width // 2
            y_end = y_start + fin_width
            gamma[:, max(0, y_start):min(N, y_end)] = 0
            
    elif pattern == 'pins':
        # Pin fin array
        n_pins_x, n_pins_y = 5, 5
        pin_radius = min(M, N) // (3 * max(n_pins_x, n_pins_y))
        
        for i in range(n_pins_x):
            for j in range(n_pins_y):
                cx = int((i + 0.5) * M / n_pins_x)
                cy = int((j + 0.5) * N / n_pins_y)
                
                for di in range(-pin_radius, pin_radius + 1):
                    for dj in range(-pin_radius, pin_radius + 1):
                        if di**2 + dj**2 <= pin_radius**2:
                            ii, jj = cx + di, cy + dj
                            if 0 <= ii < M and 0 <= jj < N:
                                gamma[ii, jj] = 0
                                
    elif pattern == 'tree':
        # Simple tree structure
        # Main trunk
        trunk_width = N // 10
        trunk_y = N // 2
        gamma[:, trunk_y - trunk_width//2:trunk_y + trunk_width//2] = 0
        
        # Branches
        n_branches = 4
        for i in range(n_branches):
            x_pos = int((i + 1) * M / (n_branches + 1))
            # Upper branch
            for dx in range(M // 6):
                y_pos = trunk_y + dx
                if y_pos < N:
                    gamma[min(x_pos + dx, M-1), y_pos] = 0
            # Lower branch
            for dx in range(M // 6):
                y_pos = trunk_y - dx
                if y_pos >= 0:
                    gamma[min(x_pos + dx, M-1), y_pos] = 0
                    
    elif pattern == 'random':
        # Random solid regions (for testing)
        np.random.seed(42)
        gamma = np.random.choice([0.0, 1.0], size=(M, N), p=[0.3, 0.7])
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")
    
    return gamma


def create_dummy_heat_flux(M: int, N: int, pattern: str = 'uniform') -> np.ndarray:
    """
    Create spatially-varying heat flux patterns for testing.
    
    Patterns:
    - 'uniform': Constant heat flux
    - 'hotspot': Single hotspot in center
    - 'multi_hotspot': Multiple hotspots
    - 'gradient': Linear gradient
    """
    q0_base = 6e4  # Base heat flux (W/m²)
    
    if pattern == 'uniform':
        return np.full((M, N), q0_base)
    
    elif pattern == 'hotspot':
        q0 = np.full((M, N), q0_base * 0.5)
        cx, cy = M // 2, N // 2
        radius = min(M, N) // 4
        for i in range(M):
            for j in range(N):
                dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                if dist < radius:
                    q0[i, j] = q0_base * (2.0 - dist / radius)
        return q0
    
    elif pattern == 'multi_hotspot':
        q0 = np.full((M, N), q0_base * 0.3)
        hotspots = [(M//4, N//4), (3*M//4, N//4), (M//2, 3*N//4)]
        radius = min(M, N) // 6
        for cx, cy in hotspots:
            for i in range(M):
                for j in range(N):
                    dist = np.sqrt((i - cx)**2 + (j - cy)**2)
                    if dist < radius:
                        q0[i, j] += q0_base * (1.5 - dist / radius)
        return q0
    
    elif pattern == 'gradient':
        # Heat flux increases from inlet to outlet
        x = np.linspace(0.5, 1.5, M)
        q0 = q0_base * np.outer(x, np.ones(N))
        return q0
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


if __name__ == "__main__":
    # Quick test
    print("Heat Transfer Solver Module")
    print("Run 'python main.py' for full demonstration")
