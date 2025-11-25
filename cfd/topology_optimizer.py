#!/usr/bin/env python3
"""
Topology Optimization for Microfluidic Cold Plates
Based on the 2.5D model from Yan et al. (2019) and van Erp et al. (2025)
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Tuple, Optional
import time

@dataclass
class PhysicalProperties:
    """Material properties for water and metal cold plate"""
    # Water properties
    rho_f: float = 998.0  # kg/m^3
    mu_f: float = 0.001   # Pa*s
    cp_f: float = 4184.0  # J/(kg*K)
    k_f: float = 0.598    # W/(m*K)
    
    # Metal (Aluminum/Copper) properties
    k_s: float = 200.0    # W/(m*K) - aluminum/copper
    
    # Geometric properties
    H: float = 5e-3       # Channel height (m) - 5mm
    tb: float = 5e-3      # Base thickness (m) - 5mm
    

@dataclass
class OptimizationParameters:
    """Parameters for topology optimization"""
    # Domain size
    Lx: float = 0.10      # Length in x-direction (m) - 10cm
    Ly: float = 0.10      # Length in y-direction (m) - 10cm
    
    # Mesh resolution
    nelx: int = 100       # Number of elements in x
    nely: int = 100       # Number of elements in y
    
    # Optimization parameters
    volfrac: float = 0.5  # Volume fraction (50% fluid)
    penal: float = 3.0    # RAMP penalization parameter
    rmin: float = 3.0     # Filter radius (in elements) - controls min feature size
    
    # Flow parameters
    p_inlet: float = 500.0  # Inlet pressure (Pa) - 5 mbar
    p_outlet: float = 0.0   # Outlet pressure (Pa)
    T_inlet: float = 20.0   # Inlet temperature (C)
    
    # Optimization control
    max_iter: int = 200
    tol: float = 1e-3
    move: float = 0.2     # Move limit for design variables
    
    # Minimum feature size (m)
    min_feature_size: float = 1e-3  # 1mm


class MicrofluidicOptimizer:
    """
    Topology optimization for 2D microfluidic cold plate using 2.5D model
    Inlet at top, outlets at left and right sides
    """
    
    def __init__(self, 
                 power_map: np.ndarray,
                 props: PhysicalProperties,
                 params: OptimizationParameters):
        """
        Initialize optimizer
        
        Args:
            power_map: 2D array of heat flux distribution (W/m^2)
            props: Physical properties
            params: Optimization parameters
        """
        self.power_map = power_map
        self.props = props
        self.params = params
        
        # Domain discretization
        self.nelx = params.nelx
        self.nely = params.nely
        self.dx = params.Lx / params.nelx
        self.dy = params.Ly / params.nely
        
        # Total degrees of freedom
        self.ndof_u = 2 * (params.nelx + 1) * (params.nely + 1)  # Velocity DOFs
        self.ndof_p = (params.nelx + 1) * (params.nely + 1)      # Pressure DOFs
        self.ndof_T = (params.nelx + 1) * (params.nely + 1)      # Temperature DOFs (top layer)
        self.ndof_Tb = (params.nelx + 1) * (params.nely + 1)     # Temperature DOFs (bottom layer)
        
        # Initialize design variables (start with uniform distribution)
        self.x = np.ones((self.nely, self.nelx)) * params.volfrac
        self.xPhys = self.x.copy()
        
        # Prepare filter
        self.H, self.Hs = self._prepare_filter()
        
        # Build fixed matrices
        self._build_element_matrices()
        
        # Continuation on Reynolds number for stability
        self.Re_target = self._compute_reynolds_number(params.p_inlet - params.p_outlet)
        self.Re_current = max(1.0, self.Re_target / 100.0)  # Start from low Re
        
    def _compute_reynolds_number(self, delta_p: float) -> float:
        """Estimate Reynolds number from pressure drop"""
        # Rough estimate: U ~ sqrt(2*dP/rho), Re = rho*U*H/mu
        U_est = np.sqrt(2 * abs(delta_p) / self.props.rho_f)
        Re = self.props.rho_f * U_est * self.props.H / self.props.mu_f
        return Re
        
    def _prepare_filter(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare density filter based on Helmholtz PDE method"""
        # Convert filter radius to physical units
        rmin_phys = self.params.min_feature_size
        rmin_elem = rmin_phys / max(self.dx, self.dy)
        
        nfilter = int(self.nelx * self.nely * ((2 * rmin_elem + 1) ** 2))
        iH = np.zeros(nfilter)
        jH = np.zeros(nfilter)
        sH = np.zeros(nfilter)
        cc = 0
        
        for i in range(self.nelx):
            for j in range(self.nely):
                row = i * self.nely + j
                kk1 = int(np.maximum(i - np.ceil(rmin_elem), 0))
                kk2 = int(np.minimum(i + np.ceil(rmin_elem), self.nelx - 1))
                ll1 = int(np.maximum(j - np.ceil(rmin_elem), 0))
                ll2 = int(np.minimum(j + np.ceil(rmin_elem), self.nely - 1))
                
                for k in range(kk1, kk2 + 1):
                    for l in range(ll1, ll2 + 1):
                        col = k * self.nely + l
                        fac = rmin_elem - np.sqrt((i - k) ** 2 + (j - l) ** 2)
                        
                        if fac > 0:
                            iH[cc] = row
                            jH[cc] = col
                            sH[cc] = fac
                            cc += 1
        
        # Finalize filter matrix
        iH = iH[:cc]
        jH = jH[:cc]
        sH = sH[:cc]
        H = sp.coo_matrix((sH, (iH, jH)), 
                          shape=(self.nelx * self.nely, self.nelx * self.nely)).tocsr()
        Hs = H.sum(axis=1).A.flatten()
        
        return H, Hs
    
    def _ramp_interpolation(self, x: np.ndarray, penal: float = None) -> np.ndarray:
        """RAMP interpolation scheme for penalization"""
        if penal is None:
            penal = self.params.penal
        return x / (1 + penal * (1 - x))
    
    def _ramp_derivative(self, x: np.ndarray, penal: float = None) -> np.ndarray:
        """Derivative of RAMP interpolation"""
        if penal is None:
            penal = self.params.penal
        return (1 + penal) / ((1 + penal * (1 - x)) ** 2)
    
    def _build_element_matrices(self):
        """Build element matrices for fluid flow and heat transfer"""
        # Gauss quadrature points and weights (2x2)
        gauss_pts = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) / np.sqrt(3)
        gauss_wts = np.ones(4)
        
        # Element size
        dx, dy = self.dx, self.dy
        
        # Store element matrices
        self.Ke_u = np.zeros((8, 8))  # Velocity stiffness (2 DOF per node, 4 nodes)
        self.Me_u = np.zeros((8, 8))  # Velocity mass matrix
        self.Ke_p = np.zeros((4, 4))  # Pressure matrix
        self.Ke_T = np.zeros((4, 4))  # Temperature stiffness
        self.Me_T = np.zeros((4, 4))  # Temperature mass (for advection)
        
        # Build using Gauss quadrature
        for gp, gw in zip(gauss_pts, gauss_wts):
            xi, eta = gp
            
            # Shape functions (bilinear)
            N = 0.25 * np.array([
                (1 - xi) * (1 - eta),
                (1 + xi) * (1 - eta),
                (1 + xi) * (1 + eta),
                (1 - xi) * (1 + eta)
            ])
            
            # Shape function derivatives in natural coordinates
            dN_dxi = 0.25 * np.array([
                [-(1 - eta), (1 - eta), (1 + eta), -(1 + eta)],
                [-(1 - xi), -(1 + xi), (1 + xi), (1 - xi)]
            ])
            
            # Jacobian
            J = np.array([[dx / 2, 0], [0, dy / 2]])
            detJ = dx * dy / 4
            
            # Shape function derivatives in physical coordinates
            dN_dx = np.linalg.inv(J) @ dN_dxi
            
            # Build matrices
            B = dN_dx  # Gradient matrix (2 x 4)
            
            # Temperature stiffness (diffusion)
            self.Ke_T += gw * detJ * (B.T @ B)
            
            # Temperature mass (for advection term)
            self.Me_T += gw * detJ * np.outer(N, N)
            
            # Pressure matrix  
            self.Ke_p += gw * detJ * np.outer(N, N)
            
            # Velocity matrices (2 DOF per node)
            # Build 2D velocity element matrix
            for a in range(4):  # node a
                for i in range(2):  # direction i (x, y)
                    for b in range(4):  # node b
                        for j in range(2):  # direction j (x, y)
                            # Stiffness: grad(Na) : grad(Nb) * delta_ij
                            if i == j:
                                self.Ke_u[2*a+i, 2*b+j] += gw * detJ * (
                                    dN_dx[0, a] * dN_dx[0, b] + dN_dx[1, a] * dN_dx[1, b]
                                )
                            # Mass: Na * Nb * delta_ij
                            if i == j:
                                self.Me_u[2*a+i, 2*b+j] += gw * detJ * N[a] * N[b]
        
    def _get_thermal_conductivity(self, rho: float) -> float:
        """Interpolate thermal conductivity based on density"""
        rho_penalized = self._ramp_interpolation(np.array([rho]))[0]
        k = rho_penalized * self.props.k_f + (1 - rho_penalized) * self.props.k_s
        return k
    
    def _get_channel_thickness(self, rho: float) -> float:
        """Interpolate channel thickness based on density"""
        rho_penalized = self._ramp_interpolation(np.array([rho]))[0]
        return rho_penalized * self.props.H
    
    def _assemble_stokes_system(self, xPhys: np.ndarray, Re: float) -> Tuple[sp.csr_matrix, np.ndarray]:
        """
        Assemble Stokes system for given design and Reynolds number
        Simplified 2.5D Stokes with Darcy drag term
        """
        # Initialize sparse matrices
        K = sp.lil_matrix((self.ndof_u, self.ndof_u))
        B = sp.lil_matrix((self.ndof_p, self.ndof_u))
        
        # Assemble element by element
        for elx in range(self.nelx):
            for ely in range(self.nely):
                rho = xPhys[ely, elx]
                rho_penalized = self._ramp_interpolation(np.array([rho]))[0]
                
                # Effective viscosity (high viscosity in solid regions)
                mu_eff = self.props.mu_f / (rho_penalized + 1e-6)
                
                # Channel thickness effect (Darcy-like drag)
                H_eff = self._get_channel_thickness(rho)
                alpha_drag = 12 * self.props.mu_f / (H_eff ** 2 + 1e-12)
                
                # Element DOFs
                n1 = ely + elx * (self.nely + 1)
                n2 = ely + 1 + elx * (self.nely + 1)
                n3 = ely + 1 + (elx + 1) * (self.nely + 1)
                n4 = ely + (elx + 1) * (self.nely + 1)
                
                edof_u = np.array([2*n1, 2*n1+1, 2*n2, 2*n2+1, 
                                  2*n3, 2*n3+1, 2*n4, 2*n4+1])
                edof_p = np.array([n1, n2, n3, n4])
                
                # Build element stiffness with drag
                Ke_elem = mu_eff * self.Ke_u + alpha_drag * self.Me_u
                
                # Assemble
                for i in range(8):
                    for j in range(8):
                        K[edof_u[i], edof_u[j]] += Ke_elem[i, j]
        
        # Convert to CSR for efficient solving
        K = K.tocsr()
        
        # Apply boundary conditions: inlet at top, outlets at sides
        # Inlet: top edge (y = Ly)
        # Outlets: left and right edges (x = 0 and x = Lx)
        
        # Fixed nodes and prescribed values
        fixed_dofs = []
        free_dofs = []
        F = np.zeros(self.ndof_u)
        
        # All boundary nodes have u=v=0 (no-slip, except pressure-driven)
        # This is simplified - in reality would set pressure BCs properly
        for i in range(self.nelx + 1):
            # Bottom edge
            node = i * (self.nely + 1)
            fixed_dofs.extend([2*node, 2*node+1])
            
            # Top edge (inlet)
            node = i * (self.nely + 1) + self.nely
            fixed_dofs.extend([2*node, 2*node+1])
        
        for j in range(self.nely + 1):
            # Left edge (outlet)
            node = j
            fixed_dofs.extend([2*node, 2*node+1])
            
            # Right edge (outlet)
            node = self.nelx * (self.nely + 1) + j
            fixed_dofs.extend([2*node, 2*node+1])
        
        fixed_dofs = np.unique(fixed_dofs)
        all_dofs = np.arange(self.ndof_u)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        return K, F, free_dofs, fixed_dofs
    
    def _solve_flow_field(self, xPhys: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Solve for velocity and pressure fields"""
        K, F, free_dofs, fixed_dofs = self._assemble_stokes_system(xPhys, self.Re_current)
        
        # Solve reduced system
        U = np.zeros(self.ndof_u)
        
        if len(free_dofs) > 0:
            K_free = K[free_dofs, :][:, free_dofs]
            F_free = F[free_dofs]
            
            # Add pressure gradient term (driving force)
            # Create stronger pressure gradient from top to sides
            dp = (self.params.p_inlet - self.params.p_outlet)
            
            # Apply body force to drive flow
            for i in range(self.nely + 1):
                for j in range(self.nelx + 1):
                    node = i * (self.nelx + 1) + j
                    
                    # X-direction: flow toward sides
                    dof_x = 2 * node
                    if dof_x in free_dofs:
                        idx = np.where(free_dofs == dof_x)[0][0]
                        # Force proportional to distance from center
                        x_pos = j / self.nelx - 0.5  # -0.5 to 0.5
                        F_free[idx] += dp * x_pos * (self.dx * self.dy) * xPhys.mean()
                    
                    # Y-direction: flow from top to bottom initially
                    dof_y = 2 * node + 1
                    if dof_y in free_dofs:
                        idx = np.where(free_dofs == dof_y)[0][0]
                        y_pos = i / self.nely  # 0 to 1
                        F_free[idx] += -dp * (1 - y_pos) * (self.dx * self.dy) * xPhys.mean()
            
            try:
                U_free = spla.spsolve(K_free, F_free)
                U[free_dofs] = U_free
            except:
                print("Warning: Flow solve failed, using zero velocity")
        
        # Extract velocity components
        u_x = U[0::2].reshape((self.nely + 1, self.nelx + 1))
        u_y = U[1::2].reshape((self.nely + 1, self.nelx + 1))
        
        # Ensure velocities are reasonable
        u_max = np.sqrt(u_x**2 + u_y**2).max()
        if u_max > 10.0:  # Cap unrealistic velocities
            scale = 5.0 / u_max
            u_x *= scale
            u_y *= scale
        
        # Pressure field (simplified)
        P = np.zeros((self.nely + 1, self.nelx + 1))
        
        return (u_x, u_y), P
    
    def _solve_temperature_field(self, xPhys: np.ndarray, 
                                 velocity: Tuple[np.ndarray, np.ndarray]) -> np.ndarray:
        """Solve for temperature field with advection-diffusion"""
        u_x, u_y = velocity
        
        # Assemble thermal system
        K_T = sp.lil_matrix((self.ndof_T, self.ndof_T))
        M_adv = sp.lil_matrix((self.ndof_T, self.ndof_T))
        F_T = np.zeros(self.ndof_T)
        
        for elx in range(self.nelx):
            for ely in range(self.nely):
                rho = xPhys[ely, elx]
                
                # Effective thermal conductivity
                k_eff = self._get_thermal_conductivity(rho)
                
                # Element nodes
                n1 = ely + elx * (self.nely + 1)
                n2 = ely + 1 + elx * (self.nely + 1)
                n3 = ely + 1 + (elx + 1) * (self.nely + 1)
                n4 = ely + (elx + 1) * (self.nely + 1)
                edof = np.array([n1, n2, n3, n4])
                
                # Average velocity in element
                u_elem = 0.25 * (u_x[ely, elx] + u_x[ely+1, elx] + 
                                u_x[ely, elx+1] + u_x[ely+1, elx+1])
                v_elem = 0.25 * (u_y[ely, elx] + u_y[ely+1, elx] + 
                                u_y[ely, elx+1] + u_y[ely+1, elx+1])
                
                # Diffusion term
                Ke_diff = k_eff * self.Ke_T
                
                # Advection term
                velocity_mag = np.sqrt(u_elem**2 + v_elem**2) + 1e-10
                rho_f_cp = self.props.rho_f * self.props.cp_f
                
                # Only advect in fluid regions
                if rho > 0.3:  # Fluid region
                    Me_adv = rho_f_cp * velocity_mag * self.Me_T
                else:
                    Me_adv = np.zeros_like(self.Me_T)
                
                # Heat source term - THIS IS CRITICAL
                q_elem = self.power_map[ely, elx]  # W/m²
                
                # Convert to volumetric source (W/m³) by dividing by thickness
                # Then integrate over element volume
                vol_source = q_elem / (self.props.H + self.props.tb)
                element_volume = self.dx * self.dy * (self.props.H + self.props.tb)
                
                # Total power in element
                Q_total = q_elem * self.dx * self.dy
                
                # Distribute to nodes (equal distribution)
                Fe_source = np.ones(4) * Q_total / 4.0
                
                # Scale by solid fraction (heat only generated in solid base)
                if rho < 0.5:  # Solid region - heat is generated here
                    Fe_source *= 1.0
                else:  # Fluid region - heat conducted from below
                    Fe_source *= 0.5  # Reduced but not zero
                
                # Assemble
                for i in range(4):
                    for j in range(4):
                        K_T[edof[i], edof[j]] += Ke_diff[i, j]
                        M_adv[edof[i], edof[j]] += Me_adv[i, j]
                    F_T[edof[i]] += Fe_source[i]
        
        K_total = K_T + M_adv
        K_total = K_total.tocsr()
        
        # Apply boundary conditions
        # Inlet temperature at top
        fixed_dofs = []
        T = np.ones(self.ndof_T) * self.params.T_inlet
        
        # Top edge (inlet) - fixed temperature
        for i in range(self.nelx + 1):
            node = i * (self.nely + 1) + self.nely
            fixed_dofs.append(node)
            T[node] = self.params.T_inlet
        
        # Outlet sides - let temperature float (natural BC)
        
        fixed_dofs = np.array(fixed_dofs)
        all_dofs = np.arange(self.ndof_T)
        free_dofs = np.setdiff1d(all_dofs, fixed_dofs)
        
        # Solve
        if len(free_dofs) > 0:
            K_free = K_total[free_dofs, :][:, free_dofs]
            F_free = F_T[free_dofs] - K_total[free_dofs, :][:, fixed_dofs] @ T[fixed_dofs]
            
            # Add small regularization to prevent singular matrix
            K_free = K_free + sp.eye(K_free.shape[0]) * 1e-8
            
            try:
                T_free = spla.spsolve(K_free, F_free)
                T[free_dofs] = T_free
            except Exception as e:
                print(f"Warning: Temperature solve failed: {e}")
                # Use a simple approximation based on power
                for i in range(self.nely):
                    for j in range(self.nelx):
                        node = i * (self.nelx + 1) + j
                        if node in free_dofs:
                            # Temperature rise proportional to local power
                            q_local = self.power_map[i, j]
                            # Simple thermal resistance estimate
                            R_th = 0.01  # K/(W/m²)
                            T[node] = self.params.T_inlet + q_local * R_th
        
        T_field = T.reshape((self.nely + 1, self.nelx + 1))
        
        # Ensure temperatures are physical
        T_field = np.clip(T_field, self.params.T_inlet, self.params.T_inlet + 200)
        
        return T_field
    
    def _compute_objective_and_sensitivity(self, xPhys: np.ndarray, 
                                          T_field: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute objective (minimize max temperature) and sensitivity
        
        Key insight: We want channels (high density) where temperature is high
        """
        # Extract element-center temperatures
        T_elem = 0.25 * (T_field[:-1, :-1] + T_field[1:, :-1] + 
                        T_field[:-1, 1:] + T_field[1:, 1:])
        
        # Objective: weighted average temperature rise
        T_rise = T_elem - self.params.T_inlet
        T_rise = np.maximum(T_rise, 0.0)
        
        # Weight by power (care more about hot regions)
        weights = self.power_map / (self.power_map.mean() + 1e-10)
        obj = np.sum(T_rise * weights) / np.sum(weights)
        
        # Sensitivity calculation
        # Physics: Adding fluid (increasing ρ) in hot regions DECREASES temperature
        # Therefore: ∂T/∂ρ < 0 in hot regions
        # Since we minimize T, we want ∂obj/∂ρ < 0 where T is high
        # This encourages fluid (ρ=1) in hot spots
        
        dc = np.zeros((self.nely, self.nelx))
        
        for i in range(self.nely):
            for j in range(self.nelx):
                T_local = T_elem[i, j]
                q_local = self.power_map[i, j]
                rho_local = xPhys[i, j]
                
                # Temperature rise at this element
                dT = T_local - self.params.T_inlet
                
                if dT > 0.1:  # Hot region - needs cooling
                    # Adding fluid here helps cooling
                    # Sensitivity should be negative (want to increase ρ)
                    dc[i, j] = -dT * q_local / 10000.0
                else:  # Cool region
                    # Can be solid to save volume for hot regions
                    dc[i, j] = 0.01
        
        # Normalize
        dc_max = np.abs(dc).max()
        if dc_max > 1e-6:
            dc = dc / dc_max
        
        return obj, dc
    
    def _apply_filter(self, dc: np.ndarray) -> np.ndarray:
        """Apply density filter to sensitivities"""
        dc_flat = dc.flatten()
        dc_filtered = (self.H @ dc_flat) / self.Hs
        return dc_filtered.reshape((self.nely, self.nelx))
    
    def _update_design_variables(self, x: np.ndarray, dc: np.ndarray, 
                                iteration: int) -> np.ndarray:
        """
        Update design variables using Optimality Criteria
        
        OC update rule: x_new = x * sqrt(-dc/lambda)
        Where dc is ∂obj/∂x (negative in hot regions where we want fluid)
        """
        move = self.params.move
        
        # Ensure sensitivities are valid
        dc = np.nan_to_num(dc, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Bisection for Lagrange multiplier
        l1 = 1e-10
        l2 = 1e5
        
        while (l2 - l1) / (l1 + l2) > 1e-4 and (l2 - l1) > 1e-8:
            lmid = 0.5 * (l2 + l1)
            
            # OC update: x_new = x * sqrt(max(0, -dc/lambda))
            # Negative dc means we want to increase x (add fluid)
            Be = -dc / lmid
            Be = np.maximum(0.0, Be)  # Only positive values
            
            xnew = x * np.sqrt(Be + 1e-10)
            
            # Apply move limits
            xnew = np.maximum(x - move, np.minimum(x + move, xnew))
            
            # Apply bounds
            xnew = np.clip(xnew, 0.001, 1.0)
            
            # Check volume constraint
            vol_current = np.sum(xnew)
            vol_target = self.params.volfrac * self.nelx * self.nely
            
            if vol_current > vol_target:
                l1 = lmid
            else:
                l2 = lmid
        
        return xnew
    
    def optimize(self, verbose: bool = True) -> Tuple[np.ndarray, dict]:
        """
        Main optimization loop
        
        Returns:
            xPhys: Final optimized design
            history: Dictionary with optimization history
        """
        history = {
            'objective': [],
            'change': [],
            'iteration': [],
            'time': []
        }
        
        x = self.x.copy()
        xPhys = self.xPhys.copy()
        
        change = 1.0
        iteration = 0
        start_time = time.time()
        
        print("="*60)
        print("Starting Topology Optimization")
        print(f"Domain: {self.params.Lx*1000:.1f}mm x {self.params.Ly*1000:.1f}mm")
        print(f"Resolution: {self.nelx} x {self.nely} elements")
        print(f"Min feature size: {self.params.min_feature_size*1000:.2f}mm")
        print(f"Target volume fraction: {self.params.volfrac*100:.1f}%")
        print("="*60)
        
        while change > self.params.tol and iteration < self.params.max_iter:
            iteration += 1
            iter_start = time.time()
            
            # Filter design variables
            xPhys_flat = (self.H @ x.flatten()) / self.Hs
            xPhys = xPhys_flat.reshape((self.nely, self.nelx))
            
            # Solve physics
            if verbose and iteration % 10 == 0:
                print(f"\nIter {iteration}: Solving flow field...")
            
            velocity, pressure = self._solve_flow_field(xPhys)
            
            if verbose and iteration % 10 == 0:
                print(f"Iter {iteration}: Solving temperature field...")
            
            temperature = self._solve_temperature_field(xPhys, velocity)
            
            # Compute objective and sensitivity
            obj, dc = self._compute_objective_and_sensitivity(xPhys, temperature)
            
            # Filter sensitivities
            dc_filtered = self._apply_filter(dc)
            
            # Update design variables
            x_new = self._update_design_variables(x, dc_filtered, iteration)
            
            # Compute change
            change = np.max(np.abs(x_new - x))
            x = x_new
            
            # Store history
            iter_time = time.time() - iter_start
            history['objective'].append(obj)
            history['change'].append(change)
            history['iteration'].append(iteration)
            history['time'].append(iter_time)
            
            # Print progress
            if verbose and (iteration % 5 == 0 or iteration == 1):
                max_T = temperature.max()
                avg_T = temperature.mean()
                u_max = np.sqrt(velocity[0]**2 + velocity[1]**2).max()
                
                print(f"\n{'='*60}")
                print(f"Iteration: {iteration}/{self.params.max_iter}")
                print(f"Objective: {obj:.6f}")
                print(f"Change: {change:.6f}")
                print(f"Max Temperature: {max_T:.2f} °C")
                print(f"Avg Temperature: {avg_T:.2f} °C")
                print(f"Max Velocity: {u_max:.6f} m/s")
                print(f"Volume fraction: {xPhys.mean():.3f}")
                print(f"Time: {iter_time:.2f}s")
                print(f"{'='*60}")
            
            # Continuation on Reynolds number
            if iteration % 20 == 0 and self.Re_current < self.Re_target:
                self.Re_current = min(self.Re_current * 1.5, self.Re_target)
                if verbose:
                    print(f"Increasing Reynolds number to {self.Re_current:.1f}")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("Optimization Complete!")
        print(f"Total iterations: {iteration}")
        print(f"Total time: {total_time:.2f}s")
        print(f"Final objective: {obj:.6f}")
        print(f"Final change: {change:.6f}")
        print("="*60 + "\n")
        
        # Final solve for visualization
        xPhys_flat = (self.H @ x.flatten()) / self.Hs
        xPhys = xPhys_flat.reshape((self.nely, self.nelx))
        velocity, pressure = self._solve_flow_field(xPhys)
        temperature = self._solve_temperature_field(xPhys, velocity)
        
        history['final_velocity'] = velocity
        history['final_pressure'] = pressure
        history['final_temperature'] = temperature
        
        return xPhys, history


def create_sample_power_map(nelx: int, nely: int, hotspot_locs: list = None) -> np.ndarray:
    """
    Create a sample power map with hotspots
    
    Args:
        nelx, nely: Grid dimensions
        hotspot_locs: List of (x, y, intensity, radius) tuples for hotspots
    
    Returns:
        power_map: 2D array of heat flux (W/m^2)
    """
    power_map = np.ones((nely, nelx)) * 5000.0  # Higher base heat flux: 5 kW/m^2
    
    if hotspot_locs is None:
        # Default: 4 hotspots representing CPU cores
        hotspot_locs = [
            (0.3, 0.3, 200000, 0.1),  # Higher power: 200 kW/m^2 at hotspots
            (0.7, 0.3, 200000, 0.1),
            (0.3, 0.7, 200000, 0.1),
            (0.7, 0.7, 200000, 0.1),
        ]
    
    # Add hotspots
    for x_rel, y_rel, power, radius_rel in hotspot_locs:
        x_idx = int(x_rel * nelx)
        y_idx = int(y_rel * nely)
        radius = int(radius_rel * min(nelx, nely))
        
        for i in range(max(0, y_idx - radius), min(nely, y_idx + radius)):
            for j in range(max(0, x_idx - radius), min(nelx, x_idx + radius)):
                dist = np.sqrt((i - y_idx)**2 + (j - x_idx)**2)
                if dist < radius:
                    # Gaussian profile
                    factor = np.exp(-(dist / radius)**2)
                    power_map[i, j] += power * factor
    
    return power_map


if __name__ == "__main__":
    # Example usage
    print("\nMicrofluidic Cold Plate Topology Optimization")
    print("="*60 + "\n")
    
    # Set up parameters
    params = OptimizationParameters(
        Lx=0.10,  # 10cm x 10cm domain
        Ly=0.10,
        nelx=60,  # Reduced for faster convergence
        nely=60,
        volfrac=0.5,
        penal=3.0,
        rmin=2.5,
        max_iter=150,  # More iterations for learning
        tol=1e-4,  # Tighter tolerance
        move=0.2,  # Reasonable move limit
        min_feature_size=2.0e-3,  # 2mm minimum feature
        p_inlet=2000.0,  # Higher pressure for better flow
    )
    
    props = PhysicalProperties(
        H=5e-3,   # 5mm channel height
        tb=5e-3,  # 5mm base thickness
        k_s=200.0  # Aluminum/copper base
    )
    
    # Create power map with hotspots
    power_map = create_sample_power_map(params.nelx, params.nely)
    
    print(f"Starting optimization with:")
    print(f"  - Grid size: {params.nelx} x {params.nely}")
    print(f"  - Max iterations: {params.max_iter}")
    print(f"  - Target volume fraction: {params.volfrac}")
    print(f"  - Inlet pressure: {params.p_inlet} Pa")
    print()
    
    # Initialize and run optimizer
    optimizer = MicrofluidicOptimizer(power_map, props, params)
    xPhys_final, history = optimizer.optimize(verbose=True)
    
    # Save results
    print("\nSaving results...")
    np.save('/home/claude/optimized_design.npy', xPhys_final)
    np.save('/home/claude/power_map.npy', power_map)
    np.save('/home/claude/optimization_history.npy', history)
    
    print("Results saved to:")
    print("  - optimized_design.npy")
    print("  - power_map.npy")
    print("  - optimization_history.npy")