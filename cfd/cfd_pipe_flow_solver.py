"""
2D Axisymmetric CFD Solver for Pipe Flow
=========================================
Solves the 2D axisymmetric incompressible Navier-Stokes equations using:
- Finite Difference Method in cylindrical coordinates (r, z)
- Pressure-velocity coupling via fractional step method
- Pipe flow with various inlet/outlet conditions

Governing Equations (axisymmetric, incompressible):
- Continuity: (1/r)∂(r*u_r)/∂r + ∂u_z/∂z = 0
- Momentum (r): ∂u_r/∂t + u_r*∂u_r/∂r + u_z*∂u_r/∂z = -1/ρ*∂p/∂r + ν(∇²u_r - u_r/r²)
- Momentum (z): ∂u_z/∂t + u_r*∂u_z/∂r + u_z*∂u_z/∂z = -1/ρ*∂p/∂z + ν*∇²u_z

Where: ∇² = ∂²/∂r² + (1/r)*∂/∂r + ∂²/∂z²

Author: CFD Solver
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class PipeFlowSolver:
    """
    2D Axisymmetric CFD Solver for pipe flow.
    
    Coordinate system:
    - r: radial direction (0 to R)
    - z: axial direction (0 to L)
    - u_r: radial velocity component
    - u_z: axial velocity component
    """
    
    def __init__(self, nr=41, nz=101, radius=0.05, length=0.5, rho=1000.0, mu=0.001, dt=None):
        """
        Initialize the pipe flow solver.
        
        Parameters:
        -----------
        nr : int
            Number of grid points in radial direction
        nz : int
            Number of grid points in axial direction
        radius : float
            Pipe radius [m]
        length : float
            Pipe length [m]
        rho : float
            Fluid density [kg/m³]
        mu : float
            Dynamic viscosity [Pa·s]
        dt : float
            Time step [s] (auto-calculated if None)
        """
        self.nr = nr
        self.nz = nz
        self.radius = radius
        self.length = length
        self.rho = rho
        self.mu = mu
        self.nu = mu / rho  # Kinematic viscosity
        
        # Grid spacing
        self.dr = radius / (nr - 1)
        self.dz = length / (nz - 1)
        
        # Create mesh - r starts at 0 (centerline) and goes to R (wall)
        # Use cell-centered approach near r=0 to avoid singularity
        self.r = np.linspace(0, radius, nr)
        self.r[0] = self.dr / 4  # Small offset to avoid division by zero at r=0
        self.z = np.linspace(0, length, nz)
        self.R, self.Z = np.meshgrid(self.r, self.z, indexing='ij')
        
        # Initialize velocity and pressure fields
        # Shape: (nr, nz) where first index is r, second is z
        self.u_r = np.zeros((nr, nz))  # Radial velocity
        self.u_z = np.zeros((nr, nz))  # Axial velocity
        self.p = np.zeros((nr, nz))    # Pressure
        
        # Source term for pressure Poisson equation
        self.b = np.zeros((nr, nz))
        
        # Flow parameters (set by setup methods)
        self.u_inlet = 0
        self.dp_dz = 0
        self.Re = None
        
        # Auto-calculate time step if not provided
        if dt is None:
            self.dt = self._calculate_stable_dt()
        else:
            self.dt = dt
            
    def _calculate_stable_dt(self, safety=0.25):
        """
        Calculate stable time step based on CFL and diffusion constraints.
        """
        # Assume max velocity ~ 1 m/s for initial estimate
        u_max = 1.0
        
        # Convective constraint
        dt_conv = min(self.dr, self.dz) / u_max
        
        # Diffusive constraint
        dt_diff = min(self.dr, self.dz)**2 / (4 * self.nu)
        
        dt = safety * min(dt_conv, dt_diff)
        return dt
    
    def set_pressure_driven_flow(self, dp_dz=-100.0):
        """
        Set up pressure-driven (Poiseuille) pipe flow.
        
        Parameters:
        -----------
        dp_dz : float
            Pressure gradient in axial direction [Pa/m] (negative for flow in +z)
        """
        self.dp_dz = dp_dz
        self.flow_type = 'pressure_driven'
        
        # Analytical solution for fully developed flow:
        # u_z(r) = -(dp/dz) * R² / (4μ) * (1 - (r/R)²)
        # u_max = -(dp/dz) * R² / (4μ)
        u_max = -dp_dz * self.radius**2 / (4 * self.mu)
        u_avg = u_max / 2  # For parabolic profile
        
        self.Re = self.rho * u_avg * (2 * self.radius) / self.mu
        
        print(f"Pressure-driven pipe flow setup:")
        print(f"  Pressure gradient: {dp_dz:.2f} Pa/m")
        print(f"  Expected u_max: {u_max:.4f} m/s")
        print(f"  Expected u_avg: {u_avg:.4f} m/s")
        print(f"  Reynolds number: {self.Re:.1f}")
        
    def set_velocity_driven_flow(self, u_inlet=0.1):
        """
        Set up velocity-driven flow with specified inlet velocity.
        
        Parameters:
        -----------
        u_inlet : float
            Uniform inlet velocity [m/s]
        """
        self.u_inlet = u_inlet
        self.flow_type = 'velocity_driven'
        
        self.Re = self.rho * u_inlet * (2 * self.radius) / self.mu
        
        print(f"Velocity-driven pipe flow setup:")
        print(f"  Inlet velocity: {u_inlet:.4f} m/s")
        print(f"  Reynolds number: {self.Re:.1f}")
        
    def set_parabolic_inlet(self, u_max=0.1):
        """
        Set up flow with fully-developed parabolic inlet profile.
        
        Parameters:
        -----------
        u_max : float
            Maximum centerline velocity [m/s]
        """
        self.u_max_inlet = u_max
        self.flow_type = 'parabolic_inlet'
        
        u_avg = u_max / 2
        self.Re = self.rho * u_avg * (2 * self.radius) / self.mu
        
        print(f"Parabolic inlet pipe flow setup:")
        print(f"  Max centerline velocity: {u_max:.4f} m/s")
        print(f"  Average velocity: {u_avg:.4f} m/s")
        print(f"  Reynolds number: {self.Re:.1f}")
        
    def apply_boundary_conditions(self):
        """
        Apply boundary conditions for pipe flow.
        
        Boundaries:
        - r = 0 (centerline, index 0): Symmetry (∂u_z/∂r = 0, u_r = 0)
        - r = R (wall, index -1): No-slip (u_z = 0, u_r = 0)
        - z = 0 (inlet): Specified velocity profile
        - z = L (outlet): Zero-gradient (Neumann)
        """
        # Centerline symmetry (r ≈ 0, index 0)
        # u_r = 0 at centerline (no flow through axis)
        # ∂u_z/∂r = 0 at centerline (symmetry) -> u_z[0] = u_z[1]
        self.u_r[0, :] = 0
        self.u_z[0, :] = self.u_z[1, :]  # Symmetry: zero gradient
        
        # Wall no-slip (r = R, index -1)
        self.u_r[-1, :] = 0
        self.u_z[-1, :] = 0
        
        # Inlet boundary (z = 0, index 0)
        if self.flow_type == 'pressure_driven':
            # Parabolic inlet profile: u_z(r) = u_max * (1 - (r/R)²)
            u_max = -self.dp_dz * self.radius**2 / (4 * self.mu)
            self.u_z[:, 0] = u_max * (1 - (self.r / self.radius)**2)
            self.u_r[:, 0] = 0
        elif self.flow_type == 'velocity_driven':
            self.u_z[:, 0] = self.u_inlet
            self.u_r[:, 0] = 0
        elif self.flow_type == 'parabolic_inlet':
            self.u_z[:, 0] = self.u_max_inlet * (1 - (self.r / self.radius)**2)
            self.u_r[:, 0] = 0
        
        # Enforce wall condition at inlet too
        self.u_z[-1, 0] = 0
            
        # Outlet boundary (z = L, index -1) - zero gradient
        self.u_z[:, -1] = self.u_z[:, -2]
        self.u_r[:, -1] = self.u_r[:, -2]
        
        # Enforce wall at outlet
        self.u_z[-1, -1] = 0
        self.u_r[-1, -1] = 0
        
    def initialize_flow_field(self):
        """
        Initialize the flow field with a reasonable starting guess.
        This helps convergence significantly.
        """
        if self.flow_type == 'pressure_driven':
            # Initialize with analytical Poiseuille profile
            # u_z(r) = u_max * (1 - (r/R)²)
            u_max = -self.dp_dz * self.radius**2 / (4 * self.mu)
            for j in range(self.nz):
                self.u_z[:, j] = u_max * (1 - (self.r / self.radius)**2)
            # Ensure wall is zero
            self.u_z[-1, :] = 0
        elif self.flow_type == 'velocity_driven':
            # Initialize with uniform flow
            self.u_z[:, :] = self.u_inlet
            self.u_z[-1, :] = 0  # Wall
        elif self.flow_type == 'parabolic_inlet':
            # Initialize with parabolic profile
            for j in range(self.nz):
                self.u_z[:, j] = self.u_max_inlet * (1 - (self.r / self.radius)**2)
            self.u_z[-1, :] = 0
                
        # Radial velocity starts at zero
        self.u_r[:, :] = 0
        
        # Initialize pressure with linear drop for pressure-driven flow
        if self.flow_type == 'pressure_driven':
            for j in range(self.nz):
                self.p[:, j] = -self.dp_dz * (self.length - self.z[j])
        
    def build_pressure_rhs(self):
        """
        Build the right-hand side of the pressure Poisson equation.
        Vectorized implementation.
        
        In cylindrical coordinates, the divergence is:
        ∇·u = (1/r)∂(r*u_r)/∂r + ∂u_z/∂z
        """
        dr, dz, dt, rho = self.dr, self.dz, self.dt, self.rho
        u_r, u_z = self.u_r, self.u_z
        r = self.r
        
        # Create r array for broadcasting (nr-2, 1) for interior points
        r_2d = r[1:-1, np.newaxis]
        r_plus = r[2:, np.newaxis]
        r_minus = r[:-2, np.newaxis]
        
        # Divergence in cylindrical coordinates (vectorized)
        # (1/r) * d(r*u_r)/dr + du_z/dz
        div_u = (
            (r_plus * u_r[2:, 1:-1] - r_minus * u_r[:-2, 1:-1]) / (2 * dr * r_2d) +
            (u_z[1:-1, 2:] - u_z[1:-1, :-2]) / (2 * dz)
        )
        
        self.b[1:-1, 1:-1] = rho * div_u / dt
                
    def solve_pressure_poisson(self, nit=100, tol=1e-6):
        """
        Solve the pressure Poisson equation in cylindrical coordinates.
        Vectorized Jacobi iteration for better performance.
        
        ∇²p = (1/r)∂/∂r(r*∂p/∂r) + ∂²p/∂z² = b
        """
        dr, dz = self.dr, self.dz
        r = self.r
        
        # Precompute coefficients for vectorized operations
        r_2d = r[1:-1, np.newaxis]
        r_plus = r_2d + dr/2
        r_minus = r_2d - dr/2
        
        # Coefficients
        a_e = r_plus / (r_2d * dr**2)
        a_w = r_minus / (r_2d * dr**2)
        a_n = 1.0 / dz**2
        a_s = 1.0 / dz**2
        a_p = a_e + a_w + a_n + a_s
        
        for iteration in range(nit):
            p_old = self.p.copy()
            
            # Vectorized update
            self.p[1:-1, 1:-1] = (
                a_e * p_old[2:, 1:-1] +
                a_w * p_old[:-2, 1:-1] +
                a_n * p_old[1:-1, 2:] +
                a_s * p_old[1:-1, :-2] -
                self.b[1:-1, 1:-1]
            ) / a_p
            
            # Pressure boundary conditions
            # Centerline: ∂p/∂r = 0
            self.p[0, :] = self.p[1, :]
            
            # Wall: ∂p/∂r = 0
            self.p[-1, :] = self.p[-2, :]
            
            # Inlet: Reference pressure
            self.p[:, 0] = self.p[:, 1]
            
            # Outlet: Zero gradient
            self.p[:, -1] = self.p[:, -2]
            
            # Check convergence
            if np.max(np.abs(self.p - p_old)) < tol:
                break
                
    def compute_velocity(self):
        """
        Update velocity field using the momentum equations in cylindrical coordinates.
        Vectorized implementation for better performance.
        
        Momentum equations:
        ∂u_r/∂t = -u_r*∂u_r/∂r - u_z*∂u_r/∂z - (1/ρ)*∂p/∂r + ν*(∇²u_r - u_r/r²)
        ∂u_z/∂t = -u_r*∂u_z/∂r - u_z*∂u_z/∂z - (1/ρ)*∂p/∂z + ν*∇²u_z + F_z
        """
        dr, dz, dt = self.dr, self.dz, self.dt
        rho, nu = self.rho, self.nu
        u_r, u_z, p = self.u_r.copy(), self.u_z.copy(), self.p
        r = self.r
        
        # Body force for pressure-driven flow
        if self.flow_type == 'pressure_driven':
            F_z = -self.dp_dz / rho
        else:
            F_z = 0.0
        
        # Create r array for broadcasting (nr, 1)
        r_2d = r[1:-1, np.newaxis]
        
        # Convection terms using upwind differencing (vectorized)
        # For u_r derivatives in r-direction
        du_r_dr_back = (u_r[1:-1, 1:-1] - u_r[:-2, 1:-1]) / dr
        du_r_dr_fwd = (u_r[2:, 1:-1] - u_r[1:-1, 1:-1]) / dr
        du_r_dr = np.where(u_r[1:-1, 1:-1] >= 0, du_r_dr_back, du_r_dr_fwd)
        
        du_z_dr_back = (u_z[1:-1, 1:-1] - u_z[:-2, 1:-1]) / dr
        du_z_dr_fwd = (u_z[2:, 1:-1] - u_z[1:-1, 1:-1]) / dr
        du_z_dr = np.where(u_r[1:-1, 1:-1] >= 0, du_z_dr_back, du_z_dr_fwd)
        
        # For derivatives in z-direction
        du_r_dz_back = (u_r[1:-1, 1:-1] - u_r[1:-1, :-2]) / dz
        du_r_dz_fwd = (u_r[1:-1, 2:] - u_r[1:-1, 1:-1]) / dz
        du_r_dz = np.where(u_z[1:-1, 1:-1] >= 0, du_r_dz_back, du_r_dz_fwd)
        
        du_z_dz_back = (u_z[1:-1, 1:-1] - u_z[1:-1, :-2]) / dz
        du_z_dz_fwd = (u_z[1:-1, 2:] - u_z[1:-1, 1:-1]) / dz
        du_z_dz = np.where(u_z[1:-1, 1:-1] >= 0, du_z_dz_back, du_z_dz_fwd)
        
        # Diffusion terms (cylindrical Laplacian) - vectorized
        d2u_r_dr2 = (u_r[2:, 1:-1] - 2*u_r[1:-1, 1:-1] + u_r[:-2, 1:-1]) / dr**2
        d2u_r_dz2 = (u_r[1:-1, 2:] - 2*u_r[1:-1, 1:-1] + u_r[1:-1, :-2]) / dz**2
        du_r_dr_c = (u_r[2:, 1:-1] - u_r[:-2, 1:-1]) / (2*dr)
        
        d2u_z_dr2 = (u_z[2:, 1:-1] - 2*u_z[1:-1, 1:-1] + u_z[:-2, 1:-1]) / dr**2
        d2u_z_dz2 = (u_z[1:-1, 2:] - 2*u_z[1:-1, 1:-1] + u_z[1:-1, :-2]) / dz**2
        du_z_dr_c = (u_z[2:, 1:-1] - u_z[:-2, 1:-1]) / (2*dr)
        
        laplacian_u_r = d2u_r_dr2 + du_r_dr_c/r_2d + d2u_r_dz2 - u_r[1:-1, 1:-1]/r_2d**2
        laplacian_u_z = d2u_z_dr2 + du_z_dr_c/r_2d + d2u_z_dz2
        
        # Pressure gradients - vectorized
        dp_dr = (p[2:, 1:-1] - p[:-2, 1:-1]) / (2*dr)
        dp_dz = (p[1:-1, 2:] - p[1:-1, :-2]) / (2*dz)
        
        # Update radial velocity
        self.u_r[1:-1, 1:-1] = u_r[1:-1, 1:-1] + dt * (
            - u_r[1:-1, 1:-1] * du_r_dr
            - u_z[1:-1, 1:-1] * du_r_dz
            - dp_dr / rho
            + nu * laplacian_u_r
        )
        
        # Update axial velocity (includes body force)
        self.u_z[1:-1, 1:-1] = u_z[1:-1, 1:-1] + dt * (
            - u_r[1:-1, 1:-1] * du_z_dr
            - u_z[1:-1, 1:-1] * du_z_dz
            - dp_dz / rho
            + nu * laplacian_u_z
            + F_z
        )
            
    def compute_residual(self, u_r_old, u_z_old):
        """
        Compute the L2 norm of velocity change.
        """
        u_r_diff = np.sum((self.u_r - u_r_old)**2)
        u_z_diff = np.sum((self.u_z - u_z_old)**2)
        return np.sqrt(u_r_diff + u_z_diff)
    
    def compute_mass_flow_rate(self):
        """
        Compute volumetric and mass flow rates at outlet.
        
        Q = ∫∫ u_z dA = ∫₀^R u_z(r) * 2πr dr
        """
        # Integrate using trapezoidal rule
        u_z_outlet = self.u_z[:, -1]
        Q = 2 * np.pi * np.trapezoid(u_z_outlet * self.r, self.r)
        m_dot = self.rho * Q
        return Q, m_dot
    
    def compute_wall_shear_stress(self):
        """
        Compute wall shear stress along the pipe.
        
        τ_w = μ * (∂u_z/∂r)|_{r=R}
        """
        tau_w = self.mu * (self.u_z[-1, :] - self.u_z[-2, :]) / self.dr
        return np.abs(tau_w)
    
    def compute_friction_factor(self):
        """
        Compute Darcy friction factor.
        
        f = (Δp / L) * D / (0.5 * ρ * u_avg²)
        
        For pressure-driven flow, use the imposed pressure gradient.
        """
        Q, _ = self.compute_mass_flow_rate()
        A = np.pi * self.radius**2
        u_avg = Q / A if Q > 0 else 1e-10
        
        D = 2 * self.radius
        
        # For pressure-driven flow, use the imposed gradient
        if self.flow_type == 'pressure_driven':
            dp_dz = abs(self.dp_dz)
        else:
            # Estimate from numerical solution
            p_inlet = np.mean(self.p[:, 1])
            p_outlet = np.mean(self.p[:, -2])
            dp_dz = abs(p_inlet - p_outlet) / self.length
            
        f = dp_dz * D / (0.5 * self.rho * u_avg**2 + 1e-10)
        
        return f, u_avg
    
    def solve(self, nt=5000, nit=50, tol=1e-7, print_interval=500, initialize=True):
        """
        Main solver loop.
        
        Parameters:
        -----------
        nt : int
            Maximum number of time steps
        nit : int
            Number of iterations for pressure solver
        tol : float
            Convergence tolerance
        print_interval : int
            Print progress every N steps
        initialize : bool
            Whether to initialize flow field with analytical guess
            
        Returns:
        --------
        history : dict
            Dictionary containing convergence history
        """
        print(f"\nStarting pipe flow simulation...")
        print(f"Grid: {self.nr} (radial) x {self.nz} (axial)")
        print(f"Domain: R = {self.radius*1000:.1f} mm, L = {self.length*1000:.1f} mm")
        print(f"Time step: {self.dt:.2e} s")
        print(f"Kinematic viscosity: {self.nu:.2e} m²/s")
        print("-" * 60)
        
        # Initialize flow field
        if initialize:
            self.initialize_flow_field()
            print("Flow field initialized with analytical profile")
        
        history = {'residuals': [], 'iterations': [], 'mass_flow': []}
        
        for n in range(nt):
            u_r_old = self.u_r.copy()
            u_z_old = self.u_z.copy()
            
            # Build RHS of pressure Poisson equation
            self.build_pressure_rhs()
            
            # Solve pressure Poisson equation
            self.solve_pressure_poisson(nit=nit)
            
            # Update velocity field
            self.compute_velocity()
            
            # Apply boundary conditions
            self.apply_boundary_conditions()
            
            # Check convergence
            residual = self.compute_residual(u_r_old, u_z_old)
            Q, m_dot = self.compute_mass_flow_rate()
            
            history['residuals'].append(residual)
            history['iterations'].append(n)
            history['mass_flow'].append(Q)
            
            if n % print_interval == 0:
                print(f"Iter {n:5d}: Residual = {residual:.2e}, Q = {Q*1e6:.4f} mL/s")
                
            if residual < tol:
                print(f"\nConverged after {n} iterations!")
                print(f"Final residual: {residual:.2e}")
                break
        else:
            print(f"\nReached maximum iterations ({nt})")
            print(f"Final residual: {residual:.2e}")
            
        return history
    
    def get_analytical_poiseuille(self):
        """
        Get analytical Poiseuille flow solution for comparison.
        
        u_z(r) = (Δp/L) * R² / (4μ) * (1 - (r/R)²)
        """
        if self.flow_type == 'pressure_driven':
            dp_dz = self.dp_dz
        else:
            # Estimate from numerical solution
            p_inlet = np.mean(self.p[:, 1])
            p_outlet = np.mean(self.p[:, -2])
            dp_dz = (p_outlet - p_inlet) / self.length
            
        u_z_analytical = -dp_dz * self.radius**2 / (4 * self.mu) * (1 - (self.r / self.radius)**2)
        return u_z_analytical
    
    def plot_results(self, save_path=None):
        """
        Plot velocity field, pressure, and flow characteristics.
        Note: r=0 (centerline) is at the bottom, r=R (wall) is at the top.
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Axial velocity contour
        ax1 = axes[0, 0]
        c1 = ax1.contourf(self.Z * 1000, self.R * 1000, self.u_z, levels=50, cmap=cm.jet)
        ax1.set_title('Axial Velocity $u_z$ [m/s]')
        ax1.set_xlabel('z [mm]')
        ax1.set_ylabel('r [mm] (0=centerline, top=wall)')
        ax1.axhline(y=0, color='w', linestyle='--', linewidth=0.5, alpha=0.5)
        plt.colorbar(c1, ax=ax1)
        
        # Radial velocity contour
        ax2 = axes[0, 1]
        c2 = ax2.contourf(self.Z * 1000, self.R * 1000, self.u_r, levels=50, cmap=cm.coolwarm)
        ax2.set_title('Radial Velocity $u_r$ [m/s]')
        ax2.set_xlabel('z [mm]')
        ax2.set_ylabel('r [mm]')
        plt.colorbar(c2, ax=ax2)
        
        # Pressure field
        ax3 = axes[1, 0]
        c3 = ax3.contourf(self.Z * 1000, self.R * 1000, self.p, levels=50, cmap=cm.viridis)
        ax3.set_title('Pressure [Pa]')
        ax3.set_xlabel('z [mm]')
        ax3.set_ylabel('r [mm]')
        plt.colorbar(c3, ax=ax3)
        
        # Velocity vectors
        ax4 = axes[1, 1]
        skip_r = max(1, self.nr // 15)
        skip_z = max(1, self.nz // 20)
        ax4.quiver(
            self.Z[::skip_r, ::skip_z] * 1000,
            self.R[::skip_r, ::skip_z] * 1000,
            self.u_z[::skip_r, ::skip_z],
            self.u_r[::skip_r, ::skip_z],
            scale=None
        )
        ax4.set_title('Velocity Vectors')
        ax4.set_xlabel('z [mm]')
        ax4.set_ylabel('r [mm]')
        ax4.axhline(y=0, color='k', linestyle='--', linewidth=0.5, alpha=0.5, label='Centerline')
        ax4.axhline(y=self.radius*1000, color='r', linestyle='-', linewidth=1, alpha=0.5, label='Wall')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
            
        plt.show()
        
    def plot_velocity_profiles(self, save_path=None):
        """
        Plot velocity profiles at different axial locations and compare with analytical.
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Axial velocity profiles at different z locations
        ax1 = axes[0]
        z_locations = [0.1, 0.3, 0.5, 0.7, 0.9]
        colors = plt.cm.viridis(np.linspace(0, 1, len(z_locations)))
        
        for z_frac, color in zip(z_locations, colors):
            j = int(z_frac * (self.nz - 1))
            z_pos = self.z[j] * 1000
            ax1.plot(self.u_z[:, j], self.r * 1000, '-', color=color, 
                    linewidth=2, label=f'z = {z_pos:.1f} mm')
        
        # Analytical solution
        u_z_analytical = self.get_analytical_poiseuille()
        ax1.plot(u_z_analytical, self.r * 1000, 'k--', linewidth=2, label='Analytical')
        
        ax1.set_xlabel('Axial Velocity $u_z$ [m/s]')
        ax1.set_ylabel('r [mm]')
        ax1.set_title('Axial Velocity Profiles')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Centerline velocity development
        ax2 = axes[1]
        u_centerline = self.u_z[0, :]
        ax2.plot(self.z * 1000, u_centerline, 'b-', linewidth=2, label='Numerical')
        
        # Analytical centerline velocity
        u_max_analytical = self.get_analytical_poiseuille()[0]
        ax2.axhline(y=u_max_analytical, color='r', linestyle='--', 
                   linewidth=2, label=f'Analytical $u_{{max}}$ = {u_max_analytical:.4f}')
        
        ax2.set_xlabel('z [mm]')
        ax2.set_ylabel('Centerline Velocity [m/s]')
        ax2.set_title('Centerline Velocity Development')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Velocity profiles saved to {save_path}")
            
        plt.show()
        
    def plot_convergence(self, history, save_path=None):
        """
        Plot convergence history.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Residual history
        ax1 = axes[0]
        ax1.semilogy(history['iterations'], history['residuals'], 'b-', linewidth=1.5)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual (L2 norm)')
        ax1.set_title('Convergence History')
        ax1.grid(True, alpha=0.3)
        
        # Mass flow rate history
        ax2 = axes[1]
        Q_ml_s = np.array(history['mass_flow']) * 1e6  # Convert to mL/s
        ax2.plot(history['iterations'], Q_ml_s, 'r-', linewidth=1.5)
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('Volumetric Flow Rate [mL/s]')
        ax2.set_title('Mass Flow Rate Development')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
            
        plt.show()
        
    def print_summary(self):
        """
        Print summary of flow statistics.
        """
        Q, m_dot = self.compute_mass_flow_rate()
        tau_w = self.compute_wall_shear_stress()
        f, u_avg = self.compute_friction_factor()
        
        # Analytical values for comparison
        u_z_analytical = self.get_analytical_poiseuille()
        Q_analytical = 2 * np.pi * np.trapezoid(u_z_analytical * self.r, self.r)
        
        # Theoretical friction factor for laminar flow: f = 64/Re
        f_theoretical = 64 / self.Re if self.Re > 0 else 0
        
        print("\n" + "=" * 60)
        print("FLOW SUMMARY")
        print("=" * 60)
        print(f"Reynolds number: {self.Re:.1f}")
        print(f"\nVelocity:")
        print(f"  Max axial velocity: {np.max(self.u_z):.6f} m/s")
        print(f"  Average velocity: {u_avg:.6f} m/s")
        print(f"  Max radial velocity: {np.max(np.abs(self.u_r)):.2e} m/s")
        print(f"\nFlow rates:")
        print(f"  Volumetric (numerical): {Q*1e6:.4f} mL/s")
        print(f"  Volumetric (analytical): {Q_analytical*1e6:.4f} mL/s")
        print(f"  Mass flow rate: {m_dot*1000:.4f} g/s")
        print(f"  Error: {abs(Q - Q_analytical)/Q_analytical * 100:.2f}%")
        print(f"\nWall shear stress:")
        print(f"  Average: {np.mean(tau_w):.4f} Pa")
        print(f"  Max: {np.max(tau_w):.4f} Pa")
        print(f"\nFriction factor:")
        print(f"  Numerical: {f:.4f}")
        print(f"  Theoretical (64/Re): {f_theoretical:.4f}")
        print("=" * 60)


def run_pipe_flow_simulation(Re=100, nr=41, nz=101, radius=0.01, length=0.2, 
                             flow_type='pressure', nt=10000, plot=True):
    """
    Run a pipe flow simulation at specified Reynolds number.
    
    Parameters:
    -----------
    Re : float
        Target Reynolds number
    nr, nz : int
        Grid resolution
    radius, length : float
        Pipe geometry [m]
    flow_type : str
        'pressure' for pressure-driven, 'velocity' for velocity inlet
    nt : int
        Maximum time steps
    plot : bool
        Whether to plot results
        
    Returns:
    --------
    solver : PipeFlowSolver
        The solver object with results
    """
    # Water-like properties
    rho = 1000.0  # kg/m³
    mu = 0.001    # Pa·s (water at ~20°C)
    
    # Calculate required pressure gradient or inlet velocity
    D = 2 * radius
    nu = mu / rho
    
    if flow_type == 'pressure':
        # u_avg = Re * nu / D
        # For Poiseuille: u_avg = -dp/dz * R² / (8*mu)
        # So: dp/dz = -8 * mu * u_avg / R²
        u_avg = Re * nu / D
        dp_dz = -8 * mu * u_avg / radius**2
    else:
        u_inlet = Re * nu / D
    
    print("=" * 60)
    print(f"PIPE FLOW SIMULATION - Re = {Re}")
    print("=" * 60)
    
    # Create solver
    solver = PipeFlowSolver(nr=nr, nz=nz, radius=radius, length=length, 
                            rho=rho, mu=mu)
    
    if flow_type == 'pressure':
        solver.set_pressure_driven_flow(dp_dz=dp_dz)
    else:
        solver.set_velocity_driven_flow(u_inlet=u_inlet)
        
    solver.apply_boundary_conditions()
    
    # Solve
    history = solver.solve(nt=nt, nit=100, tol=1e-8, print_interval=1000)
    
    # Print summary
    solver.print_summary()
    
    if plot:
        solver.plot_results()
        solver.plot_velocity_profiles()
        solver.plot_convergence(history)
        
    return solver


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("2D AXISYMMETRIC CFD SOLVER - PIPE FLOW")
    print("=" * 60)
    
    # Run simulation at Re = 100 (laminar flow)
    solver = run_pipe_flow_simulation(
        Re=100,
        nr=41,
        nz=101,
        radius=0.01,    # 10 mm radius
        length=0.2,     # 200 mm length
        flow_type='pressure',
        nt=10000,
        plot=True
    )
    
    print("\nSimulation complete!")
