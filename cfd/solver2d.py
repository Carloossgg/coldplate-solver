"""
2D Computational Fluid Dynamics (CFD) Solver
============================================
Solves the 2D incompressible Navier-Stokes equations using:
- Finite Difference Method
- Pressure-velocity coupling via SIMPLE-like algorithm
- Lid-driven cavity flow as the default test case

Author: CFD Solver
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


class CFDSolver2D:
    """
    2D CFD Solver for incompressible Navier-Stokes equations.
    
    Governing equations (incompressible flow):
    - Continuity: ∂u/∂x + ∂v/∂y = 0
    - Momentum (x): ∂u/∂t + u∂u/∂x + v∂u/∂y = -1/ρ ∂p/∂x + ν(∂²u/∂x² + ∂²u/∂y²)
    - Momentum (y): ∂v/∂t + u∂v/∂x + v∂v/∂y = -1/ρ ∂p/∂y + ν(∂²v/∂x² + ∂²v/∂y²)
    """
    
    def __init__(self, nx=41, ny=41, lx=1.0, ly=1.0, rho=1.0, nu=0.1, dt=0.001):
        """
        Initialize the CFD solver.
        
        Parameters:
        -----------
        nx, ny : int
            Number of grid points in x and y directions
        lx, ly : float
            Domain length in x and y directions
        rho : float
            Fluid density
        nu : float
            Kinematic viscosity
        dt : float
            Time step
        """
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.rho = rho
        self.nu = nu
        self.dt = dt
        
        # Grid spacing
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)
        
        # Create mesh
        self.x = np.linspace(0, lx, nx)
        self.y = np.linspace(0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        # Initialize velocity and pressure fields
        self.u = np.zeros((ny, nx))  # x-velocity
        self.v = np.zeros((ny, nx))  # y-velocity
        self.p = np.zeros((ny, nx))  # pressure
        
        # Source term for pressure Poisson equation
        self.b = np.zeros((ny, nx))
        
        # Calculate Reynolds number
        self.Re = None
        
    def set_lid_driven_cavity(self, u_lid=1.0):
        """
        Set up lid-driven cavity boundary conditions.
        
        Parameters:
        -----------
        u_lid : float
            Velocity of the top lid
        """
        self.u_lid = u_lid
        self.Re = u_lid * self.lx / self.nu
        print(f"Lid-driven cavity setup: Re = {self.Re:.1f}")
        
    def apply_boundary_conditions(self):
        """
        Apply boundary conditions for lid-driven cavity.
        
        Boundaries:
        - Top (lid): u = u_lid, v = 0
        - Bottom: u = 0, v = 0 (no-slip)
        - Left: u = 0, v = 0 (no-slip)
        - Right: u = 0, v = 0 (no-slip)
        """
        # Top boundary (moving lid)
        self.u[-1, :] = self.u_lid
        self.v[-1, :] = 0
        
        # Bottom boundary
        self.u[0, :] = 0
        self.v[0, :] = 0
        
        # Left boundary
        self.u[:, 0] = 0
        self.v[:, 0] = 0
        
        # Right boundary
        self.u[:, -1] = 0
        self.v[:, -1] = 0
        
    def build_pressure_rhs(self):
        """
        Build the right-hand side of the pressure Poisson equation.
        
        The pressure Poisson equation enforces continuity:
        ∇²p = ρ/Δt (∂u/∂x + ∂v/∂y)
        """
        dx, dy, dt, rho = self.dx, self.dy, self.dt, self.rho
        u, v = self.u, self.v
        
        self.b[1:-1, 1:-1] = (
            rho * (
                # Divergence of velocity
                (u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx) +
                (v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)
            ) / dt
            # Non-linear terms
            - rho * ((u[1:-1, 2:] - u[1:-1, :-2]) / (2 * dx)) ** 2
            - 2 * rho * (
                (u[2:, 1:-1] - u[:-2, 1:-1]) / (2 * dy) *
                (v[1:-1, 2:] - v[1:-1, :-2]) / (2 * dx)
            )
            - rho * ((v[2:, 1:-1] - v[:-2, 1:-1]) / (2 * dy)) ** 2
        )
        
    def solve_pressure_poisson(self, nit=50):
        """
        Solve the pressure Poisson equation using iterative method (Gauss-Seidel).
        
        ∇²p = b
        
        Parameters:
        -----------
        nit : int
            Number of iterations for pressure solver
        """
        dx, dy = self.dx, self.dy
        p = self.p.copy()
        
        for _ in range(nit):
            pn = p.copy()
            
            p[1:-1, 1:-1] = (
                ((pn[1:-1, 2:] + pn[1:-1, :-2]) * dy**2 +
                 (pn[2:, 1:-1] + pn[:-2, 1:-1]) * dx**2 -
                 self.b[1:-1, 1:-1] * dx**2 * dy**2) /
                (2 * (dx**2 + dy**2))
            )
            
            # Pressure boundary conditions (Neumann: ∂p/∂n = 0)
            p[-1, :] = p[-2, :]  # Top
            p[0, :] = p[1, :]    # Bottom
            p[:, 0] = p[:, 1]    # Left
            p[:, -1] = p[:, -2]  # Right
            
        self.p = p
        
    def compute_velocity(self):
        """
        Update velocity field using the momentum equations.
        
        Uses explicit time stepping with central differences for diffusion
        and upwind/central for convection.
        """
        dx, dy, dt = self.dx, self.dy, self.dt
        rho, nu = self.rho, self.nu
        u, v, p = self.u.copy(), self.v.copy(), self.p
        
        # Update u-velocity (interior points)
        self.u[1:-1, 1:-1] = (
            u[1:-1, 1:-1]
            # Convection terms
            - u[1:-1, 1:-1] * dt / dx * (u[1:-1, 1:-1] - u[1:-1, :-2])
            - v[1:-1, 1:-1] * dt / dy * (u[1:-1, 1:-1] - u[:-2, 1:-1])
            # Pressure gradient
            - dt / (2 * rho * dx) * (p[1:-1, 2:] - p[1:-1, :-2])
            # Diffusion terms
            + nu * dt / dx**2 * (u[1:-1, 2:] - 2 * u[1:-1, 1:-1] + u[1:-1, :-2])
            + nu * dt / dy**2 * (u[2:, 1:-1] - 2 * u[1:-1, 1:-1] + u[:-2, 1:-1])
        )
        
        # Update v-velocity (interior points)
        self.v[1:-1, 1:-1] = (
            v[1:-1, 1:-1]
            # Convection terms
            - u[1:-1, 1:-1] * dt / dx * (v[1:-1, 1:-1] - v[1:-1, :-2])
            - v[1:-1, 1:-1] * dt / dy * (v[1:-1, 1:-1] - v[:-2, 1:-1])
            # Pressure gradient
            - dt / (2 * rho * dy) * (p[2:, 1:-1] - p[:-2, 1:-1])
            # Diffusion terms
            + nu * dt / dx**2 * (v[1:-1, 2:] - 2 * v[1:-1, 1:-1] + v[1:-1, :-2])
            + nu * dt / dy**2 * (v[2:, 1:-1] - 2 * v[1:-1, 1:-1] + v[:-2, 1:-1])
        )
        
    def compute_residual(self, u_old, v_old):
        """
        Compute the L2 norm of velocity change (convergence criterion).
        """
        u_diff = np.sum((self.u - u_old) ** 2)
        v_diff = np.sum((self.v - v_old) ** 2)
        return np.sqrt(u_diff + v_diff)
    
    def solve(self, nt=1000, nit=50, tol=1e-6, print_interval=100):
        """
        Main solver loop.
        
        Parameters:
        -----------
        nt : int
            Maximum number of time steps
        nit : int
            Number of iterations for pressure Poisson solver
        tol : float
            Convergence tolerance
        print_interval : int
            Print progress every N steps
            
        Returns:
        --------
        history : dict
            Dictionary containing convergence history
        """
        print(f"Starting CFD simulation...")
        print(f"Grid: {self.nx} x {self.ny}")
        print(f"Time step: {self.dt}")
        print(f"Viscosity: {self.nu}")
        print("-" * 50)
        
        history = {'residuals': [], 'iterations': []}
        
        for n in range(nt):
            u_old = self.u.copy()
            v_old = self.v.copy()
            
            # Build RHS of pressure Poisson equation
            self.build_pressure_rhs()
            
            # Solve pressure Poisson equation
            self.solve_pressure_poisson(nit=nit)
            
            # Update velocity field
            self.compute_velocity()
            
            # Apply boundary conditions
            self.apply_boundary_conditions()
            
            # Check convergence
            residual = self.compute_residual(u_old, v_old)
            history['residuals'].append(residual)
            history['iterations'].append(n)
            
            if n % print_interval == 0:
                print(f"Iteration {n:5d}: Residual = {residual:.2e}")
                
            if residual < tol:
                print(f"\nConverged after {n} iterations!")
                print(f"Final residual: {residual:.2e}")
                break
        else:
            print(f"\nReached maximum iterations ({nt})")
            print(f"Final residual: {residual:.2e}")
            
        return history
    
    def compute_vorticity(self):
        """
        Compute vorticity field: ω = ∂v/∂x - ∂u/∂y
        """
        vorticity = np.zeros_like(self.u)
        vorticity[1:-1, 1:-1] = (
            (self.v[1:-1, 2:] - self.v[1:-1, :-2]) / (2 * self.dx) -
            (self.u[2:, 1:-1] - self.u[:-2, 1:-1]) / (2 * self.dy)
        )
        return vorticity
    
    def compute_stream_function(self, nit=1000, tol=1e-6):
        """
        Compute stream function by solving: ∇²ψ = -ω
        
        Parameters:
        -----------
        nit : int
            Maximum iterations
        tol : float
            Convergence tolerance
        """
        psi = np.zeros((self.ny, self.nx))
        omega = self.compute_vorticity()
        dx, dy = self.dx, self.dy
        
        for _ in range(nit):
            psi_old = psi.copy()
            
            psi[1:-1, 1:-1] = (
                ((psi_old[1:-1, 2:] + psi_old[1:-1, :-2]) * dy**2 +
                 (psi_old[2:, 1:-1] + psi_old[:-2, 1:-1]) * dx**2 +
                 omega[1:-1, 1:-1] * dx**2 * dy**2) /
                (2 * (dx**2 + dy**2))
            )
            
            # Boundary conditions (ψ = 0 on walls)
            psi[0, :] = 0
            psi[-1, :] = 0
            psi[:, 0] = 0
            psi[:, -1] = 0
            
            if np.max(np.abs(psi - psi_old)) < tol:
                break
                
        return psi
    
    def plot_results(self, save_path=None):
        """
        Plot velocity field, pressure, vorticity, and streamlines.
        """
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Velocity magnitude
        vel_mag = np.sqrt(self.u**2 + self.v**2)
        ax1 = axes[0, 0]
        c1 = ax1.contourf(self.X, self.Y, vel_mag, levels=50, cmap=cm.jet)
        ax1.set_title('Velocity Magnitude')
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')
        plt.colorbar(c1, ax=ax1)
        
        # Velocity vectors (quiver plot)
        ax2 = axes[0, 1]
        skip = max(1, self.nx // 20)
        ax2.quiver(
            self.X[::skip, ::skip], self.Y[::skip, ::skip],
            self.u[::skip, ::skip], self.v[::skip, ::skip],
            scale=15
        )
        ax2.set_title('Velocity Vectors')
        ax2.set_xlabel('x')
        ax2.set_ylabel('y')
        ax2.set_aspect('equal')
        
        # Pressure field
        ax3 = axes[1, 0]
        c3 = ax3.contourf(self.X, self.Y, self.p, levels=50, cmap=cm.coolwarm)
        ax3.set_title('Pressure Field')
        ax3.set_xlabel('x')
        ax3.set_ylabel('y')
        plt.colorbar(c3, ax=ax3)
        
        # Streamlines
        ax4 = axes[1, 1]
        psi = self.compute_stream_function()
        c4 = ax4.contour(self.X, self.Y, psi, levels=30, colors='k', linewidths=0.5)
        ax4.set_title('Streamlines')
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')
        ax4.set_aspect('equal')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
        
    def plot_centerline_profiles(self, save_path=None):
        """
        Plot u-velocity along vertical centerline and v-velocity along horizontal centerline.
        These are standard validation plots for lid-driven cavity.
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # u-velocity along vertical centerline (x = 0.5)
        mid_x = self.nx // 2
        ax1 = axes[0]
        ax1.plot(self.u[:, mid_x], self.y, 'b-', linewidth=2)
        ax1.set_xlabel('u-velocity')
        ax1.set_ylabel('y')
        ax1.set_title('u-velocity along vertical centerline (x=0.5)')
        ax1.grid(True, alpha=0.3)
        ax1.axvline(x=0, color='k', linestyle='--', alpha=0.3)
        
        # v-velocity along horizontal centerline (y = 0.5)
        mid_y = self.ny // 2
        ax2 = axes[1]
        ax2.plot(self.x, self.v[mid_y, :], 'r-', linewidth=2)
        ax2.set_xlabel('x')
        ax2.set_ylabel('v-velocity')
        ax2.set_title('v-velocity along horizontal centerline (y=0.5)')
        ax2.grid(True, alpha=0.3)
        ax2.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Centerline plot saved to {save_path}")
            
        plt.show()
        
    def plot_convergence(self, history, save_path=None):
        """
        Plot convergence history.
        """
        plt.figure(figsize=(8, 5))
        plt.semilogy(history['iterations'], history['residuals'], 'b-', linewidth=1.5)
        plt.xlabel('Iteration')
        plt.ylabel('Residual (L2 norm)')
        plt.title('Convergence History')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Convergence plot saved to {save_path}")
            
        plt.show()


def run_lid_driven_cavity(Re=100, nx=41, ny=41, nt=5000, plot=True):
    """
    Run a lid-driven cavity simulation at specified Reynolds number.
    
    Parameters:
    -----------
    Re : float
        Reynolds number
    nx, ny : int
        Grid resolution
    nt : int
        Maximum time steps
    plot : bool
        Whether to plot results
        
    Returns:
    --------
    solver : CFDSolver2D
        The solver object with results
    """
    # Calculate viscosity from Reynolds number
    # Re = U*L/nu, with U = 1.0 and L = 1.0
    nu = 1.0 / Re
    
    # Stability constraint: dt < min(dx²/(4*nu), dx/U)
    dx = 1.0 / (nx - 1)
    dt_diffusion = 0.25 * dx**2 / nu
    dt_convection = dx / 1.0  # U = 1.0
    dt = 0.5 * min(dt_diffusion, dt_convection)
    
    print(f"Running lid-driven cavity at Re = {Re}")
    print(f"Stability: dt_diff = {dt_diffusion:.2e}, dt_conv = {dt_convection:.2e}")
    print(f"Using dt = {dt:.2e}")
    print("=" * 50)
    
    # Create solver
    solver = CFDSolver2D(nx=nx, ny=ny, lx=1.0, ly=1.0, rho=1.0, nu=nu, dt=dt)
    solver.set_lid_driven_cavity(u_lid=1.0)
    solver.apply_boundary_conditions()
    
    # Solve
    history = solver.solve(nt=nt, nit=50, tol=1e-7, print_interval=500)
    
    if plot:
        solver.plot_results()
        solver.plot_centerline_profiles()
        solver.plot_convergence(history)
        
    return solver


class ChannelFlowSolver(CFDSolver2D):
    """
    Extension for pressure-driven channel flow (Poiseuille flow).
    """
    
    def __init__(self, nx=41, ny=41, lx=2.0, ly=1.0, rho=1.0, nu=0.1, dt=0.001):
        super().__init__(nx, ny, lx, ly, rho, nu, dt)
        self.dp_dx = 0  # Pressure gradient
        
    def set_channel_flow(self, dp_dx=-1.0):
        """
        Set up pressure-driven channel flow.
        
        Parameters:
        -----------
        dp_dx : float
            Pressure gradient in x-direction (should be negative for flow in +x)
        """
        self.dp_dx = dp_dx
        u_max = -dp_dx * (self.ly/2)**2 / (2 * self.nu)
        self.Re = u_max * self.ly / self.nu
        print(f"Channel flow setup: dp/dx = {dp_dx}, expected Re = {self.Re:.1f}")
        
    def apply_boundary_conditions(self):
        """
        Apply channel flow boundary conditions:
        - Top and bottom: no-slip (u = v = 0)
        - Left and right: periodic (handled separately)
        """
        # Top boundary (no-slip)
        self.u[-1, :] = 0
        self.v[-1, :] = 0
        
        # Bottom boundary (no-slip)
        self.u[0, :] = 0
        self.v[0, :] = 0
        
        # Periodic in x-direction
        self.u[:, 0] = self.u[:, -2]
        self.u[:, -1] = self.u[:, 1]
        self.v[:, 0] = self.v[:, -2]
        self.v[:, -1] = self.v[:, 1]
        
    def compute_velocity(self):
        """
        Update velocity with additional pressure gradient source term.
        """
        super().compute_velocity()
        
        # Add pressure gradient driving force
        self.u[1:-1, 1:-1] -= self.dt / self.rho * self.dp_dx


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 60)
    print("2D CFD SOLVER - Lid-Driven Cavity Flow")
    print("=" * 60)
    
    # Run simulation at Re = 100
    solver = run_lid_driven_cavity(Re=100, nx=41, ny=41, nt=5000, plot=True)
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)
    
    # Print some statistics
    print(f"\nFlow Statistics:")
    print(f"  Max u-velocity: {np.max(solver.u):.4f}")
    print(f"  Min u-velocity: {np.min(solver.u):.4f}")
    print(f"  Max v-velocity: {np.max(solver.v):.4f}")
    print(f"  Min v-velocity: {np.min(solver.v):.4f}")
    print(f"  Max pressure: {np.max(solver.p):.4f}")
    print(f"  Min pressure: {np.min(solver.p):.4f}")