#!/usr/bin/env python3
"""
Visualization module for microfluidic cold plate optimization results
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from typing import Tuple, Optional

def plot_design(xPhys: np.ndarray, title: str = "Optimized Design", 
                save_path: Optional[str] = None, dpi: int = 300):
    """
    Plot the optimized microfluidic design
    
    Args:
        xPhys: Design density field (nely x nelx)
        title: Plot title
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Create colormap: blue for fluid, gray for solid
    colors = ['#2C3E50', '#3498DB', '#AED6F1']  # Dark gray -> Blue -> Light blue
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=n_bins)
    
    im = ax.imshow(xPhys, cmap=cmap, origin='lower', vmin=0, vmax=1)
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X (elements)', fontsize=12)
    ax.set_ylabel('Y (elements)', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Density (0=Solid, 1=Fluid)', rotation=270, labelpad=20, fontsize=12)
    
    # Add annotations
    ax.text(0.5, 1.05, 'Inlet (Top)', transform=ax.transAxes, 
            ha='center', fontsize=10, color='red', fontweight='bold')
    ax.text(-0.05, 0.5, 'Outlet\n(Left)', transform=ax.transAxes, 
            ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    ax.text(1.05, 0.5, 'Outlet\n(Right)', transform=ax.transAxes, 
            ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Design plot saved to {save_path}")
    
    plt.show()
    return fig, ax


def plot_temperature_field(T_field: np.ndarray, xPhys: np.ndarray,
                           title: str = "Temperature Distribution",
                           save_path: Optional[str] = None, dpi: int = 300):
    """
    Plot temperature field overlaid on design
    
    Args:
        T_field: Temperature field (nely+1 x nelx+1)
        xPhys: Design density field (nely x nelx)
        title: Plot title
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Temperature colormap (hot)
    im = ax.imshow(T_field, cmap='hot', origin='lower', alpha=0.8)
    
    # Overlay design as contours
    ax.contour(xPhys, levels=[0.5], colors='cyan', linewidths=2, 
               extent=[0, xPhys.shape[1], 0, xPhys.shape[0]])
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X (elements)', fontsize=12)
    ax.set_ylabel('Y (elements)', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Temperature (°C)', rotation=270, labelpad=20, fontsize=12)
    
    # Stats
    T_min, T_max, T_avg = T_field.min(), T_field.max(), T_field.mean()
    stats_text = f'Min: {T_min:.1f}°C\nMax: {T_max:.1f}°C\nAvg: {T_avg:.1f}°C'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Temperature plot saved to {save_path}")
    
    plt.show()
    return fig, ax


def plot_velocity_field(velocity: Tuple[np.ndarray, np.ndarray], xPhys: np.ndarray,
                        title: str = "Velocity Field",
                        save_path: Optional[str] = None, dpi: int = 300,
                        subsample: int = 5):
    """
    Plot velocity field with streamlines and arrows
    
    Args:
        velocity: Tuple of (u_x, u_y) velocity components
        xPhys: Design density field (nely x nelx)
        title: Plot title
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure
        subsample: Subsample factor for quiver plot
    """
    u_x, u_y = velocity
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    
    # Velocity magnitude
    u_mag = np.sqrt(u_x**2 + u_y**2)
    
    # Plot magnitude as background
    im = ax.imshow(u_mag, cmap='viridis', origin='lower', alpha=0.6)
    
    # Overlay design contours
    ax.contour(xPhys, levels=[0.5], colors='white', linewidths=1.5, alpha=0.5,
               extent=[0, xPhys.shape[1], 0, xPhys.shape[0]])
    
    # Streamlines
    ny, nx = u_x.shape
    y, x = np.mgrid[0:ny, 0:nx]
    
    # Only plot streamlines in fluid regions
    u_x_masked = u_x.copy()
    u_y_masked = u_y.copy()
    
    # Mask solid regions
    mask = np.zeros_like(u_x_masked, dtype=bool)
    for i in range(min(xPhys.shape[0], ny)):
        for j in range(min(xPhys.shape[1], nx)):
            if xPhys[i, j] < 0.5:
                mask[i, j] = True
    
    u_x_masked[mask] = 0
    u_y_masked[mask] = 0
    
    try:
        ax.streamplot(x[0, :], y[:, 0], u_x_masked.T, u_y_masked.T, 
                     color='white', density=1.5, linewidth=0.5, arrowsize=0.5)
    except:
        print("Warning: Streamplot failed, skipping...")
    
    # Quiver plot (subsampled)
    skip = subsample
    ax.quiver(x[::skip, ::skip], y[::skip, ::skip], 
             u_x_masked[::skip, ::skip], u_y_masked[::skip, ::skip],
             scale=u_mag.max()*30, color='red', alpha=0.6, width=0.003)
    
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X (elements)', fontsize=12)
    ax.set_ylabel('Y (elements)', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Velocity Magnitude (m/s)', rotation=270, labelpad=20, fontsize=12)
    
    # Stats
    u_max = u_mag.max()
    u_avg = u_mag[~mask].mean() if not np.all(mask) else 0
    stats_text = f'Max: {u_max:.4f} m/s\nAvg: {u_avg:.4f} m/s'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Velocity plot saved to {save_path}")
    
    plt.show()
    return fig, ax


def plot_power_map(power_map: np.ndarray, title: str = "Power Distribution (Heat Flux)",
                  save_path: Optional[str] = None, dpi: int = 300):
    """
    Plot the input power/heat flux distribution
    
    Args:
        power_map: Heat flux distribution (W/m^2)
        title: Plot title
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure
    """
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    im = ax.imshow(power_map / 1000, cmap='hot', origin='lower')  # Convert to kW/m^2
    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('X (elements)', fontsize=12)
    ax.set_ylabel('Y (elements)', fontsize=12)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Heat Flux (kW/m²)', rotation=270, labelpad=20, fontsize=12)
    
    # Stats
    q_min, q_max, q_avg = power_map.min()/1000, power_map.max()/1000, power_map.mean()/1000
    stats_text = f'Min: {q_min:.1f} kW/m²\nMax: {q_max:.1f} kW/m²\nAvg: {q_avg:.1f} kW/m²'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Power map plot saved to {save_path}")
    
    plt.show()
    return fig, ax


def plot_convergence_history(history: dict, save_path: Optional[str] = None, dpi: int = 300):
    """
    Plot optimization convergence history
    
    Args:
        history: Dictionary with optimization history
        save_path: If provided, save figure to this path
        dpi: Resolution for saved figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    iterations = history['iteration']
    
    # Objective vs iteration
    ax = axes[0, 0]
    ax.plot(iterations, history['objective'], 'b-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Objective (Max Temp Norm)', fontsize=12)
    ax.set_title('Objective Function', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Change vs iteration (log scale)
    ax = axes[0, 1]
    ax.semilogy(iterations, history['change'], 'r-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Design Change', fontsize=12)
    ax.set_title('Design Change (Log Scale)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, which='both')
    
    # Iteration time
    ax = axes[1, 0]
    ax.plot(iterations, history['time'], 'g-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Time (s)', fontsize=12)
    ax.set_title('Iteration Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Cumulative time
    ax = axes[1, 1]
    cumtime = np.cumsum(history['time'])
    ax.plot(iterations, cumtime, 'm-', linewidth=2)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cumulative Time (s)', fontsize=12)
    ax.set_title('Cumulative Computation Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Convergence plot saved to {save_path}")
    
    plt.show()
    return fig, axes


def plot_all_results(xPhys: np.ndarray, power_map: np.ndarray, history: dict,
                    save_dir: str = '/home/claude'):
    """
    Generate all visualization plots and save them
    
    Args:
        xPhys: Optimized design
        power_map: Input power distribution
        history: Optimization history
        save_dir: Directory to save plots
    """
    print("\nGenerating visualization plots...")
    print("="*60)
    
    # Design
    plot_design(xPhys, title="Optimized Microfluidic Cold Plate Design",
               save_path=f"{save_dir}/design.png")
    
    # Power map
    plot_power_map(power_map, title="Input Heat Flux Distribution",
                  save_path=f"{save_dir}/power_map.png")
    
    # Temperature field
    if 'final_temperature' in history:
        plot_temperature_field(history['final_temperature'], xPhys,
                             title="Temperature Distribution",
                             save_path=f"{save_dir}/temperature.png")
    
    # Velocity field
    if 'final_velocity' in history:
        plot_velocity_field(history['final_velocity'], xPhys,
                           title="Coolant Velocity Field",
                           save_path=f"{save_dir}/velocity.png", subsample=4)
    
    # Convergence history
    plot_convergence_history(history, save_path=f"{save_dir}/convergence.png")
    
    # Combined summary plot
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
    
    # Design
    ax1 = fig.add_subplot(gs[0, 0])
    colors = ['#2C3E50', '#3498DB', '#AED6F1']
    cmap = LinearSegmentedColormap.from_list('custom', colors, N=100)
    im1 = ax1.imshow(xPhys, cmap=cmap, origin='lower')
    ax1.set_title('Optimized Design', fontweight='bold')
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    
    # Power map
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(power_map/1000, cmap='hot', origin='lower')
    ax2.set_title('Heat Flux (kW/m²)', fontweight='bold')
    plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    
    # Temperature
    if 'final_temperature' in history:
        ax3 = fig.add_subplot(gs[0, 2])
        im3 = ax3.imshow(history['final_temperature'], cmap='hot', origin='lower')
        ax3.set_title('Temperature (°C)', fontweight='bold')
        plt.colorbar(im3, ax=ax3, fraction=0.046, pad=0.04)
    
    # Velocity magnitude
    if 'final_velocity' in history:
        ax4 = fig.add_subplot(gs[1, 0])
        u_x, u_y = history['final_velocity']
        u_mag = np.sqrt(u_x**2 + u_y**2)
        im4 = ax4.imshow(u_mag, cmap='viridis', origin='lower')
        ax4.set_title('Velocity Magnitude (m/s)', fontweight='bold')
        plt.colorbar(im4, ax=ax4, fraction=0.046, pad=0.04)
    
    # Objective history
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(history['iteration'], history['objective'], 'b-', linewidth=2)
    ax5.set_xlabel('Iteration')
    ax5.set_ylabel('Objective')
    ax5.set_title('Objective History', fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # Change history
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.semilogy(history['iteration'], history['change'], 'r-', linewidth=2)
    ax6.set_xlabel('Iteration')
    ax6.set_ylabel('Change')
    ax6.set_title('Design Change', fontweight='bold')
    ax6.grid(True, alpha=0.3, which='both')
    
    plt.suptitle('Microfluidic Cold Plate Optimization Summary', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(f"{save_dir}/summary.png", dpi=300, bbox_inches='tight')
    print(f"Summary plot saved to {save_dir}/summary.png")
    plt.show()
    
    print("="*60)
    print("All plots generated successfully!")


if __name__ == "__main__":
    # Load and visualize results
    import os
    
    results_dir = '/home/claude'
    
    if os.path.exists(f'{results_dir}/optimized_design.npy'):
        print("\nLoading optimization results...")
        
        xPhys = np.load(f'{results_dir}/optimized_design.npy')
        power_map = np.load(f'{results_dir}/power_map.npy')
        history = np.load(f'{results_dir}/optimization_history.npy', allow_pickle=True).item()
        
        plot_all_results(xPhys, power_map, history, save_dir=results_dir)
    else:
        print("\nNo results found. Please run topology_optimizer.py first.")