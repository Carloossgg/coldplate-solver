"""
File: PlotResiduals.py
Author: Peter Tcherkezian
Description: Interactive plotting of residuals and pressure-drop histories from ExportFiles outputs:
  loads residual and pressure-drop histories, produces semilogy residual plots and pressure-drop trends, saves PNGs,
  and supports interactive zoom/pan when matplotlib is available.
"""

try:
    import matplotlib.pyplot as plt  # type: ignore
except ImportError:  # pragma: no cover - handled at runtime
    plt = None

import os
import sys

# Plot residuals and pressure drop
def plot_residuals():
    if plt is None:
        print("Error: matplotlib is not available. Install it to plot residuals.")
        return

    # File Paths
    input_file = "ExportFiles/residuals.txt"
    output_image = "ExportFiles/residuals.png"
    dp_file = "ExportFiles/pressure_drop_history.txt"
    dp_image = "ExportFiles/pressure_drop.png"
    
    if not os.path.exists(input_file):
        print(f"Error: {input_file} not found.")
        print("Please run the C++ fluid solver first.")
        return

    print(f"Reading {input_file}...")
    
    iters = []
    mass_res = []
    u_res = []
    v_res = []
    dp_core_iters = []
    dp_core_values = []
    dp_full_iters = []
    dp_full_values = []
    
    try:
        with open(input_file, 'r') as f:
            # Skip the header line
            header = f.readline()
            
            for line in f:
                parts = line.strip().split()
                if not parts: 
                    continue
                
                # Parse columns: Iter Mass U V PressureDrop
                iters.append(int(parts[0]))
                mass_res.append(float(parts[1]))
                u_res.append(float(parts[2]))
                v_res.append(float(parts[3]))
                if len(parts) > 4:
                    iter_val = int(parts[0])
                    dp_core_iters.append(iter_val)
                    dp_core_values.append(float(parts[4]))
                if len(parts) > 5:
                    iter_val = int(parts[0])
                    dp_full_iters.append(iter_val)
                    dp_full_values.append(float(parts[5]))
        
        # If a dedicated pressure-drop file exists, prefer it for plotting
        if os.path.exists(dp_file):
            print(f"Reading {dp_file}...")
            dp_core_iters.clear()
            dp_core_values.clear()
            dp_full_iters.clear()
            dp_full_values.clear()
            with open(dp_file, 'r') as f_dp:
                header = f_dp.readline()
                for line in f_dp:
                    parts = line.strip().split()
                    if not parts:
                        continue
                    iter_val = int(parts[0])
                    if len(parts) > 1:
                        dp_core_iters.append(iter_val)
                        dp_core_values.append(float(parts[1]))
                    if len(parts) > 2:
                        dp_full_iters.append(iter_val)
                        dp_full_values.append(float(parts[2]))
        
        # --- PLOT 1: RESIDUALS ---
        resid_fig = plt.figure(figsize=(10, 6))
        plt.semilogy(iters, mass_res, label='Mass (Continuity)', linewidth=1.5, color='black')
        plt.semilogy(iters, u_res, label='U-Momentum', linewidth=1.5, linestyle='--', color='blue')
        plt.semilogy(iters, v_res, label='V-Momentum', linewidth=1.5, linestyle='--', color='red')
        plt.title('CFD Solver Convergence History')
        plt.xlabel('Iteration')
        plt.ylabel('Residual (Log Scale)')
        plt.grid(True, which="both", linestyle='-', alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_image, dpi=600)
        print(f"Residual plot saved to {output_image}")
        
        # --- PLOT 2: PRESSURE DROP ---
        if dp_core_iters or dp_full_iters:
            dp_fig = plt.figure(figsize=(10, 6))
            plotted = False
            if dp_core_iters:
                plt.plot(dp_core_iters, dp_core_values, color='green', linewidth=1.5,
                         label='Core dP (after inlet buffer)')
                plotted = True
            if dp_full_iters:
                plt.plot(dp_full_iters, dp_full_values, color='orange', linewidth=1.5,
                         linestyle='--', label='Full-system dP')
                plotted = True

            if plotted:
                plt.title('Total Pressure Drop vs Iteration')
                plt.xlabel('Iteration')
                plt.ylabel('Total Pressure Drop (Pa)')
                plt.grid(True, linestyle='-', alpha=0.3)
                
                plt.legend()
                plt.tight_layout()
                plt.savefig(dp_image, dpi=600)
                print(f"Pressure drop plot saved to {dp_image}")
            else:
                print("Warning: No pressure-drop data found to plot.")
        
        # Matplotlib GUI already supports pan/zoom with the toolbar
        plt.show()  # Remove if running on a server without display
        
    except Exception as e:
        print(f"An error occurred during plotting: {e}")

if __name__ == "__main__":
    plot_residuals()