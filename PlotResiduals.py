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
    mass_res_norm = []
    u_res_norm = []
    v_res_norm = []
    cfl_values = []
    dp_core_iters = []
    dp_core_values = []
    dp_full_iters = []
    dp_full_values = []
    
    try:
        with open(input_file, 'r') as f:
            # Read header line (if present)
            header = f.readline().strip()
            header_cols = header.split() if header else []
            col_index = {name: idx for idx, name in enumerate(header_cols)}
            
            for line in f:
                parts = line.strip().split()
                if not parts: 
                    continue
                
                # Parse columns by header when available; fallback to legacy indices.
                iters.append(int(parts[0]))
                if col_index:
                    # Raw RMS
                    if "MassRMS" in col_index:
                        mass_res.append(float(parts[col_index["MassRMS"]]))
                    if "U_RMS" in col_index:
                        u_res.append(float(parts[col_index["U_RMS"]]))
                    if "V_RMS" in col_index:
                        v_res.append(float(parts[col_index["V_RMS"]]))
                    # Normalized RMS (if present)
                    if "MassRMSn" in col_index:
                        mass_res_norm.append(float(parts[col_index["MassRMSn"]]))
                    if "U_RMSn" in col_index:
                        u_res_norm.append(float(parts[col_index["U_RMSn"]]))
                    if "V_RMSn" in col_index:
                        v_res_norm.append(float(parts[col_index["V_RMSn"]]))
                else:
                    # Legacy layout: Iter Mass U V Core_dP Full_dP CFL
                    mass_res.append(float(parts[1]))
                    u_res.append(float(parts[2]))
                    v_res.append(float(parts[3]))
                
                # Default dP if columns 4 and 5 exist
                # Default dP if columns exist
                if col_index:
                    core_key = "Core_dP_AfterInletBuffer(Pa)"
                    full_key = "Full_dP_FullSystem(Pa)"
                    if core_key in col_index:
                        dp_core_iters.append(int(parts[0]))
                        dp_core_values.append(float(parts[col_index[core_key]]))
                    if full_key in col_index:
                        dp_full_iters.append(int(parts[0]))
                        dp_full_values.append(float(parts[col_index[full_key]]))
                else:
                    if len(parts) > 4:
                        dp_core_iters.append(int(parts[0]))
                        dp_core_values.append(float(parts[4]))
                    if len(parts) > 5:
                        dp_full_iters.append(int(parts[0]))
                        dp_full_values.append(float(parts[5]))
                
                # Handle CFL if exists (new column 6)
                if col_index and "CFL" in col_index:
                    cfl_values.append(float(parts[col_index["CFL"]]))
                elif len(parts) > 6:
                    cfl_values.append(float(parts[6]))
        
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
        
        # --- PLOT 1: RESIDUALS & CFL ---
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Left axis: Residuals (Log Scale)
        ax1.semilogy(iters, mass_res, label='Mass RMS (raw)', linewidth=1.5, color='black')
        ax1.semilogy(iters, u_res, label='U RMS (raw)', linewidth=1.5, linestyle='--', color='blue')
        ax1.semilogy(iters, v_res, label='V RMS (raw)', linewidth=1.5, linestyle='--', color='red')
        if mass_res_norm and u_res_norm and v_res_norm:
            ax1.semilogy(iters, mass_res_norm, label='Mass RMS (norm)', linewidth=1.0, color='black', alpha=0.6)
            ax1.semilogy(iters, u_res_norm, label='U RMS (norm)', linewidth=1.0, linestyle=':', color='blue', alpha=0.6)
            ax1.semilogy(iters, v_res_norm, label='V RMS (norm)', linewidth=1.0, linestyle=':', color='red', alpha=0.6)
        ax1.set_xlabel('Iteration')
        ax1.set_ylabel('Residual (Log Scale)')
        ax1.grid(True, which="both", linestyle='-', alpha=0.3)
        
        # Right axis: CFL (Linear Scale)
        if cfl_values:
            ax2 = ax1.twinx()
            ax2.plot(iters, cfl_values, label='Pseudo-CFL', color='magenta', linewidth=1.5, alpha=0.6)
            ax2.set_ylabel('Pseudo-CFL', color='magenta')
            ax2.tick_params(axis='y', labelcolor='magenta')
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
        else:
            ax1.legend(loc='upper right')

        plt.title('CFD Solver Convergence & CFL History')
        plt.tight_layout()
        plt.savefig(output_image, dpi=600)
        print(f"Convergence plot saved to {output_image}")
        
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
