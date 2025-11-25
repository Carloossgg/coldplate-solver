"""
Main demonstration script for the Two-Layer Heat Transfer Solver

This script demonstrates:
1. Setting up the solver with different geometries
2. Running thermal simulations
3. Visualizing results
4. Comparing different heat sink designs
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from heat_solver import (
    SolverParams, 
    SolverInputs, 
    HeatTransferSolver,
    create_dummy_velocity_field,
    create_dummy_geometry,
    create_dummy_heat_flux
)


def plot_results(solver: HeatTransferSolver, inputs: SolverInputs, outputs, title_prefix: str = ""):
    """Create comprehensive visualization of simulation results"""
    
    p = solver.p
    
    # Reshape fields for plotting
    T_t_grid = solver.reshape_to_grid(outputs.T_t)
    T_b_grid = solver.reshape_to_grid(outputs.T_b)
    gamma_grid = inputs.gamma
    u_grid = inputs.u
    v_grid = inputs.v
    
    # Velocity magnitude
    vel_mag = np.sqrt(u_grid**2 + v_grid**2)
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Custom colormap for temperature (blue to red)
    temp_cmap = plt.cm.jet
    
    # 1. Geometry (gamma field)
    ax = axes[0, 0]
    im = ax.imshow(gamma_grid.T, origin='lower', cmap='gray_r',
                   extent=[0, p.L_x*1000, 0, p.L_y*1000])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Heat Sink Geometry\n(Black=Solid/Fin, White=Fluid/Channel)')
    plt.colorbar(im, ax=ax, label='γ (0=solid, 1=fluid)')
    
    # 2. Velocity magnitude
    ax = axes[0, 1]
    im = ax.imshow(vel_mag.T, origin='lower', cmap='viridis',
                   extent=[0, p.L_x*1000, 0, p.L_y*1000])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Velocity Magnitude (m/s)')
    plt.colorbar(im, ax=ax, label='|u| (m/s)')
    
    # 3. Velocity field with streamlines
    ax = axes[0, 2]
    x_nodes = np.linspace(0, p.L_x*1000, p.M + 1)
    y_nodes = np.linspace(0, p.L_y*1000, p.N + 1)
    
    # Background: gamma field
    ax.imshow(gamma_grid.T, origin='lower', cmap='gray_r', alpha=0.3,
              extent=[0, p.L_x*1000, 0, p.L_y*1000])
    
    # Streamlines
    strm = ax.streamplot(x_nodes, y_nodes, u_grid.T, v_grid.T, 
                         color=vel_mag.T, cmap='viridis', density=1.5)
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title('Flow Streamlines')
    plt.colorbar(strm.lines, ax=ax, label='|u| (m/s)')
    
    # 4. Thermal-fluid layer temperature
    ax = axes[1, 0]
    im = ax.imshow(T_t_grid.T, origin='lower', cmap=temp_cmap,
                   extent=[0, p.L_x*1000, 0, p.L_y*1000])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Design Layer Temperature T_t (°C)\nMax: {np.max(T_t_grid):.2f}°C')
    plt.colorbar(im, ax=ax, label='T (°C)')
    
    # 5. Substrate temperature
    ax = axes[1, 1]
    im = ax.imshow(T_b_grid.T, origin='lower', cmap=temp_cmap,
                   extent=[0, p.L_x*1000, 0, p.L_y*1000])
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    ax.set_title(f'Substrate Temperature T_b (°C)\nMax: {np.max(T_b_grid):.2f}°C')
    plt.colorbar(im, ax=ax, label='T (°C)')
    
    # 6. Sensitivity field (if non-zero)
    ax = axes[1, 2]
    sens_grid = solver.reshape_to_grid(outputs.dJ_dgamma, node_based=False)
    
    # Use symmetric colormap for sensitivities
    vmax = max(abs(sens_grid.max()), abs(sens_grid.min()))
    if vmax > 1e-10:
        im = ax.imshow(sens_grid.T, origin='lower', cmap='RdBu_r',
                       extent=[0, p.L_x*1000, 0, p.L_y*1000],
                       vmin=-vmax, vmax=vmax)
        ax.set_title('Sensitivity dJ/dγ')
    else:
        im = ax.imshow(sens_grid.T, origin='lower', cmap='RdBu_r',
                       extent=[0, p.L_x*1000, 0, p.L_y*1000])
        ax.set_title('Sensitivity dJ/dγ (near zero)')
    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')
    plt.colorbar(im, ax=ax, label='dJ/dγ')
    
    # Add overall title
    fig.suptitle(f'{title_prefix}Heat Transfer Simulation Results\n'
                 f'Objective (p-norm T_b): {outputs.objective:.3f}°C | '
                 f'Solve time: {outputs.solve_time*1000:.1f} ms', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    return fig


def plot_comparison(results_dict: dict, params: SolverParams):
    """Compare results from different geometries"""
    
    fig, axes = plt.subplots(2, len(results_dict), figsize=(5*len(results_dict), 10))
    
    if len(results_dict) == 1:
        axes = axes.reshape(2, 1)
    
    temp_cmap = plt.cm.jet
    
    # Find global temperature range for consistent colorscale
    all_T_b = [r['outputs'].T_b for r in results_dict.values()]
    T_min = min(T.min() for T in all_T_b)
    T_max = max(T.max() for T in all_T_b)
    
    for idx, (name, data) in enumerate(results_dict.items()):
        solver = data['solver']
        inputs = data['inputs']
        outputs = data['outputs']
        
        gamma_grid = inputs.gamma
        T_b_grid = solver.reshape_to_grid(outputs.T_b)
        
        # Geometry
        ax = axes[0, idx]
        ax.imshow(gamma_grid.T, origin='lower', cmap='gray_r',
                  extent=[0, params.L_x*1000, 0, params.L_y*1000])
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'{name}\nGeometry')
        
        # Temperature
        ax = axes[1, idx]
        im = ax.imshow(T_b_grid.T, origin='lower', cmap=temp_cmap,
                       extent=[0, params.L_x*1000, 0, params.L_y*1000],
                       vmin=T_min, vmax=T_max)
        ax.set_xlabel('x (mm)')
        ax.set_ylabel('y (mm)')
        ax.set_title(f'T_b Max: {np.max(T_b_grid):.2f}°C\n'
                     f'Objective: {outputs.objective:.2f}°C')
        plt.colorbar(im, ax=ax, label='T (°C)')
    
    fig.suptitle('Comparison of Heat Sink Designs', fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig


def run_single_simulation(params: SolverParams, geometry_pattern: str = 'channels',
                          heat_flux_pattern: str = 'uniform', 
                          mean_velocity: float = 0.1,
                          verbose: bool = True):
    """Run a single simulation with given parameters"""
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Running simulation: {geometry_pattern} geometry, {heat_flux_pattern} heat flux")
        print(f"{'='*60}")
        print(f"Grid: {params.M} x {params.N} elements")
        print(f"Domain: {params.L_x*1000:.1f} x {params.L_y*1000:.1f} mm")
        print(f"Channel height: {2*params.H_t*1000:.2f} mm")
        print(f"Substrate height: {2*params.H_b*1000:.2f} mm")
    
    # Create solver
    solver = HeatTransferSolver(params)
    
    # Create inputs
    gamma = create_dummy_geometry(params.M, params.N, pattern=geometry_pattern)
    u, v = create_dummy_velocity_field(params, mean_velocity=mean_velocity)
    q0_field = create_dummy_heat_flux(params.M, params.N, pattern=heat_flux_pattern)
    
    inputs = SolverInputs(gamma=gamma, u=u, v=v, q0_field=q0_field)
    
    # Solve
    outputs = solver.solve(inputs)
    
    if verbose:
        print(f"\nResults:")
        print(f"  Max T_t (design layer): {np.max(outputs.T_t):.3f} °C")
        print(f"  Max T_b (substrate):    {np.max(outputs.T_b):.3f} °C")
        print(f"  Objective (p-norm):     {outputs.objective:.3f} °C")
        print(f"  Solve time:             {outputs.solve_time*1000:.2f} ms")
        print(f"  Sensitivity range:      [{outputs.dJ_dgamma.min():.2e}, {outputs.dJ_dgamma.max():.2e}]")
    
    return solver, inputs, outputs


def run_geometry_comparison(params: SolverParams):
    """Compare different heat sink geometries"""
    
    print("\n" + "="*60)
    print("GEOMETRY COMPARISON STUDY")
    print("="*60)
    
    geometries = ['channels', 'pins', 'tree', 'random']
    results = {}
    
    for geom in geometries:
        solver, inputs, outputs = run_single_simulation(
            params, 
            geometry_pattern=geom,
            verbose=False
        )
        results[geom.capitalize()] = {
            'solver': solver,
            'inputs': inputs,
            'outputs': outputs
        }
        print(f"{geom.capitalize():10s}: Max T_b = {np.max(outputs.T_b):6.2f}°C, "
              f"Objective = {outputs.objective:6.2f}°C, "
              f"Time = {outputs.solve_time*1000:5.1f}ms")
    
    return results


def run_velocity_study(params: SolverParams):
    """Study effect of flow velocity on cooling performance"""
    
    print("\n" + "="*60)
    print("VELOCITY STUDY")
    print("="*60)
    
    velocities = [0.01, 0.05, 0.1, 0.2, 0.5]
    results = []
    
    for vel in velocities:
        solver, inputs, outputs = run_single_simulation(
            params,
            geometry_pattern='pins',
            mean_velocity=vel,
            verbose=False
        )
        results.append({
            'velocity': vel,
            'max_T_b': np.max(outputs.T_b),
            'objective': outputs.objective
        })
        print(f"Velocity = {vel:5.2f} m/s: Max T_b = {np.max(outputs.T_b):6.2f}°C, "
              f"Objective = {outputs.objective:6.2f}°C")
    
    return results


def run_penalization_study(params: SolverParams):
    """Study effect of penalization parameter q_k"""
    
    print("\n" + "="*60)
    print("PENALIZATION PARAMETER STUDY")
    print("="*60)
    
    q_k_values = [0.1, 1.0, 3.0, 10.0, 30.0]
    results = []
    
    for q_k in q_k_values:
        params_copy = SolverParams(
            M=params.M, N=params.N, L_x=params.L_x, L_y=params.L_y,
            H_t=params.H_t, H_b=params.H_b, q_k=q_k
        )
        
        solver, inputs, outputs = run_single_simulation(
            params_copy,
            geometry_pattern='pins',
            verbose=False
        )
        results.append({
            'q_k': q_k,
            'max_T_b': np.max(outputs.T_b),
            'objective': outputs.objective,
            'sens_range': (outputs.dJ_dgamma.min(), outputs.dJ_dgamma.max())
        })
        print(f"q_k = {q_k:5.1f}: Max T_b = {np.max(outputs.T_b):6.2f}°C, "
              f"Sensitivity range = [{outputs.dJ_dgamma.min():8.2e}, {outputs.dJ_dgamma.max():8.2e}]")
    
    return results


def main():
    """Main demonstration function"""
    
    print("="*60)
    print("TWO-LAYER HEAT TRANSFER SOLVER FOR MICROCHANNEL HEAT SINKS")
    print("Based on Yan et al. (2019)")
    print("="*60)
    
    # Set up parameters
    params = SolverParams(
        M=40,           # 40x40 grid
        N=40,
        L_x=0.01,       # 10mm x 10mm domain
        L_y=0.01,
        H_t=0.25e-3,    # 0.5mm channel height
        H_b=0.1e-3,     # 0.2mm substrate height
        q0=6e4,         # 60 kW/m² heat flux
        T_inlet=0.0,    # 0°C inlet (relative temperature)
        q_k=1.0,        # Initial penalization
        p_norm=10       # p-norm exponent
    )
    
    # =====================================================
    # Demo 1: Single simulation with detailed visualization
    # =====================================================
    print("\n" + "-"*60)
    print("DEMO 1: Single Simulation - Pin Fin Heat Sink")
    print("-"*60)
    
    solver, inputs, outputs = run_single_simulation(
        params,
        geometry_pattern='pins',
        heat_flux_pattern='uniform',
        mean_velocity=0.1
    )
    
    # Plot results
    fig1 = plot_results(solver, inputs, outputs, title_prefix="Pin Fin Array - ")
    fig1.savefig('demo1_pin_fins.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: demo1_pin_fins.png")
    
    # =====================================================
    # Demo 2: Different heat flux patterns
    # =====================================================
    print("\n" + "-"*60)
    print("DEMO 2: Hotspot Heat Flux Pattern")
    print("-"*60)
    
    solver2, inputs2, outputs2 = run_single_simulation(
        params,
        geometry_pattern='pins',
        heat_flux_pattern='hotspot',
        mean_velocity=0.1
    )
    
    fig2 = plot_results(solver2, inputs2, outputs2, title_prefix="Hotspot Heat Source - ")
    fig2.savefig('demo2_hotspot.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: demo2_hotspot.png")
    
    # =====================================================
    # Demo 3: Geometry comparison
    # =====================================================
    results_comparison = run_geometry_comparison(params)
    fig3 = plot_comparison(results_comparison, params)
    fig3.savefig('demo3_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: demo3_comparison.png")
    
    # =====================================================
    # Demo 4: Velocity study
    # =====================================================
    velocity_results = run_velocity_study(params)
    
    # Plot velocity study
    fig4, ax = plt.subplots(figsize=(8, 5))
    velocities = [r['velocity'] for r in velocity_results]
    objectives = [r['objective'] for r in velocity_results]
    ax.plot(velocities, objectives, 'bo-', linewidth=2, markersize=8)
    ax.set_xlabel('Mean Velocity (m/s)', fontsize=12)
    ax.set_ylabel('Objective Temperature (°C)', fontsize=12)
    ax.set_title('Effect of Flow Velocity on Cooling Performance', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log')
    fig4.savefig('demo4_velocity_study.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: demo4_velocity_study.png")
    
    # =====================================================
    # Demo 5: Penalization study
    # =====================================================
    penalization_results = run_penalization_study(params)
    
    # =====================================================
    # Summary
    # =====================================================
    print("\n" + "="*60)
    print("SIMULATION SUMMARY")
    print("="*60)
    print(f"\nGrid size: {params.M} x {params.N} = {params.n_elements} elements")
    print(f"Nodes: {params.n_nodes}")
    print(f"DOFs (T_t + T_b): {2 * params.n_nodes}")
    
    print("\nBest cooling performance by geometry:")
    for name, data in sorted(results_comparison.items(), 
                             key=lambda x: x[1]['outputs'].objective):
        print(f"  {name:10s}: {data['outputs'].objective:.2f}°C")
    
    print("\nAll figures saved to current directory.")
    print("="*60)
    
    plt.show()


if __name__ == "__main__":
    main()
