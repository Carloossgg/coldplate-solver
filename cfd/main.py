#!/usr/bin/env python3
"""
Configuration script for microfluidic cold plate optimization
Adjust parameters here and run main.py
"""

from topology_optimizer import (
    PhysicalProperties, 
    OptimizationParameters,
    MicrofluidicOptimizer,
    create_sample_power_map
)
from visualize_results import plot_all_results
import numpy as np

def run_optimization(config_name: str = "default"):
    """
    Run optimization with specified configuration
    
    Args:
        config_name: Name of configuration to use
    """
    
    if config_name == "default":
        print("\n" + "="*70)
        print("Configuration: DEFAULT (100mm x 100mm, medium resolution)")
        print("="*70)
        
        params = OptimizationParameters(
            Lx=0.10,          # 10cm domain
            Ly=0.10,
            nelx=80,          # 80x80 grid
            nely=80,
            volfrac=0.5,      # 50% fluid channels
            penal=3.0,
            rmin=3.0,
            max_iter=100,
            min_feature_size=1.5e-3,  # 1.5mm min features
            p_inlet=1000.0,   # 10 mbar
        )
        
        props = PhysicalProperties(
            H=5e-3,           # 5mm channel depth
            tb=5e-3,          # 5mm base thickness
            k_s=200.0         # Aluminum
        )
        
        # Power map: 4 hotspots representing cores
        power_map = create_sample_power_map(params.nelx, params.nely, 
            hotspot_locs=[
                (0.3, 0.3, 50000, 0.1),
                (0.7, 0.3, 50000, 0.1),
                (0.3, 0.7, 50000, 0.1),
                (0.7, 0.7, 50000, 0.1),
            ])
    
    elif config_name == "high_res":
        print("\n" + "="*70)
        print("Configuration: HIGH RESOLUTION (finer features, slower)")
        print("="*70)
        
        params = OptimizationParameters(
            Lx=0.10,
            Ly=0.10,
            nelx=120,         # Higher resolution
            nely=120,
            volfrac=0.5,
            penal=3.0,
            rmin=4.0,
            max_iter=150,
            min_feature_size=1.0e-3,  # 1mm min features
            p_inlet=1000.0,
        )
        
        props = PhysicalProperties(
            H=5e-3,
            tb=5e-3,
            k_s=200.0
        )
        
        power_map = create_sample_power_map(params.nelx, params.nely)
    
    elif config_name == "small_domain":
        print("\n" + "="*70)
        print("Configuration: SMALL DOMAIN (50mm x 50mm, faster)")
        print("="*70)
        
        params = OptimizationParameters(
            Lx=0.05,          # 5cm domain
            Ly=0.05,
            nelx=60,
            nely=60,
            volfrac=0.5,
            penal=3.0,
            rmin=3.0,
            max_iter=80,
            min_feature_size=1.0e-3,
            p_inlet=800.0,
        )
        
        props = PhysicalProperties(
            H=3e-3,           # 3mm channel depth
            tb=3e-3,
            k_s=200.0
        )
        
        power_map = create_sample_power_map(params.nelx, params.nely,
            hotspot_locs=[
                (0.3, 0.3, 60000, 0.12),
                (0.7, 0.7, 60000, 0.12),
            ])
    
    elif config_name == "gpu_cooler":
        print("\n" + "="*70)
        print("Configuration: GPU COOLER (high power, 8 hotspots)")
        print("="*70)
        
        params = OptimizationParameters(
            Lx=0.12,          # 12cm domain (larger GPU)
            Ly=0.12,
            nelx=90,
            nely=90,
            volfrac=0.45,     # Less fluid, more solid for strength
            penal=3.0,
            rmin=3.5,
            max_iter=120,
            min_feature_size=2.0e-3,  # 2mm for manufacturability
            p_inlet=1500.0,   # Higher pressure
        )
        
        props = PhysicalProperties(
            H=6e-3,           # 6mm channel depth (more flow)
            tb=6e-3,
            k_s=400.0         # Copper for better heat spreading
        )
        
        # 8 hotspots in 2x4 pattern
        power_map = create_sample_power_map(params.nelx, params.nely,
            hotspot_locs=[
                (0.25, 0.25, 70000, 0.08),
                (0.50, 0.25, 70000, 0.08),
                (0.75, 0.25, 70000, 0.08),
                (0.25, 0.50, 70000, 0.08),
                (0.50, 0.50, 70000, 0.08),
                (0.75, 0.50, 70000, 0.08),
                (0.25, 0.75, 70000, 0.08),
                (0.75, 0.75, 70000, 0.08),
            ])
    
    elif config_name == "custom":
        print("\n" + "="*70)
        print("Configuration: CUSTOM - Modify this section!")
        print("="*70)
        
        # CUSTOMIZE THESE PARAMETERS
        params = OptimizationParameters(
            Lx=0.10,          # Length in X (m)
            Ly=0.10,          # Length in Y (m)
            nelx=80,          # Elements in X
            nely=80,          # Elements in Y
            volfrac=0.5,      # Volume fraction (0-1)
            penal=3.0,        # Penalization (higher = more discrete)
            rmin=3.0,         # Filter radius (affects min feature size)
            max_iter=100,     # Maximum iterations
            min_feature_size=1.5e-3,  # Minimum feature size (m)
            p_inlet=1000.0,   # Inlet pressure (Pa)
        )
        
        props = PhysicalProperties(
            rho_f=998.0,      # Water density (kg/m³)
            mu_f=0.001,       # Water viscosity (Pa·s)
            cp_f=4184.0,      # Water specific heat (J/kg·K)
            k_f=0.598,        # Water thermal conductivity (W/m·K)
            k_s=200.0,        # Solid thermal conductivity (W/m·K)
            H=5e-3,           # Channel height (m)
            tb=5e-3,          # Base thickness (m)
        )
        
        # Custom hotspot locations
        # Format: (x_relative, y_relative, heat_flux_W/m², radius_relative)
        hotspot_locs = [
            (0.3, 0.3, 50000, 0.1),
            (0.7, 0.7, 50000, 0.1),
        ]
        
        power_map = create_sample_power_map(params.nelx, params.nely, hotspot_locs)
    
    else:
        raise ValueError(f"Unknown configuration: {config_name}")
    
    # Print configuration summary
    print(f"\nDomain: {params.Lx*1000:.1f}mm x {params.Ly*1000:.1f}mm")
    print(f"Grid: {params.nelx} x {params.nely} elements")
    print(f"Min feature: {params.min_feature_size*1000:.2f}mm")
    print(f"Channel depth: {props.H*1000:.1f}mm")
    print(f"Inlet pressure: {params.p_inlet:.0f} Pa ({params.p_inlet/100:.1f} mbar)")
    print(f"Max power density: {power_map.max()/1000:.1f} kW/m²")
    print(f"Avg power density: {power_map.mean()/1000:.1f} kW/m²")
    print()
    
    # Run optimization
    optimizer = MicrofluidicOptimizer(power_map, props, params)
    xPhys_final, history = optimizer.optimize(verbose=True)
    
    # Save results
    output_dir = f'/home/claude/{config_name}'
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    np.save(f'{output_dir}/optimized_design.npy', xPhys_final)
    np.save(f'{output_dir}/power_map.npy', power_map)
    np.save(f'{output_dir}/optimization_history.npy', history)
    
    print(f"\nResults saved to {output_dir}/")
    
    # Visualize
    plot_all_results(xPhys_final, power_map, history, save_dir=output_dir)
    
    return xPhys_final, history, optimizer


def compare_configurations():
    """Run and compare multiple configurations"""
    configs = ["small_domain", "default", "gpu_cooler"]
    
    results = {}
    for config in configs:
        print(f"\n\n{'#'*70}")
        print(f"Running configuration: {config.upper()}")
        print(f"{'#'*70}\n")
        
        try:
            xPhys, history, optimizer = run_optimization(config)
            results[config] = {
                'design': xPhys,
                'history': history,
                'optimizer': optimizer
            }
        except Exception as e:
            print(f"Error running {config}: {e}")
            continue
    
    # Comparison summary
    print("\n\n" + "="*70)
    print("COMPARISON SUMMARY")
    print("="*70)
    
    for config, data in results.items():
        hist = data['history']
        opt = data['optimizer']
        
        print(f"\n{config.upper()}:")
        print(f"  Final objective: {hist['objective'][-1]:.6f}")
        print(f"  Iterations: {len(hist['iteration'])}")
        print(f"  Total time: {sum(hist['time']):.2f}s")
        
        if 'final_temperature' in hist:
            T = hist['final_temperature']
            print(f"  Max temperature: {T.max():.2f}°C")
            print(f"  Avg temperature: {T.mean():.2f}°C")
            print(f"  Temp rise: {T.max() - opt.params.T_inlet:.2f}°C")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        config = sys.argv[1]
        if config == "compare":
            compare_configurations()
        else:
            run_optimization(config)
    else:
        # Default: run the default configuration
        print("\nUsage:")
        print("  python main.py [config_name]")
        print("\nAvailable configurations:")
        print("  default       - 100mm x 100mm, medium resolution (recommended)")
        print("  high_res      - Higher resolution, finer features")
        print("  small_domain  - 50mm x 50mm, faster computation")
        print("  gpu_cooler    - Large GPU cooler with 8 hotspots")
        print("  custom        - Modify the code to customize")
        print("  compare       - Run and compare multiple configurations")
        print()
        print("Running DEFAULT configuration...\n")
        
        run_optimization("default")