// ============================================================================
// File: ThermalSolver/thermal_output.h
// ============================================================================
// Description:
//   Thermal output module for post-processing temperature results.
//   Computes volume-weighted average temperatures for different regions.
//
// ============================================================================

#ifndef THERMAL_OUTPUT_H
#define THERMAL_OUTPUT_H

#include <vector>
#include <string>

// Forward declaration of Params struct (defined in thermal_solver.cpp)
struct Params;

// ============================================================================
// Thermal Metrics Structure
// ============================================================================
// Holds computed average temperatures for different regions of the domain.
// All temperatures are volume-weighted averages in [°C].
// ============================================================================
struct ThermalMetrics {
    double T_avg_base;    // Average temperature of substrate/base plate (k < nz_solid)
    double T_avg_fluid;   // Average temperature of fluid regions (gamma > 0.01)
    double T_avg_solid;   // Average temperature of solid fins (gamma <= 0.01 AND k >= nz_solid)
    double T_avg_global;  // Volume-weighted average of entire domain
};

// ============================================================================
// Function Declarations
// ============================================================================

// Compute volume-weighted average temperatures for each region
// Accounts for non-uniform dz spacing between base plate and channel regions
ThermalMetrics compute_thermal_metrics(
    const std::vector<double>& T,           // Temperature field [n_dofs]
    const std::vector<double>& gamma,       // Density field [Nx*Ny], gamma=1 fluid, gamma=0 solid
    const std::vector<double>& dz_cells,    // Cell heights [Nz]
    const Params& p                         // Simulation parameters
);

// Save thermal metrics to file with clear labels
void save_thermal_metrics(
    const ThermalMetrics& metrics,
    const std::string& dir                  // Output directory path
);

// Print thermal metrics summary to console
void print_thermal_metrics(const ThermalMetrics& metrics);

#endif // THERMAL_OUTPUT_H
