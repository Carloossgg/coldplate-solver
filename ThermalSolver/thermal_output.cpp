// ============================================================================
// File: ThermalSolver/thermal_output.cpp
// ============================================================================
// Description:
//   Implementation of thermal output module for post-processing.
//   Computes volume-weighted average temperatures for different regions.
//
// ============================================================================

// Note: This file is #included in thermal_solver.cpp after Params is defined
// Do not compile separately

// ============================================================================
// Thermal Metrics Structure
// ============================================================================
struct ThermalMetrics {
    // Average temperatures [°C]
    double T_avg_base;    // Average temperature of substrate/base plate (k < nz_solid)
    double T_avg_fluid;   // Average temperature of fluid regions (gamma > 0.01)
    double T_avg_solid;   // Average temperature of solid fins (gamma <= 0.01 AND k >= nz_solid)
    double T_avg_global;  // Volume-weighted average of entire domain
    
    // Standard deviations [°C]
    double T_std_base;    // Std dev of base plate temperature
    double T_std_fluid;   // Std dev of fluid region temperature
    double T_std_global;  // Std dev of entire domain
    
    // Temperature ranges [°C]
    double T_range_base;  // Temperature range (max - min) in base plate
    double T_range_fluid; // Temperature range in fluid region
    double T_range_global;// Temperature range of entire domain
    
    // Uniformity indices (1 = perfectly uniform, 0 = highly non-uniform)
    double T_uniformity_base;   // Uniformity index for base plate: 1 - (std/avg)
    double T_uniformity_global; // Uniformity index for entire domain
    
    // Gradient statistics [°C/m]
    double grad_max;           // Maximum gradient magnitude |∇T| in domain
    double grad_avg;           // Volume-weighted average gradient magnitude
    double grad_max_base;      // Maximum gradient magnitude in base plate
    double grad_avg_base;      // Average gradient magnitude in base plate
    int grad_max_i, grad_max_j, grad_max_k;  // Location of maximum gradient (cell indices)
};

// ============================================================================
// Gradient Field Structure
// ============================================================================
// Stores full 3D gradient vector field for VTK export and visualization.
// All components in [°C/m].
// ============================================================================
struct GradientField {
    vector<double> dT_dx;      // x-component of gradient [Nx*Ny*Nz]
    vector<double> dT_dy;      // y-component of gradient [Nx*Ny*Nz]
    vector<double> dT_dz;      // z-component of gradient [Nx*Ny*Nz]
    vector<double> magnitude;  // |∇T| at each cell [Nx*Ny*Nz]
};

// ============================================================================
// Compute Gradient Components at a Single Cell
// ============================================================================
// Uses finite differences: central for interior, one-sided at boundaries
// Outputs dT/dx, dT/dy, dT/dz components
// ============================================================================
void compute_gradient_components(
    const vector<double>& T,
    int i, int j, int k,
    int Nx, int Ny, int Nz,
    double dx, double dy,
    const vector<double>& dz_cells,
    double& dTdx, double& dTdy, double& dTdz
) {
    int P = k * Nx * Ny + j * Nx + i;
    
    // dT/dx
    if (i == 0) {
        dTdx = (T[P + 1] - T[P]) / dx;
    } else if (i == Nx - 1) {
        dTdx = (T[P] - T[P - 1]) / dx;
    } else {
        dTdx = (T[P + 1] - T[P - 1]) / (2.0 * dx);
    }
    
    // dT/dy
    if (j == 0) {
        dTdy = (T[P + Nx] - T[P]) / dy;
    } else if (j == Ny - 1) {
        dTdy = (T[P] - T[P - Nx]) / dy;
    } else {
        dTdy = (T[P + Nx] - T[P - Nx]) / (2.0 * dy);
    }
    
    // dT/dz (non-uniform spacing)
    int P_top = P + Nx * Ny;
    int P_bot = P - Nx * Ny;
    if (k == 0) {
        dTdz = (T[P_top] - T[P]) / dz_cells[0];
    } else if (k == Nz - 1) {
        dTdz = (T[P] - T[P_bot]) / dz_cells[Nz - 1];
    } else {
        double dz_avg = 0.5 * (dz_cells[k - 1] + dz_cells[k]);
        dTdz = (T[P_top] - T[P_bot]) / (2.0 * dz_avg);
    }
}

// ============================================================================
// Compute Gradient Magnitude at a Single Cell
// ============================================================================
// Wrapper that returns |∇T| = sqrt((dT/dx)² + (dT/dy)² + (dT/dz)²)
// ============================================================================
double compute_gradient_magnitude(
    const vector<double>& T,
    int i, int j, int k,
    int Nx, int Ny, int Nz,
    double dx, double dy,
    const vector<double>& dz_cells
) {
    double dTdx, dTdy, dTdz;
    compute_gradient_components(T, i, j, k, Nx, Ny, Nz, dx, dy, dz_cells, dTdx, dTdy, dTdz);
    return sqrt(dTdx * dTdx + dTdy * dTdy + dTdz * dTdz);
}

// ============================================================================
// Compute Thermal Metrics
// ============================================================================
// Volume-weighted averaging logic:
//   - Each cell has volume V = dx * dy * dz
//   - dz varies between base plate (dz_s) and channel region (dz_f)
//   - Average = sum(T_i * V_i) / sum(V_i) for cells in each region
//
// Region definitions:
//   - Base plate: k < nz_solid (always solid, below channels)
//   - Fluid: k >= nz_solid AND gamma > 0.01 (channel fluid)
//   - Solid fins: k >= nz_solid AND gamma <= 0.01 (fin material)
// ============================================================================
ThermalMetrics compute_thermal_metrics(
    const vector<double>& T,
    const vector<double>& gamma,
    const vector<double>& dz_cells,
    const Params& p
) {
    ThermalMetrics metrics = {};
    
    int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz();
    double dx = p.dx(), dy = p.dy();
    
    // =========================================================================
    // PASS 1: Compute averages and track min/max for each region
    // =========================================================================
    
    // Accumulators for volume-weighted sums
    double sum_TV_base = 0.0, sum_V_base = 0.0;
    double sum_TV_fluid = 0.0, sum_V_fluid = 0.0;
    double sum_TV_solid = 0.0, sum_V_solid = 0.0;
    double sum_TV_global = 0.0, sum_V_global = 0.0;
    
    // Min/max trackers
    double T_min_base = 1e30, T_max_base = -1e30;
    double T_min_fluid = 1e30, T_max_fluid = -1e30;
    double T_min_global = 1e30, T_max_global = -1e30;
    
    for (int k = 0; k < Nz; k++) {
        double dz_k = dz_cells[k];
        double V_cell = dx * dy * dz_k;
        
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int P = k * Nx * Ny + j * Nx + i;
                double T_cell = T[P];
                double TV = T_cell * V_cell;
                
                // Global (all cells)
                sum_TV_global += TV;
                sum_V_global += V_cell;
                T_min_global = min(T_min_global, T_cell);
                T_max_global = max(T_max_global, T_cell);
                
                if (k < p.nz_solid) {
                    // Base plate region
                    sum_TV_base += TV;
                    sum_V_base += V_cell;
                    T_min_base = min(T_min_base, T_cell);
                    T_max_base = max(T_max_base, T_cell);
                } else {
                    double g = gamma[j * Nx + i];
                    if (g > 0.01) {
                        // Fluid region
                        sum_TV_fluid += TV;
                        sum_V_fluid += V_cell;
                        T_min_fluid = min(T_min_fluid, T_cell);
                        T_max_fluid = max(T_max_fluid, T_cell);
                    } else {
                        // Solid fin region
                        sum_TV_solid += TV;
                        sum_V_solid += V_cell;
                    }
                }
            }
        }
    }
    
    // Compute averages
    metrics.T_avg_base = (sum_V_base > 0) ? sum_TV_base / sum_V_base : 0.0;
    metrics.T_avg_fluid = (sum_V_fluid > 0) ? sum_TV_fluid / sum_V_fluid : 0.0;
    metrics.T_avg_solid = (sum_V_solid > 0) ? sum_TV_solid / sum_V_solid : 0.0;
    metrics.T_avg_global = (sum_V_global > 0) ? sum_TV_global / sum_V_global : 0.0;
    
    // Compute ranges
    metrics.T_range_base = (sum_V_base > 0) ? (T_max_base - T_min_base) : 0.0;
    metrics.T_range_fluid = (sum_V_fluid > 0) ? (T_max_fluid - T_min_fluid) : 0.0;
    metrics.T_range_global = (sum_V_global > 0) ? (T_max_global - T_min_global) : 0.0;
    
    // =========================================================================
    // PASS 2: Compute variance using the computed averages
    // =========================================================================
    // var = sum(V_i * (T_i - T_avg)^2) / sum(V_i)
    
    double sum_var_base = 0.0;
    double sum_var_fluid = 0.0;
    double sum_var_global = 0.0;
    
    for (int k = 0; k < Nz; k++) {
        double dz_k = dz_cells[k];
        double V_cell = dx * dy * dz_k;
        
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int P = k * Nx * Ny + j * Nx + i;
                double T_cell = T[P];
                
                // Global variance
                double diff_global = T_cell - metrics.T_avg_global;
                sum_var_global += V_cell * diff_global * diff_global;
                
                if (k < p.nz_solid) {
                    // Base plate variance
                    double diff_base = T_cell - metrics.T_avg_base;
                    sum_var_base += V_cell * diff_base * diff_base;
                } else {
                    double g = gamma[j * Nx + i];
                    if (g > 0.01) {
                        // Fluid variance
                        double diff_fluid = T_cell - metrics.T_avg_fluid;
                        sum_var_fluid += V_cell * diff_fluid * diff_fluid;
                    }
                }
            }
        }
    }
    
    // Compute standard deviations
    double var_base = (sum_V_base > 0) ? sum_var_base / sum_V_base : 0.0;
    double var_fluid = (sum_V_fluid > 0) ? sum_var_fluid / sum_V_fluid : 0.0;
    double var_global = (sum_V_global > 0) ? sum_var_global / sum_V_global : 0.0;
    
    metrics.T_std_base = sqrt(var_base);
    metrics.T_std_fluid = sqrt(var_fluid);
    metrics.T_std_global = sqrt(var_global);
    
    // Compute uniformity indices: 1 - (std / avg), clamped to [0, 1]
    auto compute_uniformity = [](double std_dev, double avg) -> double {
        if (avg <= 0.0) return 0.0;
        double u = 1.0 - (std_dev / avg);
        return max(0.0, min(1.0, u));
    };
    
    metrics.T_uniformity_base = compute_uniformity(metrics.T_std_base, metrics.T_avg_base);
    metrics.T_uniformity_global = compute_uniformity(metrics.T_std_global, metrics.T_avg_global);
    
    // =========================================================================
    // PASS 3: Compute gradient statistics
    // =========================================================================
    
    // Initialize gradient trackers
    metrics.grad_max = -1e30;
    metrics.grad_avg = 0.0;
    metrics.grad_max_base = -1e30;
    metrics.grad_avg_base = 0.0;
    metrics.grad_max_i = 0;
    metrics.grad_max_j = 0;
    metrics.grad_max_k = 0;
    
    double sum_gradV_global = 0.0;
    double sum_gradV_base = 0.0;
    
    for (int k = 0; k < Nz; k++) {
        double dz_k = dz_cells[k];
        double V_cell = dx * dy * dz_k;
        
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                // Compute gradient magnitude at this cell
                double grad_mag = compute_gradient_magnitude(
                    T, i, j, k, Nx, Ny, Nz, dx, dy, dz_cells
                );
                
                // Global gradient statistics
                sum_gradV_global += grad_mag * V_cell;
                if (grad_mag > metrics.grad_max) {
                    metrics.grad_max = grad_mag;
                    metrics.grad_max_i = i;
                    metrics.grad_max_j = j;
                    metrics.grad_max_k = k;
                }
                
                // Base plate gradient statistics
                if (k < p.nz_solid) {
                    sum_gradV_base += grad_mag * V_cell;
                    if (grad_mag > metrics.grad_max_base) {
                        metrics.grad_max_base = grad_mag;
                    }
                }
            }
        }
    }
    
    // Compute average gradients
    metrics.grad_avg = (sum_V_global > 0) ? sum_gradV_global / sum_V_global : 0.0;
    metrics.grad_avg_base = (sum_V_base > 0) ? sum_gradV_base / sum_V_base : 0.0;
    
    return metrics;
}

// ============================================================================
// Save Thermal Metrics to File
// ============================================================================
void save_thermal_metrics(const ThermalMetrics& metrics, const string& dir) {
    string filepath = dir + "/thermal_metrics.txt";
    ofstream f(filepath);
    
    if (!f) {
        cerr << "Warning: Could not open " << filepath << " for writing" << endl;
        return;
    }
    
    f << fixed << setprecision(4);
    f << "# Thermal Metrics - Volume-Weighted Statistics" << endl;
    f << "# All temperatures in [C]" << endl;
    f << "#" << endl;
    f << "# Average Temperatures" << endl;
    f << "T_avg_base    " << metrics.T_avg_base << endl;
    f << "T_avg_fluid   " << metrics.T_avg_fluid << endl;
    f << "T_avg_solid   " << metrics.T_avg_solid << endl;
    f << "T_avg_global  " << metrics.T_avg_global << endl;
    f << "#" << endl;
    f << "# Standard Deviations" << endl;
    f << "T_std_base    " << metrics.T_std_base << endl;
    f << "T_std_fluid   " << metrics.T_std_fluid << endl;
    f << "T_std_global  " << metrics.T_std_global << endl;
    f << "#" << endl;
    f << "# Temperature Ranges (max - min)" << endl;
    f << "T_range_base  " << metrics.T_range_base << endl;
    f << "T_range_fluid " << metrics.T_range_fluid << endl;
    f << "T_range_global" << metrics.T_range_global << endl;
    f << "#" << endl;
    f << "# Uniformity Indices (1 = uniform, 0 = non-uniform)" << endl;
    f << "T_uniformity_base   " << metrics.T_uniformity_base << endl;
    f << "T_uniformity_global " << metrics.T_uniformity_global << endl;
    f << "#" << endl;
    f << "# Thermal Gradients [C/m]" << endl;
    f << "grad_max          " << metrics.grad_max << endl;
    f << "grad_avg          " << metrics.grad_avg << endl;
    f << "grad_max_base     " << metrics.grad_max_base << endl;
    f << "grad_avg_base     " << metrics.grad_avg_base << endl;
    f << "grad_max_loc      " << metrics.grad_max_i << " " << metrics.grad_max_j << " " << metrics.grad_max_k << endl;
    
    f.close();
    cout << "Saved: " << filepath << endl;
}

// ============================================================================
// Print Thermal Metrics to Console
// ============================================================================
void print_thermal_metrics(const ThermalMetrics& metrics) {
    cout << endl;
    cout << "[Average Temperatures]" << endl;
    cout << fixed << setprecision(2);
    cout << "  T_avg_base   = " << metrics.T_avg_base << " C (substrate)" << endl;
    cout << "  T_avg_fluid  = " << metrics.T_avg_fluid << " C (channel fluid)" << endl;
    cout << "  T_avg_solid  = " << metrics.T_avg_solid << " C (solid fins)" << endl;
    cout << "  T_avg_global = " << metrics.T_avg_global << " C (entire domain)" << endl;
    
    cout << endl;
    cout << "[Temperature Uniformity]" << endl;
    cout << "  Base plate:" << endl;
    cout << "    std_dev = " << metrics.T_std_base << " C" << endl;
    cout << "    range   = " << metrics.T_range_base << " C (max-min)" << endl;
    cout << setprecision(4);
    cout << "    uniformity = " << metrics.T_uniformity_base << " (1=uniform)" << endl;
    cout << setprecision(2);
    cout << "  Global:" << endl;
    cout << "    std_dev = " << metrics.T_std_global << " C" << endl;
    cout << "    range   = " << metrics.T_range_global << " C (max-min)" << endl;
    cout << setprecision(4);
    cout << "    uniformity = " << metrics.T_uniformity_global << " (1=uniform)" << endl;
    
    cout << endl;
    cout << "[Thermal Gradients]" << endl;
    cout << setprecision(2);
    cout << "  grad_max      = " << metrics.grad_max << " C/m at cell (" 
         << metrics.grad_max_i << ", " << metrics.grad_max_j << ", " << metrics.grad_max_k << ")" << endl;
    cout << "  grad_avg      = " << metrics.grad_avg << " C/m" << endl;
    cout << "  grad_max_base = " << metrics.grad_max_base << " C/m (substrate)" << endl;
    cout << "  grad_avg_base = " << metrics.grad_avg_base << " C/m" << endl;
}

// ============================================================================
// Compute Full Gradient Field
// ============================================================================
// Computes gradient vector at every cell for VTK export
// ============================================================================
GradientField compute_gradient_field(
    const vector<double>& T,
    const vector<double>& dz_cells,
    const Params& p
) {
    int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz();
    int n = Nx * Ny * Nz;
    double dx = p.dx(), dy = p.dy();
    
    GradientField grad;
    grad.dT_dx.resize(n);
    grad.dT_dy.resize(n);
    grad.dT_dz.resize(n);
    grad.magnitude.resize(n);
    
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int P = k * Nx * Ny + j * Nx + i;
                
                double dTdx, dTdy, dTdz;
                compute_gradient_components(T, i, j, k, Nx, Ny, Nz, dx, dy, dz_cells, dTdx, dTdy, dTdz);
                
                grad.dT_dx[P] = dTdx;
                grad.dT_dy[P] = dTdy;
                grad.dT_dz[P] = dTdz;
                grad.magnitude[P] = sqrt(dTdx*dTdx + dTdy*dTdy + dTdz*dTdz);
            }
        }
    }
    
    return grad;
}

// ============================================================================
// Save Gradient Field to VTK
// ============================================================================
// Exports gradient vector field and magnitude for ParaView visualization
// ============================================================================
void save_gradient_vtk(
    const GradientField& grad,
    const vector<double>& dz_cells,
    const Params& p,
    const string& filepath
) {
    ofstream f(filepath);
    if (!f) {
        cerr << "Warning: Could not open " << filepath << " for writing" << endl;
        return;
    }
    
    int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz();
    
    // VTK Header
    f << "# vtk DataFile Version 3.0\n";
    f << "Thermal Gradient Field\n";
    f << "ASCII\n";
    f << "DATASET RECTILINEAR_GRID\n";
    f << "DIMENSIONS " << Nx+1 << " " << Ny+1 << " " << Nz+1 << "\n";
    
    // X coordinates
    f << "X_COORDINATES " << Nx+1 << " double\n";
    for (int i = 0; i <= Nx; i++) f << i * p.dx() << " ";
    f << "\n";
    
    // Y coordinates
    f << "Y_COORDINATES " << Ny+1 << " double\n";
    for (int j = 0; j <= Ny; j++) f << j * p.dy() << " ";
    f << "\n";
    
    // Z coordinates (non-uniform)
    f << "Z_COORDINATES " << Nz+1 << " double\n";
    double z = 0;
    for (int k = 0; k <= Nz; k++) {
        f << z << " ";
        if (k < Nz) z += dz_cells[k];
    }
    f << "\n";
    
    // Cell data header
    f << "CELL_DATA " << Nx * Ny * Nz << "\n";
    
    // Gradient vector field
    f << "VECTORS TemperatureGradient double\n";
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int P = k * Nx * Ny + j * Nx + i;
                f << grad.dT_dx[P] << " " << grad.dT_dy[P] << " " << grad.dT_dz[P] << "\n";
            }
        }
    }
    
    // Gradient magnitude scalar field
    f << "SCALARS GradientMagnitude double 1\n";
    f << "LOOKUP_TABLE default\n";
    for (int k = 0; k < Nz; k++) {
        for (int j = 0; j < Ny; j++) {
            for (int i = 0; i < Nx; i++) {
                int P = k * Nx * Ny + j * Nx + i;
                f << grad.magnitude[P] << "\n";
            }
        }
    }
    
    f.close();
    cout << "Saved: " << filepath << endl;
}

// ============================================================================
// Save Thermal Conductivity Field
// ============================================================================
// Exports the 2D thermal conductivity field k(gamma) to a text file.
// Format matches geometry_thermal.txt (Ny rows x Nx cols, tab-separated).
// This is the k value in the channel region (k computed from gamma via RAMP).
// Base plate k is always k_solid and not included (it's constant).
//
// k(γ) = k_f + (k_s - k_f) * (1 - γ) / (1 + q_k * γ)  [RAMP formula]
// ============================================================================
void save_k_field(
    const vector<double>& gamma,
    const Params& p,
    const string& dir
) {
    string filepath = dir + "/k_thermal.txt";
    ofstream f(filepath);
    
    if (!f) {
        cerr << "Warning: Could not open " << filepath << " for writing" << endl;
        return;
    }
    
    int Nx = p.Nx, Ny = p.Ny;
    double k_s = p.k_s, k_f = p.k_f, qk = p.qk;
    
    f << fixed << setprecision(6);
    
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            double g = gamma[j * Nx + i];
            g = max(0.0, min(1.0, g));  // Clamp to [0, 1]
            
            // RAMP interpolation (same formula as kval() in ThermalSolver3D)
            double num = 1.0 - g;
            double den = 1.0 + qk * g;
            double k = k_f + (k_s - k_f) * num / den;
            
            f << k;
            if (i < Nx - 1) f << "\t";
        }
        f << "\n";
    }
    
    f.close();
    cout << "Saved: " << filepath << endl;
}

// ============================================================================
// Print Thermal Conductivity Statistics
// ============================================================================
// Prints k field statistics and cell type counts to console
// ============================================================================
void print_k_statistics(const vector<double>& gamma, const Params& p) {
    int Nx = p.Nx, Ny = p.Ny;
    double k_s = p.k_s, k_f = p.k_f, qk = p.qk;
    
    double k_min = 1e30, k_max = -1e30;
    double sum_k = 0.0;
    int count_fluid = 0, count_solid = 0, count_buffer = 0;
    
    for (int j = 0; j < Ny; j++) {
        for (int i = 0; i < Nx; i++) {
            double g = gamma[j * Nx + i];
            g = max(0.0, min(1.0, g));
            
            // RAMP interpolation
            double num = 1.0 - g;
            double den = 1.0 + qk * g;
            double k = k_f + (k_s - k_f) * num / den;
            
            k_min = min(k_min, k);
            k_max = max(k_max, k);
            sum_k += k;
            
            // Count cell types
            if (g > 0.99) count_fluid++;
            else if (g < 0.01) count_solid++;
            else count_buffer++;
        }
    }
    
    double k_avg = sum_k / (Nx * Ny);
    
    cout << endl;
    cout << "[Thermal Conductivity Field]" << endl;
    cout << fixed << setprecision(2);
    cout << "  k_min = " << k_min << " W/m-K (at gamma=0: " << k_s << ")" << endl;
    cout << "  k_max = " << k_max << " W/m-K (at gamma=1: " << k_f << ")" << endl;
    cout << "  k_avg = " << k_avg << " W/m-K" << endl;
    cout << "  Cells: " << count_fluid << " fluid, " << count_solid << " solid, " << count_buffer << " buffer" << endl;
    cout << "  RAMP parameter q_k = " << qk << endl;
}
