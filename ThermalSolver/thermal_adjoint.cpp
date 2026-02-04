// ============================================================================
// Thermal Adjoint Solver for Topology Optimization
// ============================================================================

class ThermalAdjointSolver {
public:
    Params p;
    vector<double> lambda_T;     // Adjoint temperature
    vector<double> dJ_dgamma;    // Sensitivity output
    
    // Forward solution (from forward solve)
    const vector<double>& T;
    const vector<double>& gamma;
    const vector<double>& u_xy, v_xy;
    
    void solve_adjoint(int p_norm = 10) {
        int n = p.n_dofs();
        int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz();
        
        // 1. Compute RHS: ∂f/∂T
        vector<double> rhs(n, 0.0);
        double pnorm_val = compute_pnorm(p_norm);
        
        for (int k = 0; k < p.nz_solid; k++) {  // Substrate only
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    int P = idx(i, j, k);
                    double Tb = T[P];
                    // ∂f/∂T = (T/pnorm)^(p-1) / N
                    rhs[P] = pow(Tb / pnorm_val, p_norm - 1) / (Nx * Ny * p.nz_solid);
                }
            }
        }
        
        // 2. Solve A^T λ = rhs
        // Note: For symmetric A, A^T = A (mostly true for diffusion-dominated)
        // For asymmetric (strong advection), need proper transpose
        
        lambda_T.assign(n, 0.0);
        // ... AMGCL solve with transposed matrix ...
        
        // 3. Compute sensitivity: ∂f/∂γ = -λ^T · (∂R/∂γ)
        compute_sensitivity();
    }
    
    void compute_sensitivity() {
        int Nx = p.Nx, Ny = p.Ny, Nz = p.Nz();
        dJ_dgamma.assign(Nx * Ny, 0.0);
        
        // Contribution from thermal conductivity: ∂k/∂γ
        for (int k = p.nz_solid; k < Nz; k++) {
            for (int j = 0; j < Ny; j++) {
                for (int i = 0; i < Nx; i++) {
                    double g = gamma[j * Nx + i];
                    double dkdg = dk_dgamma(g);
                    
                    // Sensitivity contribution from diffusion terms
                    // ∂R/∂γ involves ∂k/∂γ in the diffusion operator
                    // ... detailed FVM derivative computation ...
                    
                    dJ_dgamma[j * Nx + i] += /* computed contribution */;
                }
            }
        }
    }
};