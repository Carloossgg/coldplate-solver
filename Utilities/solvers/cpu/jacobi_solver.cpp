// ============================================================================
// File: Utilities/solvers/cpu/jacobi_solver.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   CPU parallel Jacobi solver for momentum equations.
//   Uses OpenMP for parallel SpMV and residual computation.
//   Matches GPU implementation exactly for consistent results.
// ============================================================================

#include "jacobi_solver.h"
#include <omp.h>
#include <algorithm>

namespace solvers::cpu {

// Static buffer for double-buffering
static std::vector<float> g_jacobi_buffer;

int solveMomentumJacobi(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& b,
    std::vector<float>& x,
    const std::vector<int>& diagOffset,
    float tol,
    int maxIter)
{
    if (n == 0) return 0;
    
    // Ensure buffer is allocated
    if (g_jacobi_buffer.size() != static_cast<size_t>(n)) {
        g_jacobi_buffer.resize(n);
    }
    
    std::vector<float>* x_curr = &x;
    std::vector<float>* x_next = &g_jacobi_buffer;
    
    float tol_sq = tol * tol;
    int iter = 0;
    
    for (; iter < maxIter; ++iter) {
        float total_resid_sq = 0.0f;
        
        // Parallel SpMV and Jacobi update
        #pragma omp parallel for reduction(+:total_resid_sq) schedule(static)
        for (int i = 0; i < n; ++i) {
            float Ax = 0.0f;
            for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                Ax += values[k] * (*x_curr)[col_idx[k]];
            }
            float r = b[i] - Ax;
            total_resid_sq += r * r;
            // Jacobi update: x_new = x_old + r / diag
            (*x_next)[i] = (*x_curr)[i] + r / values[diagOffset[i]];
        }
        
        std::swap(x_curr, x_next);
        
        // Convergence check (RMS): sqrt(sum(r^2)/n) < tol
        if (total_resid_sq < tol_sq * n) {
            iter++;
            break;
        }
    }
    
    // Copy back if last update was in buffer
    if (x_curr == &g_jacobi_buffer) {
        std::copy(g_jacobi_buffer.begin(), g_jacobi_buffer.end(), x.begin());
    }
    
    return iter;
}

} // namespace solvers::cpu
