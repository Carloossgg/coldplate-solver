// ============================================================================
// File: Utilities/solvers/cpu/pcg_solver.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Preconditioned Conjugate Gradient (PCG) solver implementation.
//   Fully parallelized using OpenMP for SpMV, dot products, and vector ops.
// ============================================================================

#include "pcg_solver.h"
#include "../linalg.h"
#include <cmath>
#include <algorithm>

namespace solvers {
namespace cpu {

int solvePCG_CSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& b,
    std::vector<float>& x,
    float tol,
    int maxIter)
{
    if (n == 0) return 0;
    
    std::vector<float> r(n), z(n), p(n), Ap(n), diag(n);
    
    // Extract diagonal for Jacobi preconditioner
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float d = 1.0f;
        for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
            if (col_idx[k] == i) { d = values[k]; break; }
        }
        diag[i] = d;
    }
    
    // r = b - A*x
    linalg::parallelSpMV_CSR(n, row_ptr, col_idx, values, x, Ap);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) r[i] = b[i] - Ap[i];
    
    // Apply Jacobi preconditioner: z = M^{-1} * r
    auto applyJacobi = [&](const std::vector<float>& rv, std::vector<float>& zv) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            zv[i] = (std::abs(diag[i]) > 1e-20f) ? rv[i] / diag[i] : rv[i];
        }
    };
    
    applyJacobi(r, z);
    p = z;
    
    float rz = linalg::parallelDot(r, z);
    
    int iter = 0;
    for (; iter < maxIter; ++iter) {
        linalg::parallelSpMV_CSR(n, row_ptr, col_idx, values, p, Ap);
        
        float pAp = linalg::parallelDot(p, Ap);
        if (std::abs(pAp) < 1e-20f) break;
        
        float alpha = rz / pAp;
        
        // x = x + alpha * p, r = r - alpha * Ap
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            x[i] += alpha * p[i];
            r[i] -= alpha * Ap[i];
        }
        
        // Check convergence
        float err = linalg::parallelNorm(r);
        if (err < tol) break;
        
        applyJacobi(r, z);
        float rz_prev = rz;
        rz = linalg::parallelDot(r, z);
        
        float beta = rz / rz_prev;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            p[i] = z[i] + beta * p[i];
        }
    }
    
    return iter;
}

} // namespace cpu
} // namespace solvers
