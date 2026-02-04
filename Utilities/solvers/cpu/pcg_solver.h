// ============================================================================
// File: Utilities/solvers/cpu/pcg_solver.h
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Preconditioned Conjugate Gradient (PCG) solver for sparse linear systems.
//   Uses Jacobi preconditioning and parallel operations via OpenMP.
// ============================================================================

#ifndef SOLVERS_CPU_PCG_SOLVER_H
#define SOLVERS_CPU_PCG_SOLVER_H

#include <vector>

namespace solvers {
namespace cpu {

// ---------------------------------------------------------------------------
// Solve Ax = b using Preconditioned Conjugate Gradient (Jacobi preconditioner)
// Returns: number of iterations performed
// ---------------------------------------------------------------------------
int solvePCG_CSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& b,
    std::vector<float>& x,
    float tol = 1e-6f,
    int maxIter = 500);

} // namespace cpu
} // namespace solvers

#endif // SOLVERS_CPU_PCG_SOLVER_H
