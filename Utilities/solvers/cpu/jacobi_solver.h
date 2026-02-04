// ============================================================================
// File: Utilities/solvers/cpu/jacobi_solver.h
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   CPU parallel Jacobi solver for momentum equations.
//   Uses OpenMP for parallel SpMV and residual computation.
// ============================================================================

#ifndef SOLVERS_CPU_JACOBI_SOLVER_H
#define SOLVERS_CPU_JACOBI_SOLVER_H

#include <vector>

namespace solvers::cpu {

// Parallel Jacobi solver for momentum equations
// Matches GPU implementation exactly: x_new = x_old + (b - A*x_old) / diag
// Returns: number of iterations performed
int solveMomentumJacobi(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& b,
    std::vector<float>& x,
    const std::vector<int>& diagOffset,
    float tol,
    int maxIter
);

} // namespace solvers::cpu

#endif // SOLVERS_CPU_JACOBI_SOLVER_H
