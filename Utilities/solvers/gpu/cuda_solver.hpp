// ============================================================================
// File: Utilities/solvers/gpu/cuda_solver.hpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   AMGCL CUDA GPU solver - header
//   Single precision (float) only for maximum GPU speed.
// ============================================================================

#ifndef SOLVERS_GPU_CUDA_SOLVER_HPP
#define SOLVERS_GPU_CUDA_SOLVER_HPP

#include <vector>

namespace amgcl_cuda {

// Initialize CUDA (call once at start)
void initialize();

// Cleanup CUDA resources (call at end)
void cleanup();

// Solve Ax = b on GPU using AMGCL (FLOAT - fastest!)
int solve(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    std::vector<float>& solution,
    float tolerance,
    int max_iterations
);

// Solve U-momentum on GPU (Jacobi iteration)
int solveMomentumU(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    std::vector<float>& solution,
    float tolerance,
    int max_iterations
);

// Solve V-momentum on GPU (Jacobi iteration)
int solveMomentumV(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    std::vector<float>& solution,
    float tolerance,
    int max_iterations
);

} // namespace amgcl_cuda

#endif // SOLVERS_GPU_CUDA_SOLVER_HPP
