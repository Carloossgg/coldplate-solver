// ============================================================================
// File: Utilities/solvers/gpu/cuda_common.cuh
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Common CUDA structures, types, and initialization for GPU solvers.
//   All solvers use single precision (float) for 2x GPU speed.
// ============================================================================

#ifndef SOLVERS_GPU_CUDA_COMMON_CUH
#define SOLVERS_GPU_CUDA_COMMON_CUH

#include <iostream>
#include <tuple>
#include <memory>
#include <chrono>
#include <vector>

// AMGCL headers
#include <amgcl/backend/cuda.hpp>
#include <amgcl/make_solver.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

// Thrust for GPU vectors
#include <thrust/device_vector.h>
#include <cuda_runtime.h>
#include <cusparse.h>

namespace amgcl_cuda {

// ============================================================================
// Timing helper
// ============================================================================
using Clock = std::chrono::high_resolution_clock;
using Ms = std::chrono::duration<double, std::milli>;

// ============================================================================
// SINGLE PRECISION CUDA Backend (2x faster than double!)
// ============================================================================
typedef amgcl::backend::cuda<float> Backend;

typedef amgcl::make_solver<
    amgcl::amg<
        Backend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
    >,
    amgcl::solver::cg<Backend>
> Solver;

// ============================================================================
// Global state - shared across pressure solver
// ============================================================================
extern std::unique_ptr<Solver> g_solver;
extern cusparseHandle_t g_cusparse_handle;
extern cudaStream_t g_stream;
extern int g_last_n;
extern int g_last_nnz;
extern int g_call_count;
extern int g_since_rebuild;

// Persistent GPU vectors (FLOAT)
extern thrust::device_vector<float>* g_d_rhs;
extern thrust::device_vector<float>* g_d_sol;

// Pinned host memory for fast transfers (FLOAT)
extern float* g_pinned_rhs;
extern float* g_pinned_sol;
extern int g_pinned_size;

// Preconditioner lagging interval
constexpr int REBUILD_INTERVAL = 15;

// ============================================================================
// Initialize CUDA (call once at start)
// ============================================================================
void initialize();

// ============================================================================
// Ensure pinned memory is allocated
// ============================================================================
void ensurePinnedMemory(int n);

// ============================================================================
// Cleanup CUDA resources
// ============================================================================
void cleanup();

// ============================================================================
// CUDA Kernel: Jacobi iteration step
// ============================================================================
__global__ void jacobiUpdateKernel(
    const float* __restrict__ b,
    const float* __restrict__ Ax,
    const float* __restrict__ diag,
    float* __restrict__ x,
    float* __restrict__ residual,
    int n);

// ============================================================================
// CUDA Kernel: Extract diagonal from CSR matrix
// ============================================================================
__global__ void extractDiagonalKernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    float* __restrict__ diag,
    int n);

} // namespace amgcl_cuda

#endif // SOLVERS_GPU_CUDA_COMMON_CUH
