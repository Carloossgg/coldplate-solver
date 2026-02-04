// ============================================================================
// File: Utilities/solvers/gpu/pressure_solver.cu
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   GPU pressure solver using AMGCL-CUDA with AMG preconditioning.
//   Single precision (float) only for maximum GPU speed.
// ============================================================================

#include "cuda_common.cuh"
#include "cuda_solver.hpp"

namespace amgcl_cuda {

// ============================================================================
// Global state definitions (declared in cuda_common.cuh)
// ============================================================================
std::unique_ptr<Solver> g_solver;
cusparseHandle_t g_cusparse_handle = nullptr;
cudaStream_t g_stream = nullptr;
int g_last_n = 0;
int g_last_nnz = 0;
int g_call_count = 0;
int g_since_rebuild = 0;

thrust::device_vector<float>* g_d_rhs = nullptr;
thrust::device_vector<float>* g_d_sol = nullptr;

float* g_pinned_rhs = nullptr;
float* g_pinned_sol = nullptr;
int g_pinned_size = 0;

// ============================================================================
// Initialize CUDA
// ============================================================================
void initialize() {
    if (g_cusparse_handle == nullptr) {
        cusparseCreate(&g_cusparse_handle);
        cudaStreamCreate(&g_stream);
        std::cout << "AMGCL-CUDA: SINGLE PRECISION mode (2x faster)" << std::endl;
    }
}

// ============================================================================
// Ensure pinned memory is allocated
// ============================================================================
void ensurePinnedMemory(int n) {
    if (n > g_pinned_size) {
        if (g_pinned_rhs) cudaFreeHost(g_pinned_rhs);
        if (g_pinned_sol) cudaFreeHost(g_pinned_sol);
        cudaMallocHost(&g_pinned_rhs, n * sizeof(float));
        cudaMallocHost(&g_pinned_sol, n * sizeof(float));
        g_pinned_size = n;
    }
}

// ============================================================================
// Cleanup CUDA resources
// ============================================================================
void cleanup() {
    g_solver.reset();
    if (g_d_rhs) { delete g_d_rhs; g_d_rhs = nullptr; }
    if (g_d_sol) { delete g_d_sol; g_d_sol = nullptr; }
    if (g_pinned_rhs) { cudaFreeHost(g_pinned_rhs); g_pinned_rhs = nullptr; }
    if (g_pinned_sol) { cudaFreeHost(g_pinned_sol); g_pinned_sol = nullptr; }
    if (g_stream) { cudaStreamDestroy(g_stream); g_stream = nullptr; }
    if (g_cusparse_handle) {
        cusparseDestroy(g_cusparse_handle);
        g_cusparse_handle = nullptr;
    }
    g_last_n = 0;
    g_last_nnz = 0;
    g_call_count = 0;
    g_since_rebuild = 0;
    g_pinned_size = 0;
}

// ============================================================================
// Solve Ax = b on GPU (SINGLE PRECISION)
// ============================================================================
int solve(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    std::vector<float>& solution,
    float tolerance,
    int max_iterations)
{
    if (n == 0) return 0;
    
    g_call_count++;
    g_since_rebuild++;
    
    // Initialize CUDA resources
    initialize();
    ensurePinnedMemory(n);
    
    int nnz = static_cast<int>(values.size());
    bool structure_changed = (n != g_last_n) || (nnz != g_last_nnz);
    bool need_full_setup = structure_changed || !g_solver;
    bool need_rebuild = !need_full_setup && (g_since_rebuild >= REBUILD_INTERVAL);
    
    // Matrix tuple
    auto A = std::tie(n, row_ptr, col_idx, values);
    
    // Backend params
    Backend::params bprm;
    bprm.cusparse_handle = g_cusparse_handle;
    
    // Setup or rebuild AMG hierarchy
    if (need_full_setup) {
        Solver::params prm;
        prm.solver.tol = tolerance;
        prm.solver.maxiter = max_iterations;
        prm.precond.allow_rebuild = true;
        
        g_solver = std::make_unique<Solver>(A, prm, bprm);
        g_last_n = n;
        g_last_nnz = nnz;
        g_since_rebuild = 0;
        
        if (g_d_rhs) delete g_d_rhs;
        if (g_d_sol) delete g_d_sol;
        g_d_rhs = new thrust::device_vector<float>(n);
        g_d_sol = new thrust::device_vector<float>(n);
        cudaDeviceSynchronize();
        
    } else if (need_rebuild) {
        g_solver->precond().rebuild(A, bprm);
        cudaDeviceSynchronize();
        g_since_rebuild = 0;
    }
    
    // Copy to GPU (pinned memory for speed)
    std::memcpy(g_pinned_rhs, rhs.data(), n * sizeof(float));
    std::memcpy(g_pinned_sol, solution.data(), n * sizeof(float));
    
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_d_rhs->data()),
                    g_pinned_rhs, n * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_d_sol->data()),
                    g_pinned_sol, n * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream);
    cudaStreamSynchronize(g_stream);
    
    // Solve on GPU
    int iters;
    double resid;
    std::tie(iters, resid) = (*g_solver)(*g_d_rhs, *g_d_sol);
    cudaDeviceSynchronize();
    
    // Copy back from GPU
    cudaMemcpyAsync(g_pinned_sol,
                    thrust::raw_pointer_cast(g_d_sol->data()),
                    n * sizeof(float),
                    cudaMemcpyDeviceToHost, g_stream);
    cudaStreamSynchronize(g_stream);
    std::memcpy(solution.data(), g_pinned_sol, n * sizeof(float));
    
    return iters;
}

} // namespace amgcl_cuda
