// ============================================================================
// File: Utilities/solvers/gpu/momentum_solver.cu
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   GPU momentum solvers (U and V) using cuSPARSE Jacobi iteration.
//   Single precision (float) only for maximum GPU speed.
// ============================================================================

#include "cuda_common.cuh"
#include "cuda_solver.hpp"
#include <thrust/reduce.h>

namespace amgcl_cuda {

// ============================================================================
// CUDA Kernel: Jacobi iteration step
// ============================================================================
__global__ void jacobiUpdateKernel(
    const float* __restrict__ b,
    const float* __restrict__ Ax,
    const float* __restrict__ diag,
    float* __restrict__ x,
    float* __restrict__ residual,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float r = b[i] - Ax[i];
        residual[i] = r * r;  // Squared residual for reduction
        x[i] += r / diag[i];  // Jacobi update
    }
}

// ============================================================================
// CUDA Kernel: Extract diagonal from CSR matrix
// ============================================================================
__global__ void extractDiagonalKernel(
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ values,
    float* __restrict__ diag,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        for (int j = row_ptr[i]; j < row_ptr[i + 1]; ++j) {
            if (col_idx[j] == i) {
                diag[i] = values[j];
                return;
            }
        }
        diag[i] = 1.0f;  // Fallback
    }
}

// ============================================================================
// Momentum U Solver state
// ============================================================================
static cudaStream_t g_stream_u = nullptr;
static cusparseHandle_t g_cusparse_u = nullptr;

static thrust::device_vector<float>* g_momentum_u_rhs = nullptr;
static thrust::device_vector<float>* g_momentum_u_sol = nullptr;
static thrust::device_vector<float>* g_momentum_u_tmp = nullptr;
static thrust::device_vector<float>* g_momentum_u_diag = nullptr;
static thrust::device_vector<int>* g_momentum_u_row_ptr = nullptr;
static thrust::device_vector<int>* g_momentum_u_col_idx = nullptr;
static thrust::device_vector<float>* g_momentum_u_values = nullptr;
static int g_momentum_u_n = 0;

static cusparseSpMatDescr_t g_u_mat_descr = nullptr;
static cusparseDnVecDescr_t g_u_vec_x = nullptr, g_u_vec_y = nullptr;
static void* g_u_spmv_buffer = nullptr;
static size_t g_u_buffer_size = 0;

// ============================================================================
// Momentum V Solver state
// ============================================================================
static cudaStream_t g_stream_v = nullptr;
static cusparseHandle_t g_cusparse_v = nullptr;

static thrust::device_vector<float>* g_momentum_v_rhs = nullptr;
static thrust::device_vector<float>* g_momentum_v_sol = nullptr;
static thrust::device_vector<float>* g_momentum_v_tmp = nullptr;
static thrust::device_vector<float>* g_momentum_v_diag = nullptr;
static thrust::device_vector<int>* g_momentum_v_row_ptr = nullptr;
static thrust::device_vector<int>* g_momentum_v_col_idx = nullptr;
static thrust::device_vector<float>* g_momentum_v_values = nullptr;
static int g_momentum_v_n = 0;

static cusparseSpMatDescr_t g_v_mat_descr = nullptr;
static cusparseDnVecDescr_t g_v_vec_x = nullptr, g_v_vec_y = nullptr;
static void* g_v_spmv_buffer = nullptr;
static size_t g_v_buffer_size = 0;

// ============================================================================
// Solve U-momentum on GPU (Jacobi iteration)
// ============================================================================
int solveMomentumU(
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
    
    initialize();
    ensurePinnedMemory(n);
    
    // Initialize CUDA stream and cuSPARSE handle for U-momentum
    if (!g_stream_u) {
        cudaStreamCreate(&g_stream_u);
        cusparseCreate(&g_cusparse_u);
        cusparseSetStream(g_cusparse_u, g_stream_u);
    }
    
    int nnz = static_cast<int>(values.size());
    
    // Allocate GPU vectors if size changed
    bool size_changed = (n != g_momentum_u_n);
    if (size_changed) {
        if (g_momentum_u_rhs) delete g_momentum_u_rhs;
        if (g_momentum_u_sol) delete g_momentum_u_sol;
        if (g_momentum_u_tmp) delete g_momentum_u_tmp;
        if (g_momentum_u_diag) delete g_momentum_u_diag;
        if (g_momentum_u_row_ptr) delete g_momentum_u_row_ptr;
        if (g_momentum_u_col_idx) delete g_momentum_u_col_idx;
        if (g_momentum_u_values) delete g_momentum_u_values;
        
        g_momentum_u_rhs = new thrust::device_vector<float>(n);
        g_momentum_u_sol = new thrust::device_vector<float>(n);
        g_momentum_u_tmp = new thrust::device_vector<float>(n);
        g_momentum_u_diag = new thrust::device_vector<float>(n);
        g_momentum_u_row_ptr = new thrust::device_vector<int>(n + 1);
        g_momentum_u_col_idx = new thrust::device_vector<int>(nnz);
        g_momentum_u_values = new thrust::device_vector<float>(nnz);
        
        g_momentum_u_n = n;
        
        if (g_u_mat_descr) cusparseDestroySpMat(g_u_mat_descr);
        if (g_u_vec_x) cusparseDestroyDnVec(g_u_vec_x);
        if (g_u_vec_y) cusparseDestroyDnVec(g_u_vec_y);
        if (g_u_spmv_buffer) cudaFree(g_u_spmv_buffer);
        g_u_mat_descr = nullptr;
        g_u_vec_x = nullptr;
        g_u_vec_y = nullptr;
        g_u_spmv_buffer = nullptr;
    }
    
    // Copy matrix to GPU
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_u_row_ptr->data()),
                    row_ptr.data(), (n + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, g_stream_u);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_u_col_idx->data()),
                    col_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, g_stream_u);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_u_values->data()),
                    values.data(), nnz * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream_u);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_u_rhs->data()),
                    rhs.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream_u);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_u_sol->data()),
                    solution.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream_u);
    cudaStreamSynchronize(g_stream_u);
    
    // Extract diagonal
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    extractDiagonalKernel<<<numBlocks, blockSize, 0, g_stream_u>>>(
        thrust::raw_pointer_cast(g_momentum_u_row_ptr->data()),
        thrust::raw_pointer_cast(g_momentum_u_col_idx->data()),
        thrust::raw_pointer_cast(g_momentum_u_values->data()),
        thrust::raw_pointer_cast(g_momentum_u_diag->data()),
        n);
    
    // Create cuSPARSE descriptors if needed
    if (!g_u_mat_descr) {
        cusparseCreateCsr(&g_u_mat_descr, n, n, nnz,
            thrust::raw_pointer_cast(g_momentum_u_row_ptr->data()),
            thrust::raw_pointer_cast(g_momentum_u_col_idx->data()),
            thrust::raw_pointer_cast(g_momentum_u_values->data()),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        
        cusparseCreateDnVec(&g_u_vec_x, n,
            thrust::raw_pointer_cast(g_momentum_u_sol->data()), CUDA_R_32F);
        cusparseCreateDnVec(&g_u_vec_y, n,
            thrust::raw_pointer_cast(g_momentum_u_tmp->data()), CUDA_R_32F);
        
        float alpha = 1.0f, beta = 0.0f;
        cusparseSpMV_bufferSize(g_cusparse_u, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, g_u_mat_descr, g_u_vec_x, &beta, g_u_vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &g_u_buffer_size);
        cudaMalloc(&g_u_spmv_buffer, g_u_buffer_size);
    } else {
        cusparseCsrSetPointers(g_u_mat_descr,
            thrust::raw_pointer_cast(g_momentum_u_row_ptr->data()),
            thrust::raw_pointer_cast(g_momentum_u_col_idx->data()),
            thrust::raw_pointer_cast(g_momentum_u_values->data()));
        cusparseDnVecSetValues(g_u_vec_x, thrust::raw_pointer_cast(g_momentum_u_sol->data()));
        cusparseDnVecSetValues(g_u_vec_y, thrust::raw_pointer_cast(g_momentum_u_tmp->data()));
    }
    
    // Jacobi iteration
    float alpha = 1.0f, beta = 0.0f;
    float tol_sq = tolerance * tolerance;
    int iters = 0;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        cusparseSpMV(g_cusparse_u, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, g_u_mat_descr, g_u_vec_x, &beta, g_u_vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, g_u_spmv_buffer);
        
        jacobiUpdateKernel<<<numBlocks, blockSize, 0, g_stream_u>>>(
            thrust::raw_pointer_cast(g_momentum_u_rhs->data()),
            thrust::raw_pointer_cast(g_momentum_u_tmp->data()),
            thrust::raw_pointer_cast(g_momentum_u_diag->data()),
            thrust::raw_pointer_cast(g_momentum_u_sol->data()),
            thrust::raw_pointer_cast(g_momentum_u_tmp->data()),
            n);
        
        iters = iter + 1;
        
        if ((iter + 1) % 10 == 0) {
            float resid_sq = thrust::reduce(g_momentum_u_tmp->begin(), g_momentum_u_tmp->end(), 0.0f);
            if (resid_sq < tol_sq * n) break;
        }
    }
    
    cudaStreamSynchronize(g_stream_u);
    
    // Copy solution back
    cudaMemcpyAsync(solution.data(),
                    thrust::raw_pointer_cast(g_momentum_u_sol->data()),
                    n * sizeof(float),
                    cudaMemcpyDeviceToHost, g_stream_u);
    cudaStreamSynchronize(g_stream_u);
    
    return iters;
}

// ============================================================================
// Solve V-momentum on GPU (Jacobi iteration)
// ============================================================================
int solveMomentumV(
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
    
    initialize();
    ensurePinnedMemory(n);
    
    // Initialize CUDA stream and cuSPARSE handle for V-momentum
    if (!g_stream_v) {
        cudaStreamCreate(&g_stream_v);
        cusparseCreate(&g_cusparse_v);
        cusparseSetStream(g_cusparse_v, g_stream_v);
    }
    
    int nnz = static_cast<int>(values.size());
    
    // Allocate GPU vectors if size changed
    bool size_changed = (n != g_momentum_v_n);
    if (size_changed) {
        if (g_momentum_v_rhs) delete g_momentum_v_rhs;
        if (g_momentum_v_sol) delete g_momentum_v_sol;
        if (g_momentum_v_tmp) delete g_momentum_v_tmp;
        if (g_momentum_v_diag) delete g_momentum_v_diag;
        if (g_momentum_v_row_ptr) delete g_momentum_v_row_ptr;
        if (g_momentum_v_col_idx) delete g_momentum_v_col_idx;
        if (g_momentum_v_values) delete g_momentum_v_values;
        
        g_momentum_v_rhs = new thrust::device_vector<float>(n);
        g_momentum_v_sol = new thrust::device_vector<float>(n);
        g_momentum_v_tmp = new thrust::device_vector<float>(n);
        g_momentum_v_diag = new thrust::device_vector<float>(n);
        g_momentum_v_row_ptr = new thrust::device_vector<int>(n + 1);
        g_momentum_v_col_idx = new thrust::device_vector<int>(nnz);
        g_momentum_v_values = new thrust::device_vector<float>(nnz);
        
        g_momentum_v_n = n;
        
        if (g_v_mat_descr) cusparseDestroySpMat(g_v_mat_descr);
        if (g_v_vec_x) cusparseDestroyDnVec(g_v_vec_x);
        if (g_v_vec_y) cusparseDestroyDnVec(g_v_vec_y);
        if (g_v_spmv_buffer) cudaFree(g_v_spmv_buffer);
        g_v_mat_descr = nullptr;
        g_v_vec_x = nullptr;
        g_v_vec_y = nullptr;
        g_v_spmv_buffer = nullptr;
    }
    
    // Copy matrix to GPU
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_v_row_ptr->data()),
                    row_ptr.data(), (n + 1) * sizeof(int),
                    cudaMemcpyHostToDevice, g_stream_v);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_v_col_idx->data()),
                    col_idx.data(), nnz * sizeof(int),
                    cudaMemcpyHostToDevice, g_stream_v);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_v_values->data()),
                    values.data(), nnz * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream_v);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_v_rhs->data()),
                    rhs.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream_v);
    cudaMemcpyAsync(thrust::raw_pointer_cast(g_momentum_v_sol->data()),
                    solution.data(), n * sizeof(float),
                    cudaMemcpyHostToDevice, g_stream_v);
    cudaStreamSynchronize(g_stream_v);
    
    // Extract diagonal
    int blockSize = 256;
    int numBlocks = (n + blockSize - 1) / blockSize;
    extractDiagonalKernel<<<numBlocks, blockSize, 0, g_stream_v>>>(
        thrust::raw_pointer_cast(g_momentum_v_row_ptr->data()),
        thrust::raw_pointer_cast(g_momentum_v_col_idx->data()),
        thrust::raw_pointer_cast(g_momentum_v_values->data()),
        thrust::raw_pointer_cast(g_momentum_v_diag->data()),
        n);
    
    // Create cuSPARSE descriptors if needed
    if (!g_v_mat_descr) {
        cusparseCreateCsr(&g_v_mat_descr, n, n, nnz,
            thrust::raw_pointer_cast(g_momentum_v_row_ptr->data()),
            thrust::raw_pointer_cast(g_momentum_v_col_idx->data()),
            thrust::raw_pointer_cast(g_momentum_v_values->data()),
            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
        
        cusparseCreateDnVec(&g_v_vec_x, n,
            thrust::raw_pointer_cast(g_momentum_v_sol->data()), CUDA_R_32F);
        cusparseCreateDnVec(&g_v_vec_y, n,
            thrust::raw_pointer_cast(g_momentum_v_tmp->data()), CUDA_R_32F);
        
        float alpha = 1.0f, beta = 0.0f;
        cusparseSpMV_bufferSize(g_cusparse_v, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, g_v_mat_descr, g_v_vec_x, &beta, g_v_vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &g_v_buffer_size);
        cudaMalloc(&g_v_spmv_buffer, g_v_buffer_size);
    } else {
        cusparseCsrSetPointers(g_v_mat_descr,
            thrust::raw_pointer_cast(g_momentum_v_row_ptr->data()),
            thrust::raw_pointer_cast(g_momentum_v_col_idx->data()),
            thrust::raw_pointer_cast(g_momentum_v_values->data()));
        cusparseDnVecSetValues(g_v_vec_x, thrust::raw_pointer_cast(g_momentum_v_sol->data()));
        cusparseDnVecSetValues(g_v_vec_y, thrust::raw_pointer_cast(g_momentum_v_tmp->data()));
    }
    
    // Jacobi iteration
    float alpha = 1.0f, beta = 0.0f;
    float tol_sq = tolerance * tolerance;
    int iters = 0;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        cusparseSpMV(g_cusparse_v, CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha, g_v_mat_descr, g_v_vec_x, &beta, g_v_vec_y,
            CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, g_v_spmv_buffer);
        
        jacobiUpdateKernel<<<numBlocks, blockSize, 0, g_stream_v>>>(
            thrust::raw_pointer_cast(g_momentum_v_rhs->data()),
            thrust::raw_pointer_cast(g_momentum_v_tmp->data()),
            thrust::raw_pointer_cast(g_momentum_v_diag->data()),
            thrust::raw_pointer_cast(g_momentum_v_sol->data()),
            thrust::raw_pointer_cast(g_momentum_v_tmp->data()),
            n);
        
        iters = iter + 1;
        
        if ((iter + 1) % 10 == 0) {
            float resid_sq = thrust::reduce(g_momentum_v_tmp->begin(), g_momentum_v_tmp->end(), 0.0f);
            if (resid_sq < tol_sq * n) break;
        }
    }
    
    cudaStreamSynchronize(g_stream_v);
    
    // Copy solution back
    cudaMemcpyAsync(solution.data(),
                    thrust::raw_pointer_cast(g_momentum_v_sol->data()),
                    n * sizeof(float),
                    cudaMemcpyDeviceToHost, g_stream_v);
    cudaStreamSynchronize(g_stream_v);
    
    return iters;
}

} // namespace amgcl_cuda
