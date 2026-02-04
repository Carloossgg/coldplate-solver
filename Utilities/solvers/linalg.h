// ============================================================================
// File: Utilities/solvers/linalg.h
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Common linear algebra primitives for sparse matrix operations.
//   Provides parallel implementations using OpenMP for CPU solvers.
// ============================================================================

#ifndef SOLVERS_LINALG_H
#define SOLVERS_LINALG_H

#include <vector>
#include <cmath>
#include <omp.h>

namespace linalg {

// ---------------------------------------------------------------------------
// PersistentCSR: Compressed Sparse Row matrix storage
// ---------------------------------------------------------------------------
struct PersistentCSR {
    bool initialized = false;
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    std::vector<float> rhs;
    std::vector<float> sol;
};

// ---------------------------------------------------------------------------
// Parallel SpMV for CSR format: y = A * x
// ---------------------------------------------------------------------------
inline void parallelSpMV_CSR(
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& x,
    std::vector<float>& y)
{
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
            sum += values[k] * x[col_idx[k]];
        }
        y[i] = sum;
    }
}

// ---------------------------------------------------------------------------
// Parallel dot product: result = a · b
// ---------------------------------------------------------------------------
inline float parallelDot(const std::vector<float>& a, const std::vector<float>& b) {
    float result = 0.0f;
    const int n = static_cast<int>(a.size());
    #pragma omp parallel for reduction(+:result) schedule(static)
    for (int i = 0; i < n; ++i) {
        result += a[i] * b[i];
    }
    return result;
}

// ---------------------------------------------------------------------------
// Parallel AXPY: y = y + alpha * x
// ---------------------------------------------------------------------------
inline void parallelAXPY(std::vector<float>& y, float alpha, const std::vector<float>& x) {
    const int n = static_cast<int>(y.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        y[i] += alpha * x[i];
    }
}

// ---------------------------------------------------------------------------
// Parallel vector subtraction: r = b - Ax (given Ax already computed)
// ---------------------------------------------------------------------------
inline void parallelResidual(
    const std::vector<float>& b,
    const std::vector<float>& Ax,
    std::vector<float>& r)
{
    const int n = static_cast<int>(b.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        r[i] = b[i] - Ax[i];
    }
}

// ---------------------------------------------------------------------------
// Parallel L2 norm: ||x||_2
// ---------------------------------------------------------------------------
inline float parallelNorm(const std::vector<float>& x) {
    return std::sqrt(parallelDot(x, x));
}

} // namespace linalg

#endif // SOLVERS_LINALG_H
