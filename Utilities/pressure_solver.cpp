// ============================================================================
// File: Utilities/pressure_solver.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Assembles and solves the pressure-correction (Poisson) equation for the
//   SIMPLE algorithm using parallelized solvers.
//
// PRESSURE CORRECTION EQUATION:
//   The pressure correction p' is obtained from the continuity equation:
//
//   ∑_faces (ρ * d * A * ∇p') = ∑_faces (ρ * u* * A)  [mass imbalance]
//
//   Discretized on the staggered grid:
//   a_P * p'_P = a_E * p'_E + a_W * p'_W + a_N * p'_N + a_S * p'_S + b
//
// SOLUTION METHODS:
//
//   1. AMGCL SOLVER (pressureSolverType = 2, FASTEST):
//      - Algebraic Multigrid preconditioned Conjugate Gradient
//      - HIGHLY PARALLELIZED with OpenMP backend
//      - Best for large systems (>10k DOFs)
//      - Compile with -DUSE_AMGCL flag to enable
//
//   2. PARALLEL CG SOLVER (pressureSolverType = 1):
//      - Conjugate Gradient with Jacobi preconditioner
//      - FULLY PARALLELIZED: SpMV, dot products, vector ops all use OpenMP
//      - Good balance of simplicity and performance
//
//   3. ITERATIVE SOR (pressureSolverType = 0):
//      - Successive Over-Relaxation with red-black ordering
//      - Partially parallelized, slowest option
//
// PARALLELIZATION:
//   - SpMV (sparse matrix-vector): Parallel row-wise computation
//   - Dot products: Parallel reduction
//   - Vector updates: Parallel AXPY operations
//   - Matrix assembly: Thread-local triplet storage
//
// ============================================================================
#include "SIMPLE.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <omp.h>
#include <vector>
#include <tuple>
#include <Eigen/IterativeLinearSolvers>

// AMGCL includes (enabled with -DUSE_AMGCL compile flag)
#ifdef USE_AMGCL
#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/solver/bicgstab.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/relaxation/gauss_seidel.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>
#endif

// AMGCL CUDA GPU solver (enabled with -DUSE_AMGCL_CUDA compile flag)
#ifdef USE_AMGCL_CUDA
#include "solvers/gpu/cuda_solver.hpp"
#endif


#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

// =========================================================================
// Persistent CSR and AMGCL solver for CPU (Mimics GPU optimizations)
// =========================================================================
namespace {
    struct PersistentCSR_CPU {
        bool initialized = false;
        int n = 0;
        std::vector<int> row_ptr;
        std::vector<int> col_idx;
        std::vector<float> values;
        std::vector<float> rhs;
        std::vector<float> sol;
    };

    static PersistentCSR_CPU g_cpu_csr;

#ifdef USE_AMGCL
    typedef amgcl::backend::builtin<float> BackendCPU;
    typedef amgcl::make_solver<
        amgcl::amg<
            BackendCPU,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
        >,
        amgcl::solver::cg<BackendCPU>
    > SolverCPU;

    static std::unique_ptr<SolverCPU> g_cpu_amg_solver;
    static int g_since_rebuild_cpu = 0;
    static const int CPU_REBUILD_INTERVAL = 15;
#endif

    // Parallel SpMV for CSR format
    void parallelSpMV_CSR(int n, const std::vector<int>& row_ptr, const std::vector<int>& col_idx, 
                          const std::vector<float>& values, const std::vector<float>& x, std::vector<float>& y) {
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            float sum = 0.0f;
            for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                sum += values[k] * x[col_idx[k]];
            }
            y[i] = sum;
        }
    }

    // Parallel PCG for CSR format
    int solvePCG_CSR(int n, const std::vector<int>& row_ptr, const std::vector<int>& col_idx, 
                     const std::vector<float>& values, const std::vector<float>& b, std::vector<float>& x,
                     float tol, int maxIter) {
        if (n == 0) return 0;
        std::vector<float> r(n), z(n), p(n), Ap(n), diag(n);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            float d = 1.0f;
            for (int k = row_ptr[i]; k < row_ptr[i+1]; ++k) {
                if (col_idx[k] == i) { d = values[k]; break; }
            }
            diag[i] = d;
        }

        parallelSpMV_CSR(n, row_ptr, col_idx, values, x, Ap);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) r[i] = b[i] - Ap[i];

        auto applyJac = [&](const std::vector<float>& rv, std::vector<float>& zv) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) zv[i] = (std::abs(diag[i]) > 1e-20f) ? rv[i] / diag[i] : rv[i];
        };

        applyJac(r, z);
        p = z;
        float rz = 0.0f;
        #pragma omp parallel for reduction(+:rz)
        for (int i = 0; i < n; ++i) rz += r[i] * z[i];

        int iter = 0;
        for (; iter < maxIter; ++iter) {
            parallelSpMV_CSR(n, row_ptr, col_idx, values, p, Ap);
            float pAp = 0.0f;
            #pragma omp parallel for reduction(+:pAp)
            for (int i = 0; i < n; ++i) pAp += p[i] * Ap[i];

            if (std::abs(pAp) < 1e-20f) break;
            float alpha = rz / pAp;

            float rz_new = 0.0f;
            #pragma omp parallel for reduction(+:rz_new)
            for (int i = 0; i < n; ++i) {
                x[i] += alpha * p[i];
                r[i] -= alpha * Ap[i];
            }

            float err = 0.0f;
            #pragma omp parallel for reduction(+:err)
            for (int i = 0; i < n; ++i) err += r[i] * r[i];
            if (std::sqrt(err) < tol) break;

            applyJac(r, z);
            float rz_prev = rz;
            rz = 0.0f;
            #pragma omp parallel for reduction(+:rz)
            for (int i = 0; i < n; ++i) rz += r[i] * z[i];

            float beta = rz / rz_prev;
            #pragma omp parallel for
            for (int i = 0; i < n; ++i) p[i] = z[i] + beta * p[i];
        }
        return iter;
    }
}

// ============================================================================
// Parallel Conjugate Gradient with Jacobi Preconditioner
// ============================================================================
// This is much faster than Eigen's serial SimplicialLDLT for large systems
// because all operations (SpMV, dot products, AXPY) are parallelized.
// ============================================================================
namespace {

// Parallel dot product
inline float parallelDot(const Eigen::VectorXf& a, const Eigen::VectorXf& b) {
    float result = 0.0f;
    const int n = static_cast<int>(a.size());
    #pragma omp parallel for reduction(+:result) schedule(static)
    for (int i = 0; i < n; ++i) {
        result += a(i) * b(i);
    }
    return result;
}

// Parallel AXPY: y = y + alpha * x
inline void parallelAXPY(Eigen::VectorXf& y, float alpha, const Eigen::VectorXf& x) {
    const int n = static_cast<int>(y.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        y(i) += alpha * x(i);
    }
}

// Parallel sparse matrix-vector product: y = A * x
inline void parallelSpMV(const Eigen::SparseMatrix<float>& A, 
                         const Eigen::VectorXf& x, 
                         Eigen::VectorXf& y) {
    const int n = static_cast<int>(A.rows());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        float sum = 0.0f;
        for (Eigen::SparseMatrix<float>::InnerIterator it(A, i); it; ++it) {
            sum += it.value() * x(it.index());
        }
        y(i) = sum;
    }
}

// Parallel Jacobi preconditioner application: z = M^{-1} * r (M = diag(A))
inline void applyJacobiPrecond(const Eigen::VectorXf& diag, 
                                const Eigen::VectorXf& r, 
                                Eigen::VectorXf& z) {
    const int n = static_cast<int>(r.size());
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        z(i) = (diag(i) > 1e-20f) ? r(i) / diag(i) : r(i);
    }
}

// Preconditioned Conjugate Gradient solver (fully parallelized)
// Returns number of iterations, solution in x
int parallelPCG(const Eigen::SparseMatrix<float>& A,
                const Eigen::VectorXf& b,
                Eigen::VectorXf& x,
                float tol = 1e-6f,
                int maxIter = 500) {
    const int n = static_cast<int>(b.size());
    if (n == 0) return 0;
    
    // Extract diagonal for Jacobi preconditioner
    Eigen::VectorXf diag(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        diag(i) = A.coeff(i, i);
    }
    
    // Initialize
    Eigen::VectorXf r(n), z(n), p(n), Ap(n);
    
    // r = b - A*x
    parallelSpMV(A, x, Ap);  // Ap = A*x temporarily
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        r(i) = b(i) - Ap(i);
    }
    
    // z = M^{-1} * r
    applyJacobiPrecond(diag, r, z);
    
    // p = z
    p = z;
    
    float rz = parallelDot(r, z);
    float rz_old;
    float b_norm = std::sqrt(parallelDot(b, b));
    if (b_norm < 1e-20f) b_norm = 1.0f;
    
    int iter;
    for (iter = 0; iter < maxIter; ++iter) {
        // Ap = A * p
        parallelSpMV(A, p, Ap);
        
        float pAp = parallelDot(p, Ap);
        if (std::abs(pAp) < 1e-30f) break;
        
        float alpha = rz / pAp;
        
        // x = x + alpha * p
        parallelAXPY(x, alpha, p);
        
        // r = r - alpha * Ap
        parallelAXPY(r, -alpha, Ap);
        
        float r_norm = std::sqrt(parallelDot(r, r));
        if (r_norm / b_norm < tol) {
            iter++;
            break;
        }
        
        // z = M^{-1} * r
        applyJacobiPrecond(diag, r, z);
        
        rz_old = rz;
        rz = parallelDot(r, z);
        
        float beta = rz / (rz_old + 1e-30f);
        
        // p = z + beta * p
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            p(i) = z(i) + beta * p(i);
        }
    }
    
    return iter;
}

// ============================================================================
// AMGCL Solver (Algebraic Multigrid Preconditioned CG)
// ============================================================================
// This is the FASTEST option for large pressure systems.
// AMG provides optimal O(n) complexity for Poisson-like problems.
// ============================================================================
#ifdef USE_AMGCL

// CSR matrix structure for AMGCL
struct CSRMatrix {
    int n = 0;
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
};

// Convert Eigen sparse matrix to CSR format for AMGCL
CSRMatrix eigenToCSR(const Eigen::SparseMatrix<float>& A) {
    CSRMatrix csr;
    csr.n = static_cast<int>(A.rows());
    csr.row_ptr.resize(csr.n + 1);
    csr.col_idx.reserve(A.nonZeros());
    csr.values.reserve(A.nonZeros());
    
    csr.row_ptr[0] = 0;
    for (int i = 0; i < csr.n; ++i) {
        for (Eigen::SparseMatrix<float>::InnerIterator it(A, i); it; ++it) {
            csr.col_idx.push_back(static_cast<int>(it.index()));
            csr.values.push_back(it.value());
        }
        csr.row_ptr[i + 1] = static_cast<int>(csr.col_idx.size());
    }
    return csr;
}

// AMGCL solver wrapper
// Returns number of iterations, solution in x
int solveWithAMGCL(const Eigen::SparseMatrix<float>& A,
                   const Eigen::VectorXf& b,
                   Eigen::VectorXf& x,
                   float tol = 1e-6f,
                   int maxIter = 500) {
    const int n = static_cast<int>(b.size());
    if (n == 0) return 0;
    
    // Convert to CSR
    CSRMatrix csr = eigenToCSR(A);
    
    // Define AMGCL solver type: AMG-preconditioned CG
    typedef amgcl::backend::builtin<float> Backend;
    typedef amgcl::make_solver<
        amgcl::amg<
            Backend,
            amgcl::coarsening::smoothed_aggregation,
            amgcl::relaxation::spai0
        >,
        amgcl::solver::cg<Backend>
    > Solver;
    
    // Set solver parameters
    Solver::params prm;
    prm.solver.tol = tol;
    prm.solver.maxiter = maxIter;
    
    // Build preconditioner and solver (this is the expensive part, done once)
    // Use tuple adapter: AMGCL expects (n, row_ptr, col_idx, values) containers
    auto A_amgcl = std::tie(n, csr.row_ptr, csr.col_idx, csr.values);

    Solver solver(A_amgcl, prm);
    
    // Convert Eigen vectors to std::vector for AMGCL
    std::vector<float> rhs(n), sol(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        rhs[i] = b(i);
        sol[i] = x(i);  // Use previous solution as initial guess
    }
    
    // Solve
    int iters;
    float resid;
    std::tie(iters, resid) = solver(rhs, sol);
    
    // Copy solution back to Eigen vector
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        x(i) = sol[i];
    }
    
    return iters;
}

#endif // USE_AMGCL

} // anonymous namespace

// ============================================================================
// Anonymous namespace for file-local helper structures
// ============================================================================
namespace {

// ---------------------------------------------------------------------------
// PressureMapping: Maps 2D grid indices to linear system indices
// ---------------------------------------------------------------------------
// Since we only solve for FLUID cells, we need a mapping from grid (i,j)
// to linear index n (0 to nFluidCells-1) and vice versa.
struct PressureMapping {
    std::vector<std::pair<int, int>> gridToLinear;  // linear index → (i, j) grid coords
    std::vector<std::vector<int>> linearIndex;      // (i, j) → linear index (-1 if solid)
    int nFluidCells = 0;                            // Total number of fluid DOFs
};

// ---------------------------------------------------------------------------
// buildPressureMapping: Create the grid ↔ linear index mapping
// ---------------------------------------------------------------------------
// For topology optimization with adjoint method: include ALL interior cells.
// This ensures consistent matrix size and smooth sensitivities ∂J/∂γ.
// Cells with small d-coefficients will have weak pressure coupling but must
// still be included for proper gradient computation via adjoint equations.
PressureMapping buildPressureMapping(SIMPLE& s) {
    PressureMapping map;
    map.linearIndex.assign(s.M + 1, std::vector<int>(s.N + 1, -1));
    int n = 0;
    
    // Iterate over interior cells (excluding ghost layers)
    // Include ALL cells for topology optimization consistency
    for (int i = 1; i < s.M; ++i) {
        for (int j = 1; j < s.N - 1; ++j) {
            if (s.checkBoundaries(i, j) == 1.0f) continue;
            
            // Include ALL interior cells (no exclusion based on d-coefficients)
            // This is required for adjoint-based sensitivity computation
            map.linearIndex[i][j] = n;
            map.gridToLinear.push_back({i, j});
            n++;
        }
    }
    map.nFluidCells = n;
    return map;
}

} // namespace

bool SIMPLE::solvePressureSystem(int& pressureIterations, float& localResidMass) {
    pStar.setZero();
    
    // MSVC OpenMP doesn't support reduction on reference variables, so use local copy
    float residMassLocal = 0.0f;
    double residMassSumSq = 0.0;
    long long residMassCount = 0;

    // Calculate mass residual (all interior cells)
    {
        ScopedTimer t("Pres: Mass Residual");
        #pragma omp parallel for reduction(max:residMassLocal) reduction(+:residMassSumSq,residMassCount) schedule(guided)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                if (checkBoundaries(i, j) == 1.0f) continue;

                float massE = rho * uStar(i, j) * hy;
                float massW = rho * uStar(i, j - 1) * hy;
                float massN = rho * vStar(i, j) * hx;
                float massS = rho * vStar(i - 1, j) * hx;
                float b = std::abs(massW - massE + massS - massN);
                if (b > residMassLocal) residMassLocal = b;
                residMassSumSq += static_cast<double>(b) * static_cast<double>(b);
                residMassCount++;
            }
        }
    }
    localResidMass = residMassLocal;
    residMass_max = residMassLocal;
    if (residMassCount > 0) {
        residMass_RMS = static_cast<float>(std::sqrt(residMassSumSq / static_cast<double>(residMassCount)));
    } else {
        residMass_RMS = 0.0f;
    }
    residMass = localResidMass;

    bool directSolverSucceeded = false;
    if (pressureSolverType >= 1) {  // 1=Parallel CG, 2=AMGCL
        // For topology optimization: mapping is constant (all interior cells included)
        // Cache it based on grid dimensions only
        static PressureMapping mapping;
        static int cachedM = -1;
        static int cachedN = -1;

        if (cachedM != M || cachedN != N || mapping.gridToLinear.empty()) {
            mapping = buildPressureMapping(*this);
            cachedM = M;
            cachedN = N;
        }

        const int nFluidCells = mapping.nFluidCells;

#ifdef USE_AMGCL_CUDA
        // ===================================================================
        // FAST PATH: Uses FLOAT directly - NO CONVERSION OVERHEAD!
        // ===================================================================
        static bool gpu_structure_built = false;
        if (pressureSolverType == 4 && nFluidCells > 0) {
            // GPU path using FLOAT directly for maximum speed
            static std::vector<int> row_ptr, col_idx;
            static std::vector<float> values_f, rhs_f, sol_f;
            static int cached_n = 0;
            
            if (!gpu_structure_built || cached_n != nFluidCells) {
                ScopedTimer t("PresGPU: Initial Structure Builder");
                // ===== FIRST CALL: Build CSR structure (once only) =====
                std::vector<Eigen::Triplet<float>> triplets;
                triplets.reserve(nFluidCells * 5);
                
                for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                    int i = mapping.gridToLinear[linIdx].first;
                    int j = mapping.gridToLinear[linIdx].second;
                    
                    float aE_p = (j < N - 1) ? rho * dE(i, j) * hy : 0.0f;
                    float aW_p = (j > 1) ? rho * dE(i, j - 1) * hy : 0.0f;
                    float aN_p = (i < M - 1) ? rho * dN(i, j) * hx : 0.0f;
                    float aS_p = (i > 1) ? rho * dN(i - 1, j) * hx : 0.0f;
                    float aP_p = aE_p + aW_p + aN_p + aS_p;
                    if (aP_p < 1e-20f) aP_p = 1.0f;
                    
                    triplets.push_back(Eigen::Triplet<float>(linIdx, linIdx, aP_p));
                    if (j < N - 1) {
                        int nIdx = mapping.linearIndex[i][j + 1];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aE_p));
                    }
                    if (j > 1) {
                        int nIdx = mapping.linearIndex[i][j - 1];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aW_p));
                    }
                    if (i < M - 1) {
                        int nIdx = mapping.linearIndex[i + 1][j];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aN_p));
                    }
                    if (i > 1) {
                        int nIdx = mapping.linearIndex[i - 1][j];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aS_p));
                    }
                }
                
                Eigen::SparseMatrix<float> A_temp(nFluidCells, nFluidCells);
                A_temp.setFromTriplets(triplets.begin(), triplets.end());
                Eigen::SparseMatrix<float, Eigen::RowMajor> A_row(A_temp);
                
                const int nnz = static_cast<int>(A_row.nonZeros());
                row_ptr.resize(nFluidCells + 1);
                col_idx.resize(nnz);
                values_f.resize(nnz);
                rhs_f.resize(nFluidCells);
                sol_f.resize(nFluidCells);
                
                std::memcpy(row_ptr.data(), A_row.outerIndexPtr(), (nFluidCells + 1) * sizeof(int));
                std::memcpy(col_idx.data(), A_row.innerIndexPtr(), nnz * sizeof(int));
                const float* vals = A_row.valuePtr();
                std::memcpy(values_f.data(), vals, nnz * sizeof(float));
                
                cached_n = nFluidCells;
                gpu_structure_built = true;
            } else {
                ScopedTimer t("PresGPU: Assembly (Robust)");
                // ===== FAST PATH: Update values directly as FLOAT =====
                // Removed precomputed indices as they caused a regression.
                // Reverting to robust loop-based update.
                #pragma omp parallel for schedule(static)
                for (int row = 0; row < nFluidCells; ++row) {
                    int i = mapping.gridToLinear[row].first;
                    int j = mapping.gridToLinear[row].second;
                    
                    float aE_p = (j < N - 1) ? rho * dE(i, j) * hy : 0.0f;
                    float aW_p = (j > 1) ? rho * dE(i, j - 1) * hy : 0.0f;
                    float aN_p = (i < M - 1) ? rho * dN(i, j) * hx : 0.0f;
                    float aS_p = (i > 1) ? rho * dN(i - 1, j) * hx : 0.0f;
                    float aP_p = aE_p + aW_p + aN_p + aS_p;
                    if (aP_p < 1e-20f) aP_p = 1.0f;
                    
                    for (int k = row_ptr[row]; k < row_ptr[row + 1]; ++k) {
                        int col = col_idx[k];
                        if (col == row) {
                            values_f[k] = aP_p;
                        } else {
                            int ni = mapping.gridToLinear[col].first;
                            int nj = mapping.gridToLinear[col].second;
                            if (ni == i && nj == j + 1) values_f[k] = -aE_p;
                            else if (ni == i && nj == j - 1) values_f[k] = -aW_p;
                            else if (ni == i + 1 && nj == j) values_f[k] = -aN_p;
                            else if (ni == i - 1 && nj == j) values_f[k] = -aS_p;
                        }
                    }
                }
            }
            
            // Update RHS as FLOAT
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < nFluidCells; ++row) {
                int i = mapping.gridToLinear[row].first;
                int j = mapping.gridToLinear[row].second;
                float massE = rho * uStar(i, j) * hy;
                float massW = rho * uStar(i, j - 1) * hy;
                float massN = rho * vStar(i, j) * hx;
                float massS = rho * vStar(i - 1, j) * hx;
                rhs_f[row] = massW - massE + massS - massN;
                sol_f[row] = pStar(i, j);
            }
            
            // GPU Solve with FLOAT interface (NO CONVERSION!)
            {
                ScopedTimer t("PresGPU: Solve");
                pressureIterations = amgcl_cuda::solve(nFluidCells, row_ptr, col_idx, values_f,
                                                             rhs_f, sol_f, pTol, maxPressureIter);
            }
            
            // Scatter back
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < nFluidCells; ++row) {
                int i = mapping.gridToLinear[row].first;
                int j = mapping.gridToLinear[row].second;
                pStar(i, j) = sol_f[row];
            }
            for (int i = 1; i < M; ++i) pStar(i, N - 1) = 0.0f;
            
            directSolverSucceeded = true;
        }
        else
#endif
        if (nFluidCells > 0) {
            // ===================================================================
            // CPU FAST PATH: Persistent CSR + Optimized Assembly
            // ===================================================================
            if (!g_cpu_csr.initialized || g_cpu_csr.n != nFluidCells) {
                ScopedTimer t("PresCPU: Initial Structure Builder");
                std::vector<Eigen::Triplet<float>> triplets;
                triplets.reserve(nFluidCells * 5);
                
                for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                    int i = mapping.gridToLinear[linIdx].first;
                    int j = mapping.gridToLinear[linIdx].second;
                    
                    float aE_p = (j < N - 1) ? rho * dE(i, j) * hy : 0.0f;
                    float aW_p = (j > 1) ? rho * dE(i, j - 1) * hy : 0.0f;
                    float aN_p = (i < M - 1) ? rho * dN(i, j) * hx : 0.0f;
                    float aS_p = (i > 1) ? rho * dN(i - 1, j) * hx : 0.0f;
                    float aP_p = aE_p + aW_p + aN_p + aS_p;
                    if (aP_p < 1e-20f) aP_p = 1.0f;
                    
                    triplets.push_back(Eigen::Triplet<float>(linIdx, linIdx, aP_p));
                    if (j < N - 1) {
                        int nIdx = mapping.linearIndex[i][j + 1];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aE_p));
                    }
                    if (j > 1) {
                        int nIdx = mapping.linearIndex[i][j - 1];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aW_p));
                    }
                    if (i < M - 1) {
                        int nIdx = mapping.linearIndex[i + 1][j];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aN_p));
                    }
                    if (i > 1) {
                        int nIdx = mapping.linearIndex[i - 1][j];
                        if (nIdx >= 0) triplets.push_back(Eigen::Triplet<float>(linIdx, nIdx, -aS_p));
                    }
                }
                
                Eigen::SparseMatrix<float> A_temp(nFluidCells, nFluidCells);
                A_temp.setFromTriplets(triplets.begin(), triplets.end());
                Eigen::SparseMatrix<float, Eigen::RowMajor> A_row(A_temp);
                
                const int nnz = static_cast<int>(A_row.nonZeros());
                g_cpu_csr.n = nFluidCells;
                g_cpu_csr.row_ptr.resize(nFluidCells + 1);
                g_cpu_csr.col_idx.resize(nnz);
                g_cpu_csr.values.resize(nnz);
                g_cpu_csr.rhs.resize(nFluidCells);
                g_cpu_csr.sol.resize(nFluidCells);
                
                std::memcpy(g_cpu_csr.row_ptr.data(), A_row.outerIndexPtr(), (nFluidCells + 1) * sizeof(int));
                std::memcpy(g_cpu_csr.col_idx.data(), A_row.innerIndexPtr(), nnz * sizeof(int));
                std::memcpy(g_cpu_csr.values.data(), A_row.valuePtr(), nnz * sizeof(float));
                
                g_cpu_csr.initialized = true;
            } else {
                ScopedTimer t("PresCPU: Assembly (Fast)");
                #pragma omp parallel for schedule(static)
                for (int row = 0; row < nFluidCells; ++row) {
                    int i = mapping.gridToLinear[row].first;
                    int j = mapping.gridToLinear[row].second;
                    
                    float aE_p = (j < N - 1) ? rho * dE(i, j) * hy : 0.0f;
                    float aW_p = (j > 1) ? rho * dE(i, j - 1) * hy : 0.0f;
                    float aN_p = (i < M - 1) ? rho * dN(i, j) * hx : 0.0f;
                    float aS_p = (i > 1) ? rho * dN(i - 1, j) * hx : 0.0f;
                    float aP_p = aE_p + aW_p + aN_p + aS_p;
                    if (aP_p < 1e-20f) aP_p = 1.0f;
                    
                    for (int k = g_cpu_csr.row_ptr[row]; k < g_cpu_csr.row_ptr[row + 1]; ++k) {
                        int col = g_cpu_csr.col_idx[k];
                        if (col == row) {
                            g_cpu_csr.values[k] = aP_p;
                        } else {
                            int ni = mapping.gridToLinear[col].first;
                            int nj = mapping.gridToLinear[col].second;
                            if (ni == i && nj == j + 1) g_cpu_csr.values[k] = -aE_p;
                            else if (ni == i && nj == j - 1) g_cpu_csr.values[k] = -aW_p;
                            else if (ni == i + 1 && nj == j) g_cpu_csr.values[k] = -aN_p;
                            else if (ni == i - 1 && nj == j) g_cpu_csr.values[k] = -aS_p;
                        }
                    }
                }
            }
            
            // Update RHS and Sol
            #pragma omp parallel for schedule(static)
            for (int row = 0; row < nFluidCells; ++row) {
                int i = mapping.gridToLinear[row].first;
                int j = mapping.gridToLinear[row].second;
                float massE = rho * uStar(i, j) * hy;
                float massW = rho * uStar(i, j - 1) * hy;
                float massN = rho * vStar(i, j) * hx;
                float massS = rho * vStar(i - 1, j) * hx;
                g_cpu_csr.rhs[row] = massW - massE + massS - massN;
                g_cpu_csr.sol[row] = pStar(i, j);
            }
            
            // -------------------------------------------------------------------
            // CPU Solver Selection
            // -------------------------------------------------------------------
#ifdef USE_AMGCL
            if (pressureSolverType == 2) {
                ScopedTimer t("PresCPU: AMGCL Solve");
                
                // AMGCL Hierarchy is sensitive to values, so we rebuild periodically (lagging)
                if (!g_cpu_amg_solver || g_since_rebuild_cpu >= CPU_REBUILD_INTERVAL) {
                    ScopedTimer t2("PresCPU: AMGCL Precond Rebuild");
                    auto A_amgcl = std::tie(nFluidCells, g_cpu_csr.row_ptr, g_cpu_csr.col_idx, g_cpu_csr.values);
                    SolverCPU::params prm;
                    prm.solver.tol = pTol;
                    prm.solver.maxiter = maxPressureIter;
                    g_cpu_amg_solver = std::make_unique<SolverCPU>(A_amgcl, prm);
                    g_since_rebuild_cpu = 0;
                }
                
                float resid;
                std::tie(pressureIterations, resid) = (*g_cpu_amg_solver)(g_cpu_csr.rhs, g_cpu_csr.sol);
                g_since_rebuild_cpu++;
                directSolverSucceeded = true;
            }
            else
#endif
            if (pressureSolverType == 1) {
                ScopedTimer t("PresCPU: Parallel CG Solve");
                pressureIterations = solvePCG_CSR(nFluidCells, g_cpu_csr.row_ptr, g_cpu_csr.col_idx, 
                                                 g_cpu_csr.values, g_cpu_csr.rhs, g_cpu_csr.sol, pTol, maxPressureIter);
                directSolverSucceeded = true;
            }
            else if (pressureSolverType == 3) {
                ScopedTimer t("PresCPU: Direct Solve (LDLT)");
                // LDLT still uses Eigen Matrix - copy values across
                static Eigen::SparseMatrix<float> A_eigen(nFluidCells, nFluidCells);
                static bool first_ldlt = true;
                if (first_ldlt) {
                   A_eigen.resize(nFluidCells, nFluidCells);
                   first_ldlt = false;
                }
                
                // Copy values into A_eigen (this is slightly slow but LDLT is for small systems)
                std::vector<Eigen::Triplet<float>> triplets;
                triplets.reserve(g_cpu_csr.values.size());
                for (int row = 0; row < nFluidCells; ++row) {
                    for (int k = g_cpu_csr.row_ptr[row]; k < g_cpu_csr.row_ptr[row + 1]; ++k) {
                        triplets.push_back(Eigen::Triplet<float>(row, g_cpu_csr.col_idx[k], g_cpu_csr.values[k]));
                    }
                }
                A_eigen.setFromTriplets(triplets.begin(), triplets.end());
                
                static Eigen::SimplicialLDLT<Eigen::SparseMatrix<float>> ldltSolver;
                ldltSolver.compute(A_eigen);
                if (ldltSolver.info() == Eigen::Success) {
                    Eigen::Map<Eigen::VectorXf> rhs_map(g_cpu_csr.rhs.data(), nFluidCells);
                    Eigen::VectorXf pCorr = ldltSolver.solve(rhs_map);
                    for (int i=0; i<nFluidCells; ++i) g_cpu_csr.sol[i] = pCorr(i);
                    pressureIterations = 1;
                    directSolverSucceeded = true;
                }
            }
            
            if (directSolverSucceeded) {
                // Scatter back
                #pragma omp parallel for schedule(static)
                for (int row = 0; row < nFluidCells; ++row) {
                    int i = mapping.gridToLinear[row].first;
                    int j = mapping.gridToLinear[row].second;
                    pStar(i, j) = g_cpu_csr.sol[row];
                }
                for (int i = 1; i < M; ++i) pStar(i, N - 1) = 0.0f;
            }
        }
    }

    if (!directSolverSucceeded) {
        float omega = sorOmega;
        if (omega <= 0.0f || omega >= 2.0f) {
            int maxDim = std::max(M, N);
            float spectralRadius = std::cos(M_PI / maxDim);
            omega = 2.0f / (1.0f + std::sqrt(1.0f - spectralRadius * spectralRadius));
            omega = std::max(1.0f, std::min(1.95f, omega));
        }

        pressureIterations = 0;
        for (int pIter = 0; pIter < maxPressureIter; ++pIter) {
            float maxChange = 0.0f;
            pressureIterations = pIter + 1;

            #pragma omp parallel for reduction(max:maxChange) schedule(guided)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    if ((i + j) % 2 != 0) continue;
                    if (checkBoundaries(i, j) == 1.0f) {
                        pStar(i, j) = 0.0f;
                        continue;
                    }

                    float aE_p = (j < N - 1 ) ? rho * dE(i, j) * hy : 0.0f;
                    float aW_p = (j > 1 ) ? rho * dE(i, j - 1) * hy : 0.0f;
                    float aN_p = (i < M - 1 ) ? rho * dN(i, j) * hx : 0.0f;
                    float aS_p = (i > 1 ) ? rho * dN(i - 1, j) * hx : 0.0f;

                    float aP_p = aE_p + aW_p + aN_p + aS_p;

                    if (j == N - 1 || aP_p < 1e-20f) {
                        pStar(i, j) = 0.0f;
                        continue;
                    }

                    float massE = rho * uStar(i, j) * hy;
                    float massW = rho * uStar(i, j - 1) * hy;
                    float massN = rho * vStar(i, j) * hx;
                    float massS = rho * vStar(i - 1, j) * hx;
                    float b_rhs = massW - massE + massS - massN;

                    float pE = (j < N - 1) ? pStar(i, j + 1) : 0.0f;
                    float pW = (j > 1) ? pStar(i, j - 1) : 0.0f;
                    float pN = (i < M - 1) ? pStar(i + 1, j) : 0.0f;
                    float pS = (i > 1) ? pStar(i - 1, j) : 0.0f;

                    float pGS = (aE_p * pE + aW_p * pW + aN_p * pN + aS_p * pS + b_rhs) / aP_p;
                    float pNew = pStar(i, j) + omega * (pGS - pStar(i, j));
                    pNew = std::max(-10000.0f, std::min(10000.0f, pNew));

                    float change = std::abs(pNew - pStar(i, j));
                    if (change > maxChange) maxChange = change;
                    pStar(i, j) = pNew;
                }
            }

            #pragma omp parallel for reduction(max:maxChange) schedule(guided)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    if ((i + j) % 2 != 1) continue;
                    if (checkBoundaries(i, j) == 1.0f) {
                        pStar(i, j) = 0.0f;
                        continue;
                    }

                    float aE_p = (j < N - 1 ) ? rho * dE(i, j) * hy : 0.0f;
                    float aW_p = (j > 1 ) ? rho * dE(i, j - 1) * hy : 0.0f;
                    float aN_p = (i < M - 1 ) ? rho * dN(i, j) * hx : 0.0f;
                    float aS_p = (i > 1 ) ? rho * dN(i - 1, j) * hx : 0.0f;

                    float aP_p = aE_p + aW_p + aN_p + aS_p;

                    if (j == N - 1 || aP_p < 1e-20f) {
                        pStar(i, j) = 0.0f;
                        continue;
                    }

                    float massE = rho * uStar(i, j) * hy;
                    float massW = rho * uStar(i, j - 1) * hy;
                    float massN = rho * vStar(i, j) * hx;
                    float massS = rho * vStar(i - 1, j) * hx;
                    float b_rhs = massW - massE + massS - massN;

                    float pE = (j < N - 1) ? pStar(i, j + 1) : 0.0f;
                    float pW = (j > 1) ? pStar(i, j - 1) : 0.0f;
                    float pN = (i < M - 1) ? pStar(i + 1, j) : 0.0f;
                    float pS = (i > 1) ? pStar(i - 1, j) : 0.0f;

                    float pGS = (aE_p * pE + aW_p * pW + aN_p * pN + aS_p * pS + b_rhs) / aP_p;
                    float pNew = pStar(i, j) + omega * (pGS - pStar(i, j));
                    pNew = std::max(-10000.0f, std::min(10000.0f, pNew));

                    float change = std::abs(pNew - pStar(i, j));
                    if (change > maxChange) maxChange = change;
                    pStar(i, j) = pNew;
                }
            }

            if (maxChange < pTol) break;
        }
    }

    return true;
}
