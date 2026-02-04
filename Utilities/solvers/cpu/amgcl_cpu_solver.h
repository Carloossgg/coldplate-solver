// ============================================================================
// File: Utilities/solvers/cpu/amgcl_cpu_solver.h
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   AMGCL CPU solver wrapper - Algebraic Multigrid preconditioned CG.
//   Only included when USE_AMGCL is defined at compile time.
// ============================================================================

#ifndef SOLVERS_CPU_AMGCL_CPU_SOLVER_H
#define SOLVERS_CPU_AMGCL_CPU_SOLVER_H

#ifdef USE_AMGCL

#include <vector>
#include <memory>
#include <tuple>

#include <amgcl/make_solver.hpp>
#include <amgcl/solver/cg.hpp>
#include <amgcl/amg.hpp>
#include <amgcl/coarsening/smoothed_aggregation.hpp>
#include <amgcl/relaxation/spai0.hpp>
#include <amgcl/backend/builtin.hpp>
#include <amgcl/adapter/crs_tuple.hpp>

namespace solvers {
namespace cpu {

// Type definitions for AMGCL CPU solver
typedef amgcl::backend::builtin<float> AMGCLBackend;
typedef amgcl::make_solver<
    amgcl::amg<
        AMGCLBackend,
        amgcl::coarsening::smoothed_aggregation,
        amgcl::relaxation::spai0
    >,
    amgcl::solver::cg<AMGCLBackend>
> AMGCLSolver;

// ---------------------------------------------------------------------------
// Persistent AMGCL solver state (for preconditioner reuse/lagging)
// ---------------------------------------------------------------------------
struct AMGCLState {
    std::unique_ptr<AMGCLSolver> solver;
    int since_rebuild = 0;
    static const int REBUILD_INTERVAL = 15;
    
    void reset() {
        solver.reset();
        since_rebuild = 0;
    }
};

// ---------------------------------------------------------------------------
// Solve Ax = b using AMGCL (AMG-preconditioned CG)
// Returns: number of iterations performed
// ---------------------------------------------------------------------------
inline int solveAMGCL(
    AMGCLState& state,
    int n,
    const std::vector<int>& row_ptr,
    const std::vector<int>& col_idx,
    const std::vector<float>& values,
    const std::vector<float>& rhs,
    std::vector<float>& sol,
    float tol = 1e-6f,
    int maxIter = 500)
{
    if (n == 0) return 0;
    
    state.since_rebuild++;
    
    auto A = std::tie(n, row_ptr, col_idx, values);
    
    // Rebuild solver if needed
    if (!state.solver || state.since_rebuild >= AMGCLState::REBUILD_INTERVAL) {
        AMGCLSolver::params prm;
        prm.solver.tol = tol;
        prm.solver.maxiter = maxIter;
        state.solver = std::make_unique<AMGCLSolver>(A, prm);
        state.since_rebuild = 0;
    }
    
    int iters;
    float resid;
    std::tie(iters, resid) = (*state.solver)(rhs, sol);
    
    return iters;
}

} // namespace cpu
} // namespace solvers

#endif // USE_AMGCL

#endif // SOLVERS_CPU_AMGCL_CPU_SOLVER_H
