// ============================================================================
// File: Utilities/momentum_solver.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Momentum equation solver with persistent CSR structure.
//   CPU: OpenMP Jacobi | GPU: cuSPARSE Jacobi
// ============================================================================

#include "SIMPLE.h"
#include <algorithm>
#include <cmath>
#include <vector>
#include <omp.h>

#ifdef USE_AMGCL_CUDA
#include "solvers/gpu/cuda_solver.hpp"
#endif

#include "solvers/cpu/jacobi_solver.h"

namespace {

// ============================================================================
// Momentum Grid Mapping
// ============================================================================
struct MomentumMapping {
    std::vector<std::pair<int, int>> gridToLinear;
    std::vector<std::vector<int>> linearIndex;
    int nDOFs = 0;
};

// ============================================================================
// Persistent CSR Structure
// ============================================================================
struct PersistentCSR {
    bool initialized = false;
    int nDOFs = 0;
    std::vector<int> row_ptr, col_idx;
    std::vector<float> values, rhs, sol;
    std::vector<int> diagOffset, eastOffset, westOffset, northOffset, southOffset;
};

// Global persistent structures
static MomentumMapping g_uMap, g_vMap;
static PersistentCSR g_uCSR, g_vCSR;
static int g_cachedM = -1, g_cachedN = -1;

// ============================================================================
// Build mapping for U or V velocity DOFs
// ============================================================================
MomentumMapping buildMapping(const SIMPLE& s, bool isU) {
    MomentumMapping map;
    if (isU) {
        map.linearIndex.assign(s.M + 1, std::vector<int>(s.N, -1));
        for (int i = 1; i < s.M; ++i) {
            for (int j = 1; j < s.N - 1; ++j) {
                if (s.checkBoundaries(i, j) == 1.0) continue;
                map.linearIndex[i][j] = map.nDOFs++;
                map.gridToLinear.push_back({i, j});
            }
        }
    } else {
        map.linearIndex.assign(s.M, std::vector<int>(s.N + 1, -1));
        for (int i = 1; i < s.M - 1; ++i) {
            for (int j = 1; j < s.N; ++j) {
                if (s.checkBoundaries(i, j) == 1.0) continue;
                map.linearIndex[i][j] = map.nDOFs++;
                map.gridToLinear.push_back({i, j});
            }
        }
    }
    return map;
}

// ============================================================================
// Build CSR sparsity pattern
// ============================================================================
void buildCSRStructure(const SIMPLE& s, const MomentumMapping& map, PersistentCSR& csr, bool isU) {
    const int nDOFs = map.nDOFs;
    csr.nDOFs = nDOFs;
    csr.row_ptr.resize(nDOFs + 1);
    csr.diagOffset.assign(nDOFs, -1);
    csr.eastOffset.assign(nDOFs, -1);
    csr.westOffset.assign(nDOFs, -1);
    csr.northOffset.assign(nDOFs, -1);
    csr.southOffset.assign(nDOFs, -1);
    
    std::vector<std::pair<int, int>> entries;
    int nnz = 0;
    
    for (int row = 0; row < nDOFs; ++row) {
        int i = map.gridToLinear[row].first;
        int j = map.gridToLinear[row].second;
        csr.row_ptr[row] = nnz;
        entries.clear();
        
        // Neighbor offsets matching iterations.cpp: N=i+1, S=i-1, W=j-1, E=j+1
        int di[] = {1, -1, 0, 0, 0}; // N, S, W, E, diag
        int dj[] = {0, 0, -1, 1, 0};
        int types[] = {3, 4, 2, 1, 0};  // N=3, S=4, W=2, E=1, diag=0
        
        for (int k = 0; k < 5; ++k) {
            int ni = i + di[k], nj = j + dj[k];
            bool inBounds = isU ? (ni >= 0 && ni < s.M + 1 && nj >= 0 && nj < s.N)
                                : (ni >= 0 && ni < s.M && nj >= 0 && nj < s.N + 1);
            if (inBounds && (k == 4 || map.linearIndex[ni][nj] >= 0)) {
                int col = (k == 4) ? row : map.linearIndex[ni][nj];
                entries.push_back({col, types[k]});
            }
        }
        
        std::sort(entries.begin(), entries.end());
        for (size_t e = 0; e < entries.size(); ++e) {
            int* offset = nullptr;
            switch (entries[e].second) {
                case 0: offset = &csr.diagOffset[row]; break;
                case 1: offset = &csr.eastOffset[row]; break;
                case 2: offset = &csr.westOffset[row]; break;
                case 3: offset = &csr.northOffset[row]; break;
                case 4: offset = &csr.southOffset[row]; break;
            }
            if (offset) *offset = nnz + static_cast<int>(e);
        }
        nnz += static_cast<int>(entries.size());
    }
    
    csr.row_ptr[nDOFs] = nnz;
    csr.col_idx.resize(nnz);
    csr.values.resize(nnz, 0.0f);
    csr.rhs.resize(nDOFs, 0.0f);
    csr.sol.resize(nDOFs, 0.0f);
    
    // Fill column indices
    for (int row = 0; row < nDOFs; ++row) {
        int i = map.gridToLinear[row].first;
        int j = map.gridToLinear[row].second;
        int offset = csr.row_ptr[row];
        entries.clear();
        
        int di[] = {1, -1, 0, 0, 0}; // N, S, W, E, diag
        int dj[] = {0, 0, -1, 1, 0};
        
        for (int k = 0; k < 5; ++k) {
            int ni = i + di[k], nj = j + dj[k];
            bool inBounds = isU ? (ni >= 0 && ni < s.M + 1 && nj >= 0 && nj < s.N)
                                : (ni >= 0 && ni < s.M && nj >= 0 && nj < s.N + 1);
            if (inBounds && (k == 4 || map.linearIndex[ni][nj] >= 0)) {
                int col = (k == 4) ? row : map.linearIndex[ni][nj];
                entries.push_back({col, 0});
            }
        }
        std::sort(entries.begin(), entries.end());
        for (size_t e = 0; e < entries.size(); ++e) {
            csr.col_idx[offset + e] = entries[e].first;
        }
    }
    csr.initialized = true;
}

// ============================================================================
// Generic momentum solver (works for both U and V)
// ============================================================================
template<bool IsU>
int solveMomentumGeneric(SIMPLE& s, float& resid) {
    const float hx = s.hx, hy = s.hy, vol = hx * hy;
    const float rho = s.rho, eta = s.eta;
    
    // Select appropriate structures
    MomentumMapping& map = IsU ? g_uMap : g_vMap;
    PersistentCSR& csr = IsU ? g_uCSR : g_vCSR;
    
    // Initialize if needed
    if (g_cachedM != s.M || g_cachedN != s.N) {
        g_uMap = buildMapping(s, true);
        g_vMap = buildMapping(s, false);
        buildCSRStructure(s, g_uMap, g_uCSR, true);
        buildCSRStructure(s, g_vMap, g_vCSR, false);
        g_cachedM = s.M; g_cachedN = s.N;
    }
    
    const int nDOFs = map.nDOFs;
    if (nDOFs == 0) return 0;
    
    // Constants
    const float maxVel = 3.0f * std::max(std::abs(s.inletVelocity), 0.1f);
    const float sinkCoeff = (s.enableTwoPointFiveD && s.Ht_channel > 0.0f)
                             ? s.twoPointFiveDSinkMultiplier * (12.0f * eta / (s.Ht_channel * s.Ht_channel)) : 0.0f;
    const float sinkDiag = sinkCoeff * vol;
    const float DeScale = hy / hx;
    const float DnScale = hx / hy;
    constexpr float halfCellFluidTol = 0.999f;
    
    // Assembly
    {
        ScopedTimer t(IsU ? "MomU: Assembly" : "MomV: Assembly");
        #pragma omp parallel for schedule(static)
        for (int row = 0; row < nDOFs; ++row) {
            int i = map.gridToLinear[row].first;
            int j = map.gridToLinear[row].second;
            
            // Use centralized interpolation helpers to keep alpha indexing
            // consistent between explicit and implicit momentum paths.
            float alphaLocal = IsU ? alphaAtU(s, i, j) : alphaAtV(s, i, j);
            
            // Face velocities (different for U vs V)
            float ue, uw, vn, vs;
            if constexpr (IsU) {
                ue = std::clamp(0.5f * (s.u(i, j) + s.u(i, j+1)), -maxVel, maxVel);
                uw = std::clamp(0.5f * (s.u(i, j-1) + s.u(i, j)), -maxVel, maxVel);
                vn = std::clamp(0.5f * (s.v(i, j) + s.v(i, j+1)), -maxVel, maxVel);
                vs = std::clamp(0.5f * (s.v(i - 1, j) + s.v(i - 1, j+1)), -maxVel, maxVel);
            } else {
                ue = std::clamp(0.5f * (s.u(i, j) + s.u(i+1, j)), -maxVel, maxVel);
                uw = std::clamp(0.5f * (s.u(i, j-1) + s.u(i+1, j-1)), -maxVel, maxVel);
                vn = std::clamp(0.5f * (s.v(i, j) + s.v(i+1, j)), -maxVel, maxVel);
                vs = std::clamp(0.5f * (s.v(i-1, j) + s.v(i, j)), -maxVel, maxVel);
            }
            
            // Convective fluxes (scaled by user factor when 2.5D is ON)
            // Master switch behavior: when 2.5D is OFF, convection scaling is OFF.
            const float convScale = s.enableTwoPointFiveD ? s.twoPointFiveDConvectionFactor : 1.0f;
            float Fe = convScale * rho * hy * ue, Fw = convScale * rho * hy * uw;
            float Fn = convScale * rho * hx * vn, Fs = convScale * rho * hx * vs;
            
            // Local effective viscosity (base + residual-based artificial viscosity)
            float muLocal = eta;
            if (s.enableResidualViscosity) {
                muLocal += IsU ? muArtAtU(s, i, j) : muArtAtV(s, i, j);
            }
            float De = muLocal * DeScale;
            float Dn = muLocal * DnScale;

            // Upwind coefficients
            float aE = De + std::max(0.0f, -Fe);
            float aW = De + std::max(0.0f, Fw);
            float aN = Dn + std::max(0.0f, -Fn);
            float aS = Dn + std::max(0.0f, Fs);

            // Half-cell wall diffusion on external no-slip boundaries:
            // - U-equation: horizontal walls (top/bottom) affect tangential u
            // - V-equation: vertical walls (left/right) affect tangential v
            if constexpr (IsU) {
                if (hasExternalNoSlipSouthForU(s, i, j, halfCellFluidTol)) aS += Dn;
                if (hasExternalNoSlipNorthForU(s, i, j, halfCellFluidTol)) aN += Dn;
            } else {
                if (hasExternalNoSlipWestForV(s, i, j, halfCellFluidTol)) aW += De;
                if (hasExternalNoSlipEastForV(s, i, j, halfCellFluidTol)) aE += De;
            }
            float sumA = aE + aW + aN + aS;
            
            // Pseudo-transient (fully disabled when enablePseudoTimeStepping=false)
            float transCoeff = 0.0f;
            if (s.enablePseudoTimeStepping) {
                const float speed = IsU ? s.pseudoSpeedAtU(i, j) : s.pseudoSpeedAtV(i, j);
                const float dtLocal = s.computePseudoDtFromSpeed(speed);
                transCoeff = rho * vol / dtLocal;
            }
            
            // Diagonal
            float brinkmanDrag = alphaLocal * vol;
            float aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeff + sinkDiag + brinkmanDrag;
            aP0 = std::max(aP0, 1e-10f);
            float aP = aP0 / s.uvAlpha;
            
            // Source term
            float Sp;
            float velOld, velCur;
            if constexpr (IsU) {
                Sp = (s.p(i, j) - s.p(i, j+1)) * hy;
                velOld = s.uOld(i, j); velCur = s.u(i, j);
            } else {
                Sp = (s.p(i, j) - s.p(i+1, j)) * hx;
                velOld = s.vOld(i, j); velCur = s.v(i, j);
            }
            Sp += transCoeff * velOld;
            Sp += (1.0f - s.uvAlpha) / s.uvAlpha * aP0 * velCur;
            
            // SOU correction
            if (s.convectionScheme == 1) {
                Sp += IsU ? computeSOUCorrectionU(s, i, j, Fe, Fw, Fn, Fs)
                          : computeSOUCorrectionV(s, i, j, Fe, Fw, Fn, Fs);
            }
            
            if constexpr (IsU) {
                s.dE(i, j) = hy / aP0;
            } else {
                s.dN(i, j) = hx / aP0;
            }
            
            // Update matrix (handle boundary contributions)
            auto updateCoeff = [&](int* offset, float coeff, float neighborVal) {
                if (*offset >= 0) csr.values[*offset] = -coeff;
                else Sp += coeff * neighborVal;
            };
            
            if (csr.diagOffset[row] >= 0) csr.values[csr.diagOffset[row]] = aP;
            
            if constexpr (IsU) {
                updateCoeff(&csr.eastOffset[row], aE, s.u(i, j+1));
                updateCoeff(&csr.westOffset[row], aW, s.u(i, j-1));
                updateCoeff(&csr.northOffset[row], aN, s.u(i+1, j));
                updateCoeff(&csr.southOffset[row], aS, s.u(i-1, j));
                csr.sol[row] = s.u(i, j);
            } else {
                updateCoeff(&csr.eastOffset[row], aE, s.v(i, j+1));
                updateCoeff(&csr.westOffset[row], aW, s.v(i, j-1));
                updateCoeff(&csr.northOffset[row], aN, s.v(i+1, j));
                updateCoeff(&csr.southOffset[row], aS, s.v(i-1, j));
                csr.sol[row] = s.v(i, j);
            }
            csr.rhs[row] = Sp;
        }
    }
    
    // Solve
    int iterations = 0;
    {
        ScopedTimer t(IsU ? "MomU: Linear Solve" : "MomV: Linear Solve");
    #ifdef USE_AMGCL_CUDA
        if (s.pressureSolverType == 4) {
            iterations = IsU ? amgcl_cuda::solveMomentumU(nDOFs, csr.row_ptr, csr.col_idx, csr.values,
                                                          csr.rhs, csr.sol, s.momentumTol, s.maxMomentumIter)
                             : amgcl_cuda::solveMomentumV(nDOFs, csr.row_ptr, csr.col_idx, csr.values,
                                                          csr.rhs, csr.sol, s.momentumTol, s.maxMomentumIter);
        } else
    #endif
        {
            iterations = solvers::cpu::solveMomentumJacobi(nDOFs, csr.row_ptr, csr.col_idx, csr.values,
                                                           csr.rhs, csr.sol, csr.diagOffset, s.momentumTol, s.maxMomentumIter);
        }
    }
    
    // Scatter and residual
    float localResid = 0.0f;
    {
        ScopedTimer t(IsU ? "MomU: Scatter/Residual" : "MomV: Scatter/Residual");
        #pragma omp parallel for reduction(max:localResid)
        for (int row = 0; row < nDOFs; ++row) {
            int i = map.gridToLinear[row].first;
            int j = map.gridToLinear[row].second;
            float newVal = std::clamp(csr.sol[row], -maxVel, maxVel);
            if constexpr (IsU) {
                localResid = std::max(localResid, std::abs(newVal - s.u(i, j)));
                s.uStar(i, j) = newVal;
            } else {
                localResid = std::max(localResid, std::abs(newVal - s.v(i, j)));
                s.vStar(i, j) = newVal;
            }
        }
    }
    
    resid = localResid;
    return iterations;
}

} // anonymous namespace

// ============================================================================
// Public API
// ============================================================================
int solveUMomentumImplicit(SIMPLE& s, float& residU) {
    return solveMomentumGeneric<true>(s, residU);
}

int solveVMomentumImplicit(SIMPLE& s, float& residV) {
    return solveMomentumGeneric<false>(s, residV);
}
