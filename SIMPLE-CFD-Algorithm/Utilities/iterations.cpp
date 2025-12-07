// File: Utilities/iterations.cpp
// Author: Peter Tcherkezian
// Description: Orchestrates a full SIMPLE/SIMPLEC iteration on the staggered grid:
//   - assembles/solves U and V momentum with under-relaxation, optional SOU deferred correction, and Brinkman masking
//   - updates pseudo-time/CFL ramps and transient residual tracking for convergence diagnostics
//   - performs pressure correction (direct or SOR) and velocity correction
//   - applies boundary conditions, updates residuals, returns pressure-iteration count for logging
#include "SIMPLE.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

double SIMPLE::calculateStep(int& pressureIterations)
{
    const double vol = hx * hy;
    const bool pseudoActive = enablePseudoTimeStepping;
    
    // Thread-local max residuals
    double localResidU = 0.0;
    double localResidV = 0.0;
    double localResidMass = 0.0;
    
    // Transient term residual tracking
    double localTransResidU = 0.0;
    double localTransResidV = 0.0;

    // Pseudo-transient for stability
    // Use cell size for CFL calculation (this is correct for stability)
    // The mesh-independence comes from running to true steady state (TransRes → 0)
    const double hChar = std::min(hx, hy);
    
    double uDtMin = timeStep, uDtMax = 0.0, uDtSum = 0.0;
    double vDtMin = timeStep, vDtMax = 0.0, vDtSum = 0.0;
    long long uDtCount = 0;
    long long vDtCount = 0;

    auto updateStats = [&](double dt, double& minDt, double& maxDt, double& sumDt, long long& count) {
        minDt = std::min(minDt, dt);
        maxDt = std::max(maxDt, dt);
        sumDt += dt;
        count++;
    };

    // Compute pseudo-dt using cell size for CFL (standard approach)
    auto computeLocalDt = [&](double normalVel) -> double {
        if (!pseudoActive) return std::numeric_limits<double>::infinity();
        if (!useLocalPseudoTime) return timeStep;
        double speed = std::max(std::abs(normalVel), minPseudoSpeed);
        double dtCfl = pseudoCFL * hChar / speed;
        return std::min(timeStep, dtCfl);
    };

    // Save old velocities
    uOld = u;
    vOld = v;

    // Maximum velocity for bounds checking (tighter for stability)
    const double maxVel = 3.0 * std::max(std::abs(inletVelocity), 0.1);
    
    // Second-order upwind (SOU) scheme flag
    const bool useSOU = (convectionScheme == 1);

    // =========================================================================
    // STEP 1: U-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    #pragma omp parallel for collapse(2) reduction(max:localResidU,localTransResidU) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            
            // Fast mask check instead of checkBoundaries + isSolidCell
            if (!fluidU(*this, i, j) || checkBoundaries(i, j) == 1.0) {
                uStar(i, j) = 0.0;
                dE(i, j) = 0.0;
                continue;
            }

            double De = eta * hy / hx;
            double Dn = eta * hx / hy;

            // Face velocities for flux calculation
            double ue = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j) + u(i, j + 1))));
            double uw = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j - 1) + u(i, j))));
            double vn = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i - 1, j) + v(i - 1, j + 1))));
            double vs = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i, j) + v(i, j + 1))));

            double Fe = rho * hy * ue;
            double Fw = rho * hy * uw;
            double Fn = rho * hx * vn;
            double Fs = rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            double aE = De + std::max(0.0, -Fe);
            double aW = De + std::max(0.0, Fw);
            double aN = Dn + std::max(0.0, -Fn);
            double aS = Dn + std::max(0.0, Fs);

            double Sdc = useSOU ? computeSOUCorrectionU(*this, i, j, Fe, Fw, Fn, Fs) : 0.0;

            double sumA = aE + aW + aN + aS;
            double dtLocal = computeLocalDt(u(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, uDtMin, uDtMax, uDtSum, uDtCount);
            }
            double transCoeffLocal = rho * vol / dtLocal;

            double aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal;
            aP0 = std::max(aP0, 1e-10);

            // SIMPLE: dDenom = aP0 + sumA (neglects neighbor contributions)
            // SIMPLEC: dDenom = aP0 - sumA (accounts for neighbor contributions consistently)
            double dDenom = useSIMPLEC ? std::max(aP0 - sumA, 1e-12) : std::max(aP0 + sumA, 1e-12);
            dE(i, j) = hy / dDenom;

            double Sp = (p(i, j) - p(i, j + 1)) * hy;
            Sp += transCoeffLocal * uOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            double aP = aP0 / uvAlpha;
            Sp += (1.0 - uvAlpha) / uvAlpha * aP0 * u(i, j);

            double uNew = (aE * u(i, j + 1) + aW * u(i, j - 1) +
                          aN * u(i - 1, j) + aS * u(i + 1, j) + Sp) / aP;

            uNew = std::max(-maxVel, std::min(maxVel, uNew));

            double diff = std::abs(uNew - u(i, j));
            if (diff > localResidU) localResidU = diff;
            
            // Track transient term residual: |transCoeff * (uNew - u)|
            // At true steady state, this should be ~0
            // This measures how much the pseudo-transient term is affecting the solution
            double transTermU = transCoeffLocal * diff;
            if (transTermU > localTransResidU) localTransResidU = transTermU;
            
            uStar(i, j) = uNew;
        }
    }
    residU = localResidU;
    transientResidU = localTransResidU;

    // =========================================================================
    // STEP 2: V-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    localResidV = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(max:localResidV,localTransResidV) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            
            // Fast mask check instead of checkBoundaries + isSolidCell
            if (!fluidV(*this, i, j) || checkBoundaries(i, j) == 1.0) {
                vStar(i, j) = 0.0;
                dN(i, j) = 0.0;
                continue;
            }

            double De = eta * hy / hx;
            double Dn = eta * hx / hy;

            // Face velocities for flux calculation
            double ue = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j) + u(i + 1, j))));
            double uw = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j - 1) + u(i + 1, j - 1))));
            double vn = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i, j) + v(i + 1, j))));
            double vs = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i - 1, j) + v(i, j))));

            double Fe = rho * hy * ue;
            double Fw = rho * hy * uw;
            double Fn = rho * hx * vn;
            double Fs = rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            double aE = De + std::max(0.0, -Fe);
            double aW = De + std::max(0.0, Fw);
            double aN = Dn + std::max(0.0, -Fn);
            double aS = Dn + std::max(0.0, Fs);

            double Sdc = useSOU ? computeSOUCorrectionV(*this, i, j, Fe, Fw, Fn, Fs) : 0.0;

            double sumA = aE + aW + aN + aS;
            double dtLocal = computeLocalDt(v(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, vDtMin, vDtMax, vDtSum, vDtCount);
            }
            double transCoeffLocal = rho * vol / dtLocal;

            double aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal;
            aP0 = std::max(aP0, 1e-10);

            // SIMPLE: dDenom = aP0 + sumA (neglects neighbor contributions)
            // SIMPLEC: dDenom = aP0 - sumA (accounts for neighbor contributions consistently)
            double dDenom = useSIMPLEC ? std::max(aP0 - sumA, 1e-12) : std::max(aP0 + sumA, 1e-12);
            dN(i, j) = hx / dDenom;

            double Sp = (p(i, j) - p(i + 1, j)) * hx;
            Sp += transCoeffLocal * vOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            double aP = aP0 / uvAlpha;
            Sp += (1.0 - uvAlpha) / uvAlpha * aP0 * v(i, j);

            double vNew = (aE * v(i, j + 1) + aW * v(i, j - 1) +
                          aN * v(i + 1, j) + aS * v(i - 1, j) + Sp) / aP;

            vNew = std::max(-maxVel, std::min(maxVel, vNew));

            double diff = std::abs(vNew - v(i, j));
            if (diff > localResidV) localResidV = diff;
            
            // Track transient term residual: |transCoeff * (vNew - v)|
            double transTermV = transCoeffLocal * diff;
            if (transTermV > localTransResidV) localTransResidV = transTermV;
            
            vStar(i, j) = vNew;
        }
    }
    residV = localResidV;
    transientResidV = localTransResidV;

    setVelocityBoundaryConditions(uStar, vStar);

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && uDtCount > 0) {
        pseudoStatsU.min = uDtMin;
        pseudoStatsU.max = uDtMax;
        pseudoStatsU.avg = uDtSum / double(uDtCount);
        pseudoStatsU.samples = uDtCount;
        pseudoStatsU.valid = true;
    } else {
        pseudoStatsU.valid = false;
    }

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && vDtCount > 0) {
        pseudoStatsV.min = vDtMin;
        pseudoStatsV.max = vDtMax;
        pseudoStatsV.avg = vDtSum / double(vDtCount);
        pseudoStatsV.samples = vDtCount;
        pseudoStatsV.valid = true;
    } else {
        pseudoStatsV.valid = false;
    }

    // =========================================================================
    // STEP 3: PRESSURE CORRECTION (Iterative SOR or Direct Sparse Solver)
    // =========================================================================
    bool directSolverSucceeded = solvePressureSystem(pressureIterations, localResidMass);
    residMass = localResidMass;

    // =========================================================================
    // STEP 4: CORRECT PRESSURE (Parallelized with precomputed masks)
    // =========================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            if (fluidP(*this, i, j)) {
                p(i, j) += pAlpha * pStar(i, j);
            }
        }
    }

    // Set pressure boundary conditions (outlet reference pressure = 0.0)
    // No normalization needed - boundary condition handles reference pressure
    setPressureBoundaryConditions(p);

    // =========================================================================
    // STEP 5: CORRECT VELOCITIES (Parallelized)
    // =========================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            if (checkBoundaries(i, j) == 1.0) continue;
            if (alphaAtU(*this, i, j) > 1e6) continue;
            
            double du = dE(i, j) * (pStar(i, j) - pStar(i, j + 1));
            du = std::max(-maxVel, std::min(maxVel, du));
            u(i, j) = uStar(i, j) + du;
            u(i, j) = std::max(-maxVel, std::min(maxVel, u(i, j)));
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            if (checkBoundaries(i, j) == 1.0) continue;
            if (alphaAtV(*this, i, j) > 1e6) continue;
            
            double dv = dN(i, j) * (pStar(i, j) - pStar(i + 1, j));
            dv = std::max(-maxVel, std::min(maxVel, dv));
            v(i, j) = vStar(i, j) + dv;
            v(i, j) = std::max(-maxVel, std::min(maxVel, v(i, j)));
        }
    }

    setVelocityBoundaryConditions(u, v);

    // =========================================================================
    // STEP 6: MASS BALANCE CORRECTION (Serial - small loop)
    // =========================================================================
    double massIn = 0.0, massOut = 0.0;
    for (int i = 1; i < M; ++i) {
        massIn += rho * u(i, 0) * hy;
        massOut += rho * u(i, N - 1) * hy;
    }

    if (std::abs(massOut) > 1e-10 && std::abs(massIn) > 1e-10) {
        double ratio = massIn / massOut;
        if (ratio > 0.5 && ratio < 2.0) {
            for (int i = 1; i < M; ++i) {
                u(i, N - 1) *= ratio;
            }
        }
    }

    return std::max({residU, residV, residMass});
}
