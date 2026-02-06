// ============================================================================
// File: Utilities/time_control.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Implements CFL ramping and pseudo-time step statistics logging for
//   convergence acceleration in steady-state CFD simulations.
//
// CFL RAMPING:
//   The pseudo-CFL number controls the size of the pseudo-time step:
//     Δt = CFL * Δx / |u|
//
//   Starting with a low CFL (conservative) ensures stability at startup.
//   As residuals decrease, we can safely increase CFL for faster convergence.
//
//   Ramping formula:
//     CFL = CFL_initial * (Res_start / Res_current)^exponent
//
//   With smoothing to prevent oscillations:
//     CFL_new = (1 - smooth) * CFL_old + smooth * CFL_target
//
//   Parameters:
//   - cflRampStartRes: Residual threshold to begin ramping
//   - cflRampExponent: Controls how aggressively CFL increases
//   - cflRampSmooth: Smoothing factor (0.1 = 90% old, 10% new)
//   - pseudoCFLMax: Upper limit on CFL
//
// PSEUDO-TIME STATISTICS:
//   When logPseudoDtStats is enabled, prints min/max/avg pseudo-Δt
//   for each momentum component. Useful for debugging stability issues
//   and understanding where the CFL limit is active.
//
// ============================================================================
#include "SIMPLE.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <limits>

// ============================================================================
// resetPseudoTimeControllerState: Initialize CFL/controller history at startup
// ============================================================================
void SIMPLE::resetPseudoTimeControllerState() {
    comsolErrorPrev = -1.0f;
    comsolErrorPrevPrev = -1.0f;
    serResidPrev = 0.0f;
    lsCurrentAlpha = 1.0f;
    lsBacktrackCount = 0;

    if (!enablePseudoTimeStepping) {
        pseudoCFLRatio = 0.0f;
        return;
    }

    if (pseudoControllerMode == 1) {
        pseudoCFL = std::max(computeComsolManualCFL(1), comsolPIDCFLMin);
    } else if (pseudoControllerMode == 2) {
        pseudoCFL = std::clamp(comsolPIDInitialCFL, comsolPIDCFLMin, comsolPIDCFLMax);
    } else if (enableCflRamp) {
        pseudoCFL = pseudoCFLInitial;
    }

    pseudoCFLRatio = computePseudoCFLRatio(pseudoCFL);
}

// ============================================================================
// computeComsolManualCFL: Piecewise manual schedule from COMSOL Eq. (3-66)
// ============================================================================
float SIMPLE::computeComsolManualCFL(int nonlinearIteration) const {
    const int n = std::max(1, nonlinearIteration);
    const float a = std::max(comsolManualBase, 1.0f);

    const int s1 = std::max(0, comsolManualStage1Span);
    const int s2 = std::max(0, comsolManualStage2Span);
    const int s3 = std::max(0, comsolManualStage3Span);

    float cfl = std::pow(a, static_cast<float>(std::min(n, s1)));

    if (n > comsolManualStage2Offset) {
        const int exp2 = std::min(n - comsolManualStage2Offset, s2);
        cfl += comsolManualStage2Multiplier * std::pow(a, static_cast<float>(exp2));
    }

    if (n > comsolManualStage3Offset) {
        const int exp3 = std::min(n - comsolManualStage3Offset, s3);
        cfl += comsolManualStage3Multiplier * std::pow(a, static_cast<float>(exp3));
    }

    // Keep CFL finite and physically meaningful.
    if (!std::isfinite(cfl)) {
        cfl = comsolPIDCFLMax;
    }
    return std::clamp(cfl, comsolPIDCFLMin, comsolPIDCFLMax);
}

// ============================================================================
// computePseudoCFLRatio: COMSOL-style progress ratio
// ============================================================================
float SIMPLE::computePseudoCFLRatio(float cfl) const {
    const float cflInf = std::max(pseudoCFLInfinity, 1.0001f);
    const float cflSafe = std::max(cfl, 1.0f);
    const float denom = std::log(cflInf);

    if (!std::isfinite(denom) || denom <= 0.0f) {
        return 1.0f;
    }

    float ratio = std::log(cflSafe) / denom;
    if (!std::isfinite(ratio)) {
        ratio = 0.0f;
    }
    return std::clamp(ratio, 0.0f, 1.0f);
}

// ============================================================================
// pseudoSpeedAtU/V: Local speed magnitude at staggered velocity faces
// ============================================================================
float SIMPLE::pseudoSpeedAtU(int i, int j) const {
    const int i0 = std::clamp(i, 0, M - 1);
    const int im1 = std::clamp(i - 1, 0, M - 1);
    const int j0 = std::clamp(j, 0, N);
    const int jp1 = std::clamp(j + 1, 0, N);

    const float uFace = u(std::clamp(i, 0, M), std::clamp(j, 0, N - 1));
    const float vInterp = 0.25f * (
        v(im1, j0) + v(im1, jp1) +
        v(i0,  j0) + v(i0,  jp1)
    );
    return std::sqrt(uFace * uFace + vInterp * vInterp);
}

float SIMPLE::pseudoSpeedAtV(int i, int j) const {
    const int i0 = std::clamp(i, 0, M);
    const int ip1 = std::clamp(i + 1, 0, M);
    const int j0 = std::clamp(j, 0, N - 1);
    const int jm1 = std::clamp(j - 1, 0, N - 1);

    const float vFace = v(std::clamp(i, 0, M - 1), std::clamp(j, 0, N));
    const float uInterp = 0.25f * (
        u(i0, jm1) + u(ip1, jm1) +
        u(i0, j0)  + u(ip1, j0)
    );
    return std::sqrt(uInterp * uInterp + vFace * vFace);
}

// ============================================================================
// computePseudoDtFromSpeed: Unified local/global pseudo time step
// ============================================================================
float SIMPLE::computePseudoDtFromSpeed(float speed) const {
    if (!enablePseudoTimeStepping) {
        return std::numeric_limits<float>::infinity();
    }

    const float dtGlobal = std::max(timeStep, 1e-20f);
    if (!useLocalPseudoTime) {
        return dtGlobal;
    }

    const float hChar = std::min(hx, hy);
    const float speedSafe = std::max(std::abs(speed), minPseudoSpeed);
    const float dtLocal = std::max(pseudoCFL * hChar / speedSafe, 1e-20f);

    // Legacy mode keeps a hard global cap. COMSOL-style modes use pure local dt.
    if (pseudoControllerMode == 0) {
        return std::min(dtLocal, dtGlobal);
    }
    return dtLocal;
}

// ============================================================================
// updatePseudoTimeController: Unified pseudo-time CFL update
// ============================================================================
void SIMPLE::updatePseudoTimeController(float currentError, int completedIteration) {
    if (!enablePseudoTimeStepping) {
        pseudoCFLRatio = 0.0f;
        return;
    }

    const float err = std::max(currentError, 1e-30f);

    // COMSOL manual schedule (Eq. 3-66)
    if (pseudoControllerMode == 1) {
        // completedIteration = n, we prepare CFL for iteration n+1
        pseudoCFL = computeComsolManualCFL(completedIteration + 1);
        pseudoCFLRatio = computePseudoCFLRatio(pseudoCFL);
        return;
    }

    // COMSOL multiplicative PID schedule (Eq. 20-7)
    if (pseudoControllerMode == 2) {
        if (pseudoCFL <= 0.0f || !std::isfinite(pseudoCFL)) {
            pseudoCFL = std::clamp(comsolPIDInitialCFL, comsolPIDCFLMin, comsolPIDCFLMax);
        }

        float gain = 1.0f;
        if (comsolErrorPrev > 0.0f) {
            const float ratioP = std::max(comsolErrorPrev / err, 1e-30f);
            const float ratioI = std::max(comsolPIDTol / err, 1e-30f);
            gain *= std::pow(ratioP, comsolPIDkP);
            gain *= std::pow(ratioI, comsolPIDkI);

            if (comsolErrorPrevPrev > 0.0f) {
                const float ratioD = std::max(
                    (comsolErrorPrev * comsolErrorPrev) / (err * comsolErrorPrevPrev),
                    1e-30f
                );
                gain *= std::pow(ratioD, comsolPIDkD);
            }
        }

        if (!std::isfinite(gain)) {
            gain = 1.0f;
        }
        gain = std::clamp(gain, comsolPIDGainMin, comsolPIDGainMax);

        pseudoCFL *= gain;
        pseudoCFL = std::clamp(pseudoCFL, comsolPIDCFLMin, comsolPIDCFLMax);

        comsolErrorPrevPrev = comsolErrorPrev;
        comsolErrorPrev = err;
        pseudoCFLRatio = computePseudoCFLRatio(pseudoCFL);
        return;
    }

    // Legacy controller stack
    updateCflSER(err, completedIteration - 1);
    applyLineSearch(err);
    updateCflRamp(err);
    pseudoCFLRatio = computePseudoCFLRatio(pseudoCFL);
}

// ============================================================================
// updateCflRamp: Adjust pseudo-CFL based on current residual level
// ============================================================================
void SIMPLE::updateCflRamp(float currRes) {
    if (!enablePseudoTimeStepping || !enableCflRamp) return;
    if (enableSER) return;  // SER takes precedence

    currRes = std::max(currRes, 1e-12f);

    if (currRes > cflRampStartRes) {
        pseudoCFL = pseudoCFLInitial;
    } else {
        float ratio     = cflRampStartRes / currRes;
        float rawFactor = std::pow(ratio, cflRampExponent);
        float targetCFL = std::min(pseudoCFLInitial * rawFactor, pseudoCFLMax);

        pseudoCFL = (1.0f - cflRampSmooth) * pseudoCFL
                  +  cflRampSmooth         * targetCFL;

        if (!std::isfinite(pseudoCFL) || pseudoCFL <= 0.0f) {
            pseudoCFL = pseudoCFLInitial;
        }
    }
}

void logPseudoStats(const SIMPLE& solver, const char* label, const SIMPLE::PseudoDtStats& stats) {
    if (!solver.enablePseudoTimeStepping || !solver.useLocalPseudoTime || !stats.valid) return;
    std::cout << "        Pseudo-dt " << label << " [min/avg/max] = "
              << std::scientific << std::setprecision(3)
              << stats.min << " / " << stats.avg << " / " << stats.max
              << "  (samples=" << stats.samples << ")" << std::defaultfloat << std::endl;
}

// ============================================================================
// updateCflSER: Switched Evolution Relaxation (Mulder & van Leer)
// ============================================================================
// SER adjusts CFL based on residual reduction ratio:
//   CFL^k = min(CFL^(k-1) * |R^(k-1)|_L2 / |R^k|_L2, CFL_max)
// If residual decreases, CFL increases proportionally.
// If residual increases, CFL decreases proportionally.
// Reference: Mulder & van Leer, "Experiments with implicit upwind methods"
// ============================================================================
void SIMPLE::updateCflSER(float currentResidL2, int iteration) {
    if (!enablePseudoTimeStepping || !enableSER) return;
    if (iteration < serMinIter) return;  // Wait for initial iterations
    
    // Ensure valid residual
    currentResidL2 = std::max(currentResidL2, 1e-15f);
    
    // Need valid previous residual to compute ratio
    if (serResidPrev <= 0.0f) {
        serResidPrev = currentResidL2;
        return;
    }
    
    // SER formula: CFL^k = CFL^(k-1) * |R^(k-1)| / |R^k|
    float ratio = serResidPrev / currentResidL2;
    
    // Limit ratio to prevent extreme jumps (more conservative now)
    ratio = std::max(ratio, serMaxDecrease);  // Don't decrease too aggressively
    ratio = std::min(ratio, serMaxIncrease);  // Don't increase too aggressively
    
    float targetCFL = pseudoCFL * ratio;
    
    // Clamp to valid range
    targetCFL = std::max(targetCFL, serCFLMin);
    targetCFL = std::min(targetCFL, serCFLMax);
    
    // Apply with smoothing to prevent oscillations
    pseudoCFL = (1.0f - serSmooth) * pseudoCFL + serSmooth * targetCFL;
    
    // Safety check
    if (!std::isfinite(pseudoCFL) || pseudoCFL <= 0.0f) {
        pseudoCFL = serCFLMin;
    }
    
    // Update previous residual for next iteration
    serResidPrev = currentResidL2;
}

// ============================================================================
// applyLineSearch: Backtracking line search for robustness
// ============================================================================
// Monitors if residual increases unexpectedly. If so, temporarily reduces CFL
// to stabilize the solution. This prevents SER from becoming too aggressive.
// Works in conjunction with SER to provide adaptive robustness.
// ============================================================================
void SIMPLE::applyLineSearch(float currentResidL2) {
    if (!enablePseudoTimeStepping || !enableLineSearch) return;
    if (!enableSER) return;  // Line search only works with SER
    
    // Need valid previous residual to check for increase
    if (serResidPrev <= 0.0f) {
        lsCurrentAlpha = 1.0f;
        lsBacktrackCount = 0;
        return;
    }
    
    // Check if residual increased beyond tolerance
    float residRatio = currentResidL2 / serResidPrev;
    
    if (residRatio > lsResidIncreaseTol) {
        // Residual increased too much - backtrack
        lsBacktrackCount++;
        
        if (lsBacktrackCount <= lsMaxTries) {
            // Reduce step size (which effectively reduces CFL impact)
            lsCurrentAlpha *= lsAlphaReduce;
            lsCurrentAlpha = std::max(lsCurrentAlpha, lsAlphaMin);
            
            // Temporarily reduce CFL to stabilize
            pseudoCFL *= lsAlphaReduce;
            pseudoCFL = std::max(pseudoCFL, serCFLMin);
            
            // Optionally print warning (can be commented out for production)
            // std::cout << "  [Line Search] Residual increased " << residRatio 
            //           << "x - reducing CFL to " << pseudoCFL << std::endl;
        }
    } else {
        // Residual decreased or stayed within tolerance - gradually restore step size
        if (lsCurrentAlpha < 1.0f) {
            lsCurrentAlpha = std::min(lsCurrentAlpha * 1.1f, 1.0f);
        }
        lsBacktrackCount = 0;
    }
}

