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

