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
void SIMPLE::updateCflRamp(double currRes) {
    if (!enablePseudoTimeStepping || !enableCflRamp) return;

    currRes = std::max(currRes, 1e-12);

    if (currRes > cflRampStartRes) {
        pseudoCFL = pseudoCFLInitial;
    } else {
        double ratio     = cflRampStartRes / currRes;
        double rawFactor = std::pow(ratio, cflRampExponent);
        double targetCFL = std::min(pseudoCFLInitial * rawFactor, pseudoCFLMax);

        pseudoCFL = (1.0 - cflRampSmooth) * pseudoCFL
                  +  cflRampSmooth         * targetCFL;

        if (!std::isfinite(pseudoCFL) || pseudoCFL <= 0.0) {
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

