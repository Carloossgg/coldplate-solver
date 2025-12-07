// File: Utilities/time_control.cpp
// Author: Peter Tcherkezian
// Description: Pseudo-transient time stepping and CFL ramp control:
//   updates pseudo-time/CFL limits based on residual history and iteration count, and logs pseudo-dt stats for
//   diagnosing convergence speed/stability.
#include "SIMPLE.h"
#include <algorithm>
#include <cmath>
#include <iostream>
#include <iomanip>

// Update pseudo CFL ramp based on current mass residual (same logic as before)
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

