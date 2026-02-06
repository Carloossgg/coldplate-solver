// ============================================================================
// File: Utilities/stabilization.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Residual-based artificial viscosity for stabilization. This computes
//   per-face momentum residuals and converts them to a lagged, cell-centered
//   artificial viscosity field μ_art to damp oscillations locally.
// ============================================================================

#include "SIMPLE.h"
#include <algorithm>
#include <cmath>

void SIMPLE::updateMomentumResiduals()
{
    // Clear fields
    uResidualFace.setZero();
    vResidualFace.setZero();

    // U residuals on u-faces (interior only)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            uResidualFace(i, j) = std::abs(uStar(i, j) - uOld(i, j));
        }
    }

    // V residuals on v-faces (interior only)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            vResidualFace(i, j) = std::abs(vStar(i, j) - vOld(i, j));
        }
    }
}

void SIMPLE::updateResidualViscosity()
{
    if (!enableResidualViscosity) {
        muArt.setZero();
        return;
    }

    const int physRows = M - 1;
    const int physCols = N - 1;
    if (physRows <= 0 || physCols <= 0) {
        muArt.setZero();
        return;
    }

    Eigen::MatrixXf residCenter = Eigen::MatrixXf::Zero(physRows, physCols);

    // Build cell-centered residual magnitude from face residuals
    for (int i = 0; i < physRows; ++i) {
        for (int j = 0; j < physCols; ++j) {
            const int uRow = i + 1;
            const int vCol = j + 1;

            const float uRes = 0.5f * (uResidualFace(uRow, j) + uResidualFace(uRow, j + 1));
            const float vRes = 0.5f * (vResidualFace(i, vCol) + vResidualFace(i + 1, vCol));
            residCenter(i, j) = std::sqrt(uRes * uRes + vRes * vRes);
        }
    }

    // Optional smoothing to avoid noise-driven viscosity spikes
    if (residualViscSmoothIters > 0 && residualViscSmoothWeight > 0.0f) {
        Eigen::MatrixXf temp = residCenter;
        const float w = std::max(0.0f, std::min(residualViscSmoothWeight, 1.0f));
        for (int iter = 0; iter < residualViscSmoothIters; ++iter) {
            for (int i = 0; i < physRows; ++i) {
                for (int j = 0; j < physCols; ++j) {
                    float sum = residCenter(i, j);
                    int count = 1;
                    if (i > 0) { sum += residCenter(i - 1, j); count++; }
                    if (i + 1 < physRows) { sum += residCenter(i + 1, j); count++; }
                    if (j > 0) { sum += residCenter(i, j - 1); count++; }
                    if (j + 1 < physCols) { sum += residCenter(i, j + 1); count++; }
                    const float avg = sum / static_cast<float>(count);
                    temp(i, j) = (1.0f - w) * residCenter(i, j) + w * avg;
                }
            }
            residCenter.swap(temp);
        }
    }

    // Convert residual magnitude to artificial viscosity
    const float hChar = std::min(hx, hy);
    const float muMax = std::max(0.0f, residualViscMaxFactor * eta);
    for (int i = 0; i < physRows; ++i) {
        for (int j = 0; j < physCols; ++j) {
            float r = std::max(0.0f, residCenter(i, j) - residualViscMinVel);
            float muLocal = residualViscCoeff * rho * hChar * r;
            if (muLocal > muMax) muLocal = muMax;
            muArt(i, j) = muLocal;
        }
    }

    // Pad last internal row/column to keep interpolation consistent
    for (int i = 0; i < physRows; ++i) {
        muArt(i, physCols) = muArt(i, physCols - 1);
    }
    for (int j = 0; j < N; ++j) {
        const int srcCol = std::min(j, physCols - 1);
        muArt(physRows, j) = muArt(physRows - 1, srcCol);
    }
}
