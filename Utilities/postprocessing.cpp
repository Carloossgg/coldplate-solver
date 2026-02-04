// ============================================================================
// File: Utilities/postprocessing.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Implements physical location-based pressure and velocity sampling for
//   accurate pressure-drop measurements independent of mesh cell boundaries.
//
// PHYSICAL LOCATION SAMPLING:
//   Rather than sampling at fixed cell indices (which change with refinement),
//   we sample at fixed physical x-coordinates (in meters). This enables:
//   - Consistent pressure-drop comparison across different mesh resolutions
//   - User-configurable sampling planes (xPlaneCoreInlet, xPlaneCoreOutlet)
//   - Accurate interpolation between cell-center values
//
// PRESSURE DROP TYPES:
//   1. STATIC pressure drop: ΔP_static = P_inlet - P_outlet
//      Measures the work done against friction/obstacles
//
//   2. DYNAMIC pressure: P_dynamic = ½ρV²
//      Kinetic energy per unit volume
//
//   3. TOTAL pressure drop: ΔP_total = (P_static + P_dynamic)_in - (...)_out
//      Accounts for both static pressure loss and velocity changes
//      Uses MASS-FLOW-WEIGHTED total pressure (STAR pressure-drop style)
//
// INTERPOLATION:
//   For a given physical x-coordinate, we:
//   1. Find the two cell columns that bracket this x (colLeft, colRight)
//   2. Compute interpolation factor t = (x - x_left) / (x_right - x_left)
//   3. Interpolate: value = (1-t)*value_left + t*value_right
//
// FLUID-ONLY HANDLING:
//   A row is sampled only if both adjacent cells are pure fluid (gamma ~= 1).
//   Any solid/intermediate-density cell is excluded from plane averaging.
//
// ============================================================================

#include "../SIMPLE.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// PlaneMetrics struct is declared in SIMPLE.h

// ============================================================================
// Helper: Convert cell column index to physical x-coordinate (cell center)
// ============================================================================
// ============================================================================
// Helper: Convert cell column index to physical x-coordinate (cell center)
// ============================================================================
static float cellCenterX(int col, float hx) {
    return (col + 0.5f) * hx;
}

// ============================================================================
// Helper: Find the two cell columns that bracket a given physical x-coordinate
// Returns (colLeft, colRight, t) where t is the interpolation factor [0,1]
// x = (1-t)*x_left + t*x_right
// ============================================================================
static void findBracketingColumns(float xPhysical, float hx, int N,
                                   int& colLeft, int& colRight, float& t) {
    // Physical x of first cell center is 0.5*hx, last is (N-0.5)*hx
    float xMin = 0.5f * hx;
    float xMax = (N - 0.5f) * hx;
    
    // Clamp to domain
    float xClamped = std::max(xMin, std::min(xPhysical, xMax));
    
    // Find which cell center interval we're in
    // Cell j has center at (j + 0.5) * hx
    // So xClamped is between cell centers at j and j+1 when:
    //   (j + 0.5)*hx <= xClamped < (j + 1.5)*hx
    // => j = floor(xClamped/hx - 0.5)
    float jFloat = xClamped / hx - 0.5f;
    colLeft = static_cast<int>(std::floor(jFloat));
    colLeft = std::max(0, std::min(colLeft, N - 2));  // Ensure valid range for interpolation
    colRight = colLeft + 1;
    
    // Interpolation factor
    float xLeft = cellCenterX(colLeft, hx);
    float xRight = cellCenterX(colRight, hx);
    if (std::abs(xRight - xLeft) < 1e-12f) {
        t = 0.0f;
    } else {
        t = (xClamped - xLeft) / (xRight - xLeft);
    }
    t = std::max(0.0f, std::min(1.0f, t));  // Clamp t to [0,1]
}

// ============================================================================
// samplePlaneAtX: Sample pressure and velocity at a given physical x-coordinate
// Uses linear interpolation between adjacent cell columns
// Skips rows unless both adjacent cells are pure fluid (gamma ~= 1)
// ============================================================================
PlaneMetrics samplePlaneAtX(const SIMPLE& solver, float xPhysical) {
    PlaneMetrics metrics;
    
    const int M = solver.M;
    const int N = solver.N;
    const float hx = solver.hx;
    const float hy = solver.hy;
    const float rho = solver.rho;
    
    // Validate inputs
    if (M <= 0 || N <= 0 || hx <= 0.0f || hy <= 0.0f) {
        return metrics;
    }
    
    // Domain bounds
    float domainXMin = 0.0f;
    float domainXMax = N * hx;
    
    // Check if xPhysical is outside domain and warn
    if (xPhysical < domainXMin || xPhysical > domainXMax) {
        std::cerr << "Warning: Sampling plane x=" << xPhysical 
                  << " m is outside domain [0, " << domainXMax << " m]. Clamping." << std::endl;
    }
    
    // Find bracketing columns for interpolation
    int colLeft, colRight;
    float t;
    findBracketingColumns(xPhysical, hx, N, colLeft, colRight, t);
    
    // Accumulators
    float staticAreaSum = 0.0f;
    float dynAreaSum = 0.0f;
    float totalWeighted = 0.0f;
    float area = 0.0f;
    float fluxSum = 0.0f;
    
    // Iterate over rows (interior pressure rows: 1 to M-1)
    // Pressure grid is (M+1) x (N+1), cellType is M x N
    // Pressure p(i, j) corresponds to cellType(i-1, j-1)
    for (int i = 1; i < M; ++i) {
        const int cellRow = i - 1;  // Corresponding row in cellType
        
        // Bounds check
        if (cellRow < 0 || cellRow >= M) continue;
        if (colLeft < 0 || colLeft >= N || colRight < 0 || colRight >= N) continue;
        
        // Include only pure-fluid rows for reporting:
        // both adjacent cells must have gamma ~= 1 (not solid/intermediate).
        constexpr float pureFluidTol = 1e-6f;
        if (solver.gamma(cellRow, colLeft) < 1.0f - pureFluidTol ||
            solver.gamma(cellRow, colRight) < 1.0f - pureFluidTol) {
            continue;
        }
        
        const float faceArea = hy;
        area += faceArea;
        
        // Interpolate pressure: p is at (i, j+1) for cellType column j
        // So for cellType columns colLeft and colRight, pressure columns are colLeft+1 and colRight+1
        int pColLeft = colLeft + 1;
        int pColRight = colRight + 1;
        
        // Bounds check for pressure grid
        pColLeft = std::max(0, std::min(pColLeft, static_cast<int>(solver.p.cols()) - 1));
        pColRight = std::max(0, std::min(pColRight, static_cast<int>(solver.p.cols()) - 1));
        
        float pLeft = solver.p(i, pColLeft);
        float pRight = solver.p(i, pColRight);
        float staticP = (1.0f - t) * pLeft + t * pRight;
        
        // Interpolate u-velocity
        // u is at cell faces: u(i, j) is the velocity at the left face of cell (i-1, j)
        // For interpolation at physical x, we need to interpolate between u values
        // u grid is (M+1) x N
        int uColLeft = std::max(0, std::min(colLeft, static_cast<int>(solver.u.cols()) - 1));
        int uColRight = std::max(0, std::min(colRight, static_cast<int>(solver.u.cols()) - 1));
        
        float uLeft = solver.u(i, uColLeft);
        float uRight = solver.u(i, uColRight);
        float uNormal = (1.0f - t) * uLeft + t * uRight;
        
        // Interpolate v-velocity (tangential)
        // v is at horizontal faces: v(i, j) is the velocity at the bottom face of cell (i, j-1)
        // v grid is M x (N+1)
        // For each row i in pressure grid, we need v at rows i-1 and i
        int vColLeft = std::max(0, std::min(colLeft, static_cast<int>(solver.v.cols()) - 1));
        int vColRight = std::max(0, std::min(colRight, static_cast<int>(solver.v.cols()) - 1));
        
        int vRowTop = std::max(0, std::min(i - 1, static_cast<int>(solver.v.rows()) - 1));
        int vRowBottom = std::max(0, std::min(i, static_cast<int>(solver.v.rows()) - 1));
        
        // Average v at top and bottom of the cell, then interpolate in x
        float vTopLeft = solver.v(vRowTop, vColLeft);
        float vTopRight = solver.v(vRowTop, vColRight);
        float vBottomLeft = solver.v(vRowBottom, vColLeft);
        float vBottomRight = solver.v(vRowBottom, vColRight);
        
        float vTop = (1.0f - t) * vTopLeft + t * vTopRight;
        float vBottom = (1.0f - t) * vBottomLeft + t * vBottomRight;
        float vTangential = 0.5f * (vTop + vBottom);
        
        // Compute dynamic pressure
        float velMag = std::sqrt(uNormal * uNormal + vTangential * vTangential);
        float dynP = 0.5f * rho * velMag * velMag;
        float totalPAbs = staticP + dynP;  // In this solver, p is gauge; absolute offset cancels in dP.
        
        staticAreaSum += staticP * faceArea;
        dynAreaSum += dynP * faceArea;
        
        // STAR-style mass-flow weighting: weight by |m_dot| at each face
        float flux = rho * uNormal * faceArea;
        float fluxAbs = std::abs(flux);
        if (fluxAbs > 0.0f) {
            fluxSum += fluxAbs;
            totalWeighted += totalPAbs * fluxAbs;
        }
    }
    
    // Compute averages
    metrics.flowArea = area;
    metrics.massFlux = fluxSum;
    if (area > 0.0f) {
        metrics.avgStatic = staticAreaSum / area;
        metrics.avgDynamic = dynAreaSum / area;
        metrics.valid = true;
    }
    if (fluxSum > 0.0f) {
        metrics.avgTotal = totalWeighted / fluxSum;
    } else if (area > 0.0f) {
        // No through-flow available: fall back to area-averaged total.
        metrics.avgTotal = metrics.avgStatic + metrics.avgDynamic;
    }
    
    return metrics;
}

// ============================================================================
// printPlaneInfo: Debug helper to print plane sampling information
// ============================================================================
void printPlaneInfo(const char* name, float xPhysical, const PlaneMetrics& m) {
    std::cout << "  " << name << " (x=" << std::fixed << std::setprecision(4) 
              << xPhysical * 1000.0f << " mm):" << std::endl;
    std::cout << "    Flow area:     " << m.flowArea << " m^2/depth" << std::endl;
    std::cout << "    Static P:      " << std::setprecision(1) << m.avgStatic << " Pa" << std::endl;
    std::cout << "    Dynamic P:     " << m.avgDynamic << " Pa" << std::endl;
    std::cout << "    Total P:       " << m.avgTotal << " Pa" << std::endl;
}

