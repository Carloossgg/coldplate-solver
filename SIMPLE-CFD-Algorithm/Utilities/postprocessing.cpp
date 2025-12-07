// File: Utilities/postprocessing.cpp
// Author: Peter Tcherkezian
// Description: Physical-location sampling of pressure/velocity:
//   interpolates staggered-grid fields onto user-specified x-planes to report core/full pressure drops, areas,
//   and mass-flow-weighted pressures for summaries.

#include "../SIMPLE.h"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>

// PlaneMetrics struct is declared in SIMPLE.h

// ============================================================================
// Helper: Convert cell column index to physical x-coordinate (cell center)
// ============================================================================
static double cellCenterX(int col, double hx) {
    return (col + 0.5) * hx;
}

// ============================================================================
// Helper: Find the two cell columns that bracket a given physical x-coordinate
// Returns (colLeft, colRight, t) where t is the interpolation factor [0,1]
// x = (1-t)*x_left + t*x_right
// ============================================================================
static void findBracketingColumns(double xPhysical, double hx, int N,
                                   int& colLeft, int& colRight, double& t) {
    // Physical x of first cell center is 0.5*hx, last is (N-0.5)*hx
    double xMin = 0.5 * hx;
    double xMax = (N - 0.5) * hx;
    
    // Clamp to domain
    double xClamped = std::max(xMin, std::min(xPhysical, xMax));
    
    // Find which cell center interval we're in
    // Cell j has center at (j + 0.5) * hx
    // So xClamped is between cell centers at j and j+1 when:
    //   (j + 0.5)*hx <= xClamped < (j + 1.5)*hx
    // => j = floor(xClamped/hx - 0.5)
    double jFloat = xClamped / hx - 0.5;
    colLeft = static_cast<int>(std::floor(jFloat));
    colLeft = std::max(0, std::min(colLeft, N - 2));  // Ensure valid range for interpolation
    colRight = colLeft + 1;
    
    // Interpolation factor
    double xLeft = cellCenterX(colLeft, hx);
    double xRight = cellCenterX(colRight, hx);
    if (std::abs(xRight - xLeft) < 1e-12) {
        t = 0.0;
    } else {
        t = (xClamped - xLeft) / (xRight - xLeft);
    }
    t = std::max(0.0, std::min(1.0, t));  // Clamp t to [0,1]
}

// ============================================================================
// samplePlaneAtX: Sample pressure and velocity at a given physical x-coordinate
// Uses linear interpolation between adjacent cell columns
// Skips rows where either adjacent cell is solid
// ============================================================================
PlaneMetrics samplePlaneAtX(const SIMPLE& solver, double xPhysical) {
    PlaneMetrics metrics;
    
    const int M = solver.M;
    const int N = solver.N;
    const double hx = solver.hx;
    const double hy = solver.hy;
    const double rho = solver.rho;
    
    // Validate inputs
    if (M <= 0 || N <= 0 || hx <= 0.0 || hy <= 0.0) {
        return metrics;
    }
    
    // Domain bounds
    double domainXMin = 0.0;
    double domainXMax = N * hx;
    
    // Check if xPhysical is outside domain and warn
    if (xPhysical < domainXMin || xPhysical > domainXMax) {
        std::cerr << "Warning: Sampling plane x=" << xPhysical 
                  << " m is outside domain [0, " << domainXMax << " m]. Clamping." << std::endl;
    }
    
    // Find bracketing columns for interpolation
    int colLeft, colRight;
    double t;
    findBracketingColumns(xPhysical, hx, N, colLeft, colRight, t);
    
    // Accumulators
    double staticAreaSum = 0.0;
    double dynAreaSum = 0.0;
    double totalWeighted = 0.0;
    double area = 0.0;
    double fluxSum = 0.0;
    
    // Iterate over rows (interior pressure rows: 1 to M-1)
    // Pressure grid is (M+1) x (N+1), cellType is M x N
    // Pressure p(i, j) corresponds to cellType(i-1, j-1)
    for (int i = 1; i < M; ++i) {
        const int cellRow = i - 1;  // Corresponding row in cellType
        
        // Bounds check
        if (cellRow < 0 || cellRow >= M) continue;
        if (colLeft < 0 || colLeft >= N || colRight < 0 || colRight >= N) continue;
        
        // Skip if EITHER adjacent cell is solid (conservative approach)
        if (solver.cellType(cellRow, colLeft) != 0 || solver.cellType(cellRow, colRight) != 0) {
            continue;
        }
        
        const double faceArea = hy;
        area += faceArea;
        
        // Interpolate pressure: p is at (i, j+1) for cellType column j
        // So for cellType columns colLeft and colRight, pressure columns are colLeft+1 and colRight+1
        int pColLeft = colLeft + 1;
        int pColRight = colRight + 1;
        
        // Bounds check for pressure grid
        pColLeft = std::max(0, std::min(pColLeft, static_cast<int>(solver.p.cols()) - 1));
        pColRight = std::max(0, std::min(pColRight, static_cast<int>(solver.p.cols()) - 1));
        
        double pLeft = solver.p(i, pColLeft);
        double pRight = solver.p(i, pColRight);
        double staticP = (1.0 - t) * pLeft + t * pRight;
        
        // Interpolate u-velocity
        // u is at cell faces: u(i, j) is the velocity at the left face of cell (i-1, j)
        // For interpolation at physical x, we need to interpolate between u values
        // u grid is (M+1) x N
        int uColLeft = std::max(0, std::min(colLeft, static_cast<int>(solver.u.cols()) - 1));
        int uColRight = std::max(0, std::min(colRight, static_cast<int>(solver.u.cols()) - 1));
        
        double uLeft = solver.u(i, uColLeft);
        double uRight = solver.u(i, uColRight);
        double uNormal = (1.0 - t) * uLeft + t * uRight;
        
        // Interpolate v-velocity (tangential)
        // v is at horizontal faces: v(i, j) is the velocity at the bottom face of cell (i, j-1)
        // v grid is M x (N+1)
        // For each row i in pressure grid, we need v at rows i-1 and i
        int vColLeft = std::max(0, std::min(colLeft, static_cast<int>(solver.v.cols()) - 1));
        int vColRight = std::max(0, std::min(colRight, static_cast<int>(solver.v.cols()) - 1));
        
        int vRowTop = std::max(0, std::min(i - 1, static_cast<int>(solver.v.rows()) - 1));
        int vRowBottom = std::max(0, std::min(i, static_cast<int>(solver.v.rows()) - 1));
        
        // Average v at top and bottom of the cell, then interpolate in x
        double vTopLeft = solver.v(vRowTop, vColLeft);
        double vTopRight = solver.v(vRowTop, vColRight);
        double vBottomLeft = solver.v(vRowBottom, vColLeft);
        double vBottomRight = solver.v(vRowBottom, vColRight);
        
        double vTop = (1.0 - t) * vTopLeft + t * vTopRight;
        double vBottom = (1.0 - t) * vBottomLeft + t * vBottomRight;
        double vTangential = 0.5 * (vTop + vBottom);
        
        // Compute dynamic pressure
        double velMag = std::sqrt(uNormal * uNormal + vTangential * vTangential);
        double dynP = 0.5 * rho * velMag * velMag;
        
        staticAreaSum += staticP * faceArea;
        dynAreaSum += dynP * faceArea;
        
        // Mass-weighted total pressure (only for positive flux = flow in expected direction)
        double flux = rho * uNormal * faceArea;
        if (flux > 0.0) {
            fluxSum += flux;
            totalWeighted += (staticP + dynP) * flux;
        }
    }
    
    // Compute averages
    metrics.flowArea = area;
    metrics.massFlux = fluxSum;
    if (area > 0.0) {
        metrics.avgStatic = staticAreaSum / area;
        metrics.avgDynamic = dynAreaSum / area;
        metrics.valid = true;
    }
    if (fluxSum > 0.0) {
        metrics.avgTotal = totalWeighted / fluxSum;
    } else {
        // Fallback if no positive flux
        metrics.avgTotal = metrics.avgStatic + metrics.avgDynamic;
    }
    
    return metrics;
}

// ============================================================================
// printPlaneInfo: Debug helper to print plane sampling information
// ============================================================================
void printPlaneInfo(const char* name, double xPhysical, const PlaneMetrics& m) {
    std::cout << "  " << name << " (x=" << std::fixed << std::setprecision(4) 
              << xPhysical * 1000.0 << " mm):" << std::endl;
    std::cout << "    Flow area:     " << m.flowArea << " m^2/depth" << std::endl;
    std::cout << "    Static P:      " << std::setprecision(1) << m.avgStatic << " Pa" << std::endl;
    std::cout << "    Dynamic P:     " << m.avgDynamic << " Pa" << std::endl;
    std::cout << "    Total P:       " << m.avgTotal << " Pa" << std::endl;
}

