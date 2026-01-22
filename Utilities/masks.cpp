// ============================================================================
// File: Utilities/masks.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Precomputes boolean fluid/solid masks for all staggered grid locations.
//   These masks enable fast lookup during momentum assembly to skip solid cells
//   without expensive geometric checks each iteration.
//
// MASK COMPUTATION:
//   For each velocity/pressure location, we check if adjacent cells are solid:
//   
//   - isFluidU[i,j]: True if u(i,j) is in fluid region
//     A u-face is fluid only if BOTH adjacent cells are fluid:
//     isFluidU = !isSolid(i,j) && !isSolid(i,j+1)
//
//   - isFluidV[i,j]: True if v(i,j) is in fluid region
//     A v-face is fluid only if BOTH adjacent cells are fluid:
//     isFluidV = !isSolid(i,j) && !isSolid(i+1,j)
//
//   - isFluidP[i,j]: True if p(i,j) is in fluid region
//     isFluidP = !isSolid(i,j)
//
// SOLID THRESHOLD:
//   Cells are classified as solid if cellType > 0.5
//   This allows for continuous (density-based) geometry where:
//   - cellType = 0.0 → pure fluid
//   - cellType = 1.0 → pure solid
//   - 0 < cellType < 1 → buffer zone (treated as fluid for mask, but penalized)
//
// STORAGE:
//   Masks are stored as 1D std::vector<bool> for cache efficiency.
//   Linear index = i * cols + j
//
// ============================================================================
#include "SIMPLE.h"
#include <iostream>

// ============================================================================
// buildFluidMasks: Precompute fluid/solid masks for all grid locations
// ============================================================================
// ============================================================================
// buildFluidMasks: Precompute fluid/solid masks for all grid locations
// ============================================================================
void SIMPLE::buildFluidMasks() {
    const int uRows = M + 1;
    const int uCols = N;
    const int vRows = M;
    const int vCols = N + 1;
    const int pRows = M + 1;
    const int pCols = N + 1;

    isFluidU.resize(uRows * uCols, false);
    isFluidV.resize(vRows * vCols, false);
    isFluidP.resize(pRows * pCols, false);

    // Helper: Determine if a cell is "computationally solid"
    // For TopOpt with Brinkman, we only skip if it is nearly 100% solid.
    // Threshold 0.99 allows Porous/Interface cells to be solved.
    auto isSolid = [&](int i, int j) -> bool {
        int ci = i - 1;
        int cj = j - 1;
        if (ci < 0 || ci >= M || cj < 0 || cj >= N) return false;
        
        // cellType: 0=fluid, 1=solid
        // Threshold > 0.99 means only "Pure Solid" is skipped.
        // Gamma 0.1 to 0.99 (Porous) is treated as Fluid (Active).
        return cellType(ci, cj) > 0.99;
    };

    // 1. U-Velocity Mask (Vertical Face)
    for (int i = 0; i < uRows; ++i) {
        for (int j = 0; j < uCols; ++j) {
            bool fluid = false;

            // Boundary Check
            if (i >= 1 && i < M && j >= 1 && j < N - 1) {
                // Interior Face: Open if LEFT or RIGHT is not pure solid?
                // Actually, standard staggered grid says a face is blocked if EITHER side is solid.
                // BUT for Brinkman, we want to solve even if one side is porous.
                // Logic: A face is "solid" (skipped) only if BOTH sides are pure solid?
                // Or if either side is pure solid?
                //
                // Standard No-Slip: Block if EITHER is solid.
                // Brinkman: Block only if BOTH is solid? 
                // Let's stick to consistent logic: If a cell is "Active" (<=0.99), we solve u on its face.
                
                // If Face separates two Pure Solids -> Skip.
                // If Face touches at least one Active Cell -> Solve?
                // 
                // Let's use standard logic with the high threshold:
                // Block if Left is Solid OR Right is Solid.
                // Since "Solid" means >0.99, a porous cell (0.5) is NOT solid.
                // So flow can enter a porous cell.
                bool leftSolid = isSolid(i, j);
                bool rightSolid = isSolid(i, j + 1);
                
                if (!leftSolid && !rightSolid) {
                    fluid = true;
                }
            }
            // Inlet/Outlet Exception
            else if (i >= 1 && i < M) {
                 if (j == 0 && !isSolid(i, 1)) fluid = true;       // Inlet
                 if (j == N - 1 && !isSolid(i, N)) fluid = true;   // Outlet
            }
            isFluidU[i * uCols + j] = fluid;
        }
    }

    // 2. V-Velocity Mask (Horizontal Face)
    for (int i = 0; i < vRows; ++i) {
        for (int j = 0; j < vCols; ++j) {
             bool fluid = false;
             if (i >= 1 && i < M - 1 && j >= 1 && j < N) {
                 bool topSolid = isSolid(i, j);
                 bool botSolid = isSolid(i + 1, j);
                 
                 // Open if neither is pure solid
                 if (!topSolid && !botSolid) {
                     fluid = true;
                 }
             }
             isFluidV[i * vCols + j] = fluid;
        }
    }

    // 3. Pressure Mask (Cell Center)
    for (int i = 0; i < pRows; ++i) {
        for (int j = 0; j < pCols; ++j) {
            bool fluid = false;
            if (i >= 1 && i < M && j >= 1 && j < N) {
                // Active if not pure solid
                if (!isSolid(i, j)) {
                    fluid = true; // Solve pressure in porous/fluid zones
                }
            }
            isFluidP[i * pCols + j] = fluid;
        }
    }

    int fluidUCount = 0, fluidVCount = 0, fluidPCount = 0;
    for (bool b : isFluidU) if (b) fluidUCount++;
    for (bool b : isFluidV) if (b) fluidVCount++;
    for (bool b : isFluidP) if (b) fluidPCount++;
    std::cout << "Fluid masks (Thresh 0.99): U=" << fluidUCount << "/" << isFluidU.size()
              << ", V=" << fluidVCount << "/" << isFluidV.size()
              << ", P=" << fluidPCount << "/" << isFluidP.size() << std::endl;
}

