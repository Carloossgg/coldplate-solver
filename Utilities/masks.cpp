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

    auto isSolid = [&](int i, int j) -> bool {
        int ci = i - 1;
        int cj = j - 1;
        if (ci < 0 || ci >= M || cj < 0 || cj >= N) return false;
        // cellType: 0=fluid, 1=solid, intermediate=buffer
        // For masks: treat cells with cellType > 0.5 as solid
        return cellType(ci, cj) > 0.5;
    };

    for (int i = 0; i < uRows; ++i) {
        for (int j = 0; j < uCols; ++j) {
            bool solid = isSolid(i, j) || isSolid(i, j + 1);
            isFluidU[i * uCols + j] = !solid;
        }
    }

    for (int i = 0; i < vRows; ++i) {
        for (int j = 0; j < vCols; ++j) {
            bool solid = isSolid(i, j) || isSolid(i + 1, j);
            isFluidV[i * vCols + j] = !solid;
        }
    }

    for (int i = 0; i < pRows; ++i) {
        for (int j = 0; j < pCols; ++j) {
            isFluidP[i * pCols + j] = !isSolid(i, j);
        }
    }

    int fluidUCount = 0, fluidVCount = 0, fluidPCount = 0;
    for (bool b : isFluidU) if (b) fluidUCount++;
    for (bool b : isFluidV) if (b) fluidVCount++;
    for (bool b : isFluidP) if (b) fluidPCount++;
    std::cout << "Fluid masks: U=" << fluidUCount << "/" << isFluidU.size()
              << ", V=" << fluidVCount << "/" << isFluidV.size()
              << ", P=" << fluidPCount << "/" << isFluidP.size() << std::endl;
}

