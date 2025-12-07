// File: Utilities/masks.cpp
// Author: Peter Tcherkezian
// Description: Builds fluid/solid masks for staggered grid (U, V, P) and alpha accessors:
//   precomputes boolean masks for fluid cells/faces (Brinkman penalization) and provides interpolated porosity (alpha)
//   at U/V locations for momentum assembly.
#include "SIMPLE.h"
#include <iostream>

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
        return cellType(ci, cj) != 0;
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

