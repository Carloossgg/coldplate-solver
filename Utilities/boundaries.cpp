// ============================================================================
// File: Utilities/boundaries.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Implements boundary conditions for the staggered velocity and pressure grids.
//   Called after each momentum/pressure solve to enforce physical constraints.
//
// BOUNDARY CONDITION TYPES:
//
//   1. INLET (left boundary, x = 0):
//      - U: Fixed velocity (Dirichlet) for fluid cells, zero for solid cells
//           u(i, 0) = U_inlet if fluid, 0 if solid
//      - V: Zero (no tangential velocity at inlet, helps stability)
//           v(i, 0) = 0
//      - P: Zero normal gradient (Neumann), dp/dx = 0
//           p(i, 0) = p(i, 1)
//
//   2. OUTLET (right boundary, x = L):
//      - U: Zero gradient (Neumann), du/dx = 0
//           u(i, N-1) = u(i, N-2)
//      - V: Zero gradient (Neumann), dv/dx = 0
//           v(i, N) = v(i, N-1)
//      - P: Fixed reference (Dirichlet), p = 0
//           p(i, N) = 0  (this sets the pressure datum)
//
//   3. WALLS (top/bottom boundaries, y = 0 and y = H):
//      - U: No-slip (Dirichlet), u = 0
//      - V: No-slip (Dirichlet), v = 0
//      - P: Zero normal gradient (Neumann), dp/dy = 0
//           p(0, j) = p(1, j), p(M, j) = p(M-1, j)
//
// INTERNAL OBSTACLES:
//   Internal solid regions are NOT treated as geometric boundaries.
//   Instead, they are handled through Brinkman penalization (alpha field)
//   in the momentum equations. This simplifies the boundary logic and
//   allows for continuous (density-based) topology optimization.
//
// ============================================================================
#include "SIMPLE.h"
#include <algorithm>
#include <cmath>

// ============================================================================
// setVelocityBoundaryConditions: Apply velocity BCs to u and v fields
// ============================================================================
// Called after momentum solve (on uStar, vStar) and after velocity correction
// (on u, v) to ensure boundary values are enforced.
// ============================================================================
void SIMPLE::setVelocityBoundaryConditions(Eigen::MatrixXf& uIn,
                                           Eigen::MatrixXf& vIn)
{
    // Use the ramped inlet velocity set in runIterations()
    const float Uin = inletVelocity;

    // ---------------------------
    // Bottom wall (y = 0): no-slip
    // ---------------------------
    for (int j = 0; j < N; ++j) {
        uIn(0, j) = 0.0f;        // u on bottom faces
    }
    for (int j = 0; j < N + 1; ++j) {
        vIn(0, j) = 0.0f;        // v on bottom faces
    }

    // ---------------------------
    // Top wall (y = H): no-slip
    // ---------------------------
    for (int j = 0; j < N; ++j) {
        uIn(M, j) = 0.0f;        // u on top faces
    }
    for (int j = 0; j < N + 1; ++j) {
        vIn(M - 1, j) = 0.0f;    // v on top faces
    }

    // Left boundary (x = 0): impose inlet velocity on the full fluid part of
    // the inlet face, including inlet-wall corner faces when adjacent inlet
    // cells are fluid. This matches a "uniform velocity inlet patch" treatment.
    //
    // Use gamma (design field) rather than cellType so Brinkman wall forcing
    // and TO densities are handled consistently with momentum equations.
    auto inletFluidFromURow = [&](int uRow) -> bool {
        if (gamma.rows() == 0 || gamma.cols() == 0) return false;
        // Map u-row to nearest physical cell row at inlet.
        int cellRow;
        if (uRow <= 0) {
            cellRow = 0;  // Bottom inlet corner -> first physical row
        } else if (uRow >= M) {
            // Top inlet corner -> last physical row (M-2 with padded indexing)
            cellRow = std::max(0, M - 2);
        } else {
            cellRow = uRow - 1;
        }
        cellRow = std::max(0, std::min(cellRow, static_cast<int>(gamma.rows()) - 1));
        return gamma(cellRow, 0) > 0.5f;
    };

    for (int i = 0; i <= M; ++i) {
        uIn(i, 0) = inletFluidFromURow(i) ? Uin : 0.0f;
    }

    // v on left boundary: no normal flow (x-normal boundary, so v is tangential).
    // We simply set v = 0 everywhere on the inlet column for stability.
    for (int i = 0; i < M; ++i) {
        vIn(i, 0) = 0.0f;
    }

    // ---------------------------
    // Right boundary (x = L): zero-gradient outlet for velocity
    // ---------------------------
    for (int i = 0; i < M + 1; ++i) {
        uIn(i, N - 1) = uIn(i, N - 2);   // du/dx ≈ 0
    }
    for (int i = 0; i < M; ++i) {
        vIn(i, N) = vIn(i, N - 1);       // dv/dx ≈ 0
    }

}

// ----------------------------------------------------------
// Pressure boundary conditions
// ----------------------------------------------------------
void SIMPLE::setPressureBoundaryConditions(Eigen::MatrixXf& pIn)
{
    // p is (M+1 x N+1). Indices: i=0..M, j=0..N.
    //
    // Interior cell-center pressures live at i = 1..M-1, j = 1..N-1
    // (MAC-style staggered layout). The rows/cols at i=0, i=M, j=0, j=N
    // act as ghost layers updated here.

    // Bottom & top: zero normal gradient (dp/dy = 0)
    for (int j = 1; j < N; ++j) {
        pIn(0,     j) = pIn(1,     j);   // bottom (ghost) from first interior row
        pIn(M,     j) = pIn(M - 1, j);   // top    (ghost) from last interior row
    }

    // Left: zero normal gradient (inlet)
    for (int i = 1; i < M; ++i) {
        pIn(i, 0) = pIn(i, 1);
    }

    // Right: pressure outlet, set reference p = 0
    for (int i = 1; i < M; ++i) {
        pIn(i, N) = 0.0;
    }
}

ExternalWallFlags detectExternalNoSlipWalls(const SIMPLE& s)
{
    ExternalWallFlags flags;
    if (s.M < 2 || s.N < 2) return flags;

    const float velTol = 1e-8f;
    const float pGradTol = 1e-6f;
    const float pPinTol = 1e-8f;

    // Helper: iterate inclusive integer range [a, b].
    auto allAbsBelowRange = [&](auto&& accessor, int a, int b, float tol) -> bool {
        if (a > b) return true;
        for (int k = a; k <= b; ++k) {
            if (std::abs(accessor(k)) > tol) return false;
        }
        return true;
    };

    // Corner-aware checks:
    // In this solver, inlet can overwrite u at (bottom-left, top-left) corners.
    // For wall detection we ignore corner points and check the side interior.
    const int uSideStart = 1;
    const int uSideEnd = s.N - 2;
    const bool uBottomZero = allAbsBelowRange([&](int j) { return s.u(0, j); }, uSideStart, uSideEnd, velTol);
    const bool uTopZero = allAbsBelowRange([&](int j) { return s.u(s.M, j); }, uSideStart, uSideEnd, velTol);

    const bool vBottomZero = allAbsBelowRange([&](int j) { return s.v(0, j); }, 0, s.N, velTol);
    const bool vTopZero = allAbsBelowRange([&](int j) { return s.v(s.M - 1, j); }, 0, s.N, velTol);

    bool pBottomNeumann = true;
    bool pTopNeumann = true;
    for (int j = 1; j < s.N; ++j) {
        if (std::abs(s.p(0, j) - s.p(1, j)) > pGradTol) pBottomNeumann = false;
        if (std::abs(s.p(s.M, j) - s.p(s.M - 1, j)) > pGradTol) pTopNeumann = false;
    }
    flags.bottom = uBottomZero && vBottomZero && pBottomNeumann;
    flags.top = uTopZero && vTopZero && pTopNeumann;

    // Left side (x=0): inlet usually makes this non-wall by nonzero u.
    const bool uLeftZero = allAbsBelowRange([&](int i) { return s.u(i, 0); }, 1, s.M - 1, velTol);
    const bool vLeftZero = allAbsBelowRange([&](int i) { return s.v(i, 0); }, 0, s.M - 1, velTol);
    bool pLeftNeumann = true;
    for (int i = 1; i < s.M; ++i) {
        if (std::abs(s.p(i, 0) - s.p(i, 1)) > pGradTol) pLeftNeumann = false;
    }
    flags.left = uLeftZero && vLeftZero && pLeftNeumann;

    // Right side (x=L): if pressure is pinned to zero, treat as outlet (not wall).
    const bool uRightZero = allAbsBelowRange([&](int i) { return s.u(i, s.N - 1); }, 1, s.M - 1, velTol);
    const bool vRightZero = allAbsBelowRange([&](int i) { return s.v(i, s.N); }, 0, s.M - 1, velTol);
    bool pRightNeumann = true;
    bool pRightPinnedZero = true;
    for (int i = 1; i < s.M; ++i) {
        if (std::abs(s.p(i, s.N) - s.p(i, s.N - 1)) > pGradTol) pRightNeumann = false;
        if (std::abs(s.p(i, s.N)) > pPinTol) pRightPinnedZero = false;
    }
    flags.right = uRightZero && vRightZero && pRightNeumann && !pRightPinnedZero;

    return flags;
}

// checkBoundaries is now inline in SIMPLE.h for performance
// (called ~4M times per iteration)
