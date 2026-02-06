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

bool hasExternalNoSlipSouthForU(const SIMPLE& s, int i, int j, float fluidTol)
{
    if (i - 1 != 0) return false;               // Must be adjacent to bottom boundary
    if (j < 1 || j > s.N - 2) return false;     // Interior u-face only
    if (gammaAtU(s, i, j) < fluidTol) return false; // Exclude Brinkman/solid regions

    const float velTol = 1e-8f;
    const bool uZero = std::abs(s.u(0, j)) <= velTol;
    const bool vZero = std::abs(s.v(0, j)) <= velTol && std::abs(s.v(0, j + 1)) <= velTol;
    return uZero && vZero;
}

bool hasExternalNoSlipNorthForU(const SIMPLE& s, int i, int j, float fluidTol)
{
    if (i + 1 != s.M) return false;             // Must be adjacent to top boundary
    if (j < 1 || j > s.N - 2) return false;     // Interior u-face only
    if (gammaAtU(s, i, j) < fluidTol) return false; // Exclude Brinkman/solid regions

    const float velTol = 1e-8f;
    const bool uZero = std::abs(s.u(s.M, j)) <= velTol;
    const bool vZero = std::abs(s.v(s.M - 1, j)) <= velTol &&
                       std::abs(s.v(s.M - 1, j + 1)) <= velTol;
    return uZero && vZero;
}

bool hasExternalNoSlipWestForV(const SIMPLE& s, int i, int j, float fluidTol)
{
    if (j - 1 != 0) return false;               // Must be adjacent to left boundary
    if (i < 1 || i > s.M - 2) return false;     // Interior v-face only
    if (gammaAtV(s, i, j) < fluidTol) return false; // Exclude Brinkman/solid regions

    // Left side is inlet by default in this solver when targetVel > 0.
    // Do not reinterpret inlet faces as walls even if the current iterate is near zero.
    const float velTol = 1e-8f;
    if (std::abs(s.targetVel) > velTol) return false;

    const bool vZero = std::abs(s.v(i, 0)) <= velTol;
    const bool uZero = std::abs(s.u(i, 0)) <= velTol && std::abs(s.u(i + 1, 0)) <= velTol;
    return vZero && uZero;
}

bool hasExternalNoSlipEastForV(const SIMPLE& s, int i, int j, float fluidTol)
{
    if (j + 1 != s.N) return false;             // Must be adjacent to right boundary
    if (i < 1 || i > s.M - 2) return false;     // Interior v-face only
    if (gammaAtV(s, i, j) < fluidTol) return false; // Exclude Brinkman/solid regions

    const float velTol = 1e-8f;
    const float pPinTol = 1e-8f;

    // Right side is outlet by default when pressure is pinned to zero.
    // Do not reinterpret pressure-outlet faces as hard walls.
    if (std::abs(s.p(i, s.N)) <= pPinTol) return false;

    const bool vZero = std::abs(s.v(i, s.N)) <= velTol;
    const bool uZero = std::abs(s.u(i, s.N - 1)) <= velTol &&
                       std::abs(s.u(i + 1, s.N - 1)) <= velTol;
    return vZero && uZero;
}

ExternalWallFlags detectExternalNoSlipWalls(const SIMPLE& s)
{
    ExternalWallFlags flags;
    if (s.M < 2 || s.N < 2) return flags;

    // Side-level summary built from face-local checks.
    for (int j = 1; j < s.N - 1; ++j) {
        if (hasExternalNoSlipSouthForU(s, 1, j)) { flags.bottom = true; break; }
    }
    for (int j = 1; j < s.N - 1; ++j) {
        if (hasExternalNoSlipNorthForU(s, s.M - 1, j)) { flags.top = true; break; }
    }
    for (int i = 1; i < s.M - 1; ++i) {
        if (hasExternalNoSlipWestForV(s, i, 1)) { flags.left = true; break; }
    }
    for (int i = 1; i < s.M - 1; ++i) {
        if (hasExternalNoSlipEastForV(s, i, s.N - 1)) { flags.right = true; break; }
    }

    return flags;
}

// checkBoundaries is now inline in SIMPLE.h for performance
// (called ~4M times per iteration)
