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

// ============================================================================
// setVelocityBoundaryConditions: Apply velocity BCs to u and v fields
// ============================================================================
// Called after momentum solve (on uStar, vStar) and after velocity correction
// (on u, v) to ensure boundary values are enforced.
// ============================================================================
void SIMPLE::setVelocityBoundaryConditions(Eigen::MatrixXd& uIn,
                                           Eigen::MatrixXd& vIn)
{
    // Use the ramped inlet velocity set in runIterations()
    const double Uin = inletVelocity;

    // ---------------------------
    // Bottom wall (y = 0): no-slip
    // ---------------------------
    for (int j = 0; j < N; ++j) {
        uIn(0, j) = 0.0;        // u on bottom faces
    }
    for (int j = 0; j < N + 1; ++j) {
        vIn(0, j) = 0.0;        // v on bottom faces
    }

    // ---------------------------
    // Top wall (y = H): no-slip
    // ---------------------------
    for (int j = 0; j < N; ++j) {
        uIn(M, j) = 0.0;        // u on top faces
    }
    for (int j = 0; j < N + 1; ++j) {
        vIn(M - 1, j) = 0.0;    // v on top faces
    }

    // Left boundary (x = 0): set inlet velocity only on fluid cells
    for (int i = 1; i < M; ++i) {
        const int cellRow = i - 1;  // Corresponding row in cellType
        if (cellRow >= 0 && cellRow < static_cast<int>(cellType.rows()) && cellType(cellRow, 0) < 0.5) {
            // Fluid part of inlet: impose uniform velocity
            uIn(i, 0) = Uin;
        } else {
            // Solid part of inlet or out of bounds: no-slip
            uIn(i, 0) = 0.0;
        }
    }

    // v on left boundary: no normal flow (x-normal boundary, so v is tangential).
    // We simply set v = 0 everywhere on the inlet column for stability.
    for (int i = 0; i < M; ++i) {
        vIn(i, 0) = 0.0;
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
void SIMPLE::setPressureBoundaryConditions(Eigen::MatrixXd& pIn)
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

// ----------------------------------------------------------
// Geometry: 1 = solid, 0 = fluid (from cellType)
// ----------------------------------------------------------
double SIMPLE::checkBoundaries(int i, int j)
{
    // 1. Outer box (domain limits) are the only true "hard" boundaries
    // this function cares about now. Internal obstacles are handled
    // via alpha in the momentum equation, not via checkBoundaries.
    if (i <= 0) return 1;     // Top wall
    if (i >= M) return 1;     // Bottom wall
    if (j <= 0) return 1;     // Inlet (x = 0 plane)
    if (j >= N) return 1;     // Outlet (x = L plane)

    // 2. Internal obstacles are not treated as boundaries here.
    //    This keeps the SIMPLE algebra cleaner, and solids
    //    are handled through the drag/Brinkman term via alpha.
    return 0;
}
