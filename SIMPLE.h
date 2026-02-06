// ============================================================================
// File: SIMPLE.h
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Core interface and parameters for the incompressible laminar SIMPLE
//   solver on a structured Cartesian grid using the Finite Volume Method (FVM).
//
//   This header serves as the CENTRAL REGISTRY for:
//     1. All solver control parameters (relaxation, CFL ramps, convergence criteria)
//     2. Algorithm options (direct vs iterative pressure solve)
//     3. Physics options (2.5D model, Brinkman penalization for topology optimization)
//     4. Staggered grid field storage (u, v, p matrices)
//     5. Geometry and boundary condition data structures
//     6. Function declarations for utilities in Utilities/*.cpp
//
//   STAGGERED GRID LAYOUT (MAC - Marker And Cell):
//   ┌─────────────────────────────────────────────────────────────────────────┐
//   │  The staggered grid stores variables at different locations:            │
//   │                                                                         │
//   │     p(i,j)         p(i,j+1)                                            │
//   │       ●──────u(i,j)──────●      ● = pressure (cell centers)            │
//   │       │               │          ──│ = u-velocity (vertical faces)      │
//   │       │               │          ═══ = v-velocity (horizontal faces)    │
//   │    v(i,j)   CELL    v(i,j+1)                                           │
//   │       │    (i,j)      │                                                │
//   │       │               │         Grid indexing:                         │
//   │       ●──────u(i+1,j)─●           u: (M+1) x N  (at vertical faces)    │
//   │     p(i+1,j)      p(i+1,j+1)      v: M x (N+1)  (at horizontal faces)  │
//   │                                    p: (M+1) x (N+1) (cell centers+ghost)│
//   └─────────────────────────────────────────────────────────────────────────┘
//
//   COORDINATE SYSTEM:
//     - i = row index (y-direction), increases downward (fluid flow direction)
//     - j = column index (x-direction), increases to the right (main flow direction)
//     - Origin at top-left corner of the domain
//
// ============================================================================
#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <omp.h>
#include "Utilities/Timer.h"

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

class SIMPLE {

public:

    // =========================================================================
    // CONVERGENCE CONTROL
    // =========================================================================
    // These parameters determine when the solver stops iterating.
    // The solver runs until EITHER:
    //   1. All residuals drop below epsilon, OR
    //   2. Pressure drop stabilizes (if usePressureDropConvergence=true), OR
    //   3. maxIterations is reached
    
    float  epsilon       = 1e-7f;     // Target residual threshold (float precision = 1e-6 is safer)
    int    maxIterations = 1000;    // Maximum outer SIMPLE iterations (safety limit)
    
    // =========================================================================
    // OUTPUT & RESTART CONTROL
    // =========================================================================
    // Checkpointing allows resuming from saved state if solver crashes or
    // you want to refine the solution further.
    
    int    saveStateInterval = 1000;   // Save checkpoint every N iterations (0 = disable)
    bool   reuseInitialFields = false;// If true, load u.txt/v.txt/p.txt as initial guess
    std::string restartDirectory = "ExportFiles";  // Directory for restart files

    // =========================================================================
    // PRESSURE SAMPLING PLANES
    // =========================================================================
    // Physical x-coordinates [m] where pressure is sampled to compute pressure drop.
    // "Core" planes exclude inlet/outlet buffers for cleaner pressure drop measurement.
    // "Full" planes span the entire domain (auto-calculated: 0 to N*hx).
    
    float xPlaneCoreInlet  = 0.01f;   // Core inlet sampling plane [m] (after inlet buffer)
    float xPlaneCoreOutlet = 0.05f;  // Core outlet sampling plane [m] (before outlet buffer)

    // =========================================================================
    // PRESSURE-DROP BASED CONVERGENCE
    // =========================================================================
    // Alternative convergence criterion: stop when pressure drop stabilizes.
    // Useful when residuals oscillate but the engineering quantity of interest
    // (pressure drop) has already converged.
    
    bool   usePressureDropConvergence = true;   // Enable this convergence mode
    int    dpConvergenceWindow = 500;          // Moving window size [iterations]
    float  dpConvergencePercent = 1.0f;         // Converged if std_dev < this % of mean
    bool   usePressureDropSlopeGate = false;    // Additional check: slope must be near-zero
    int    dpSlopeWindowIters = 1000;           // Window for slope calculation
    float  dpSlopeMaxDelta   = 10.0f;           // Max allowed Δp change [Pa] over window
    
    // =========================================================================
    // PARALLELIZATION (OpenMP)
    // =========================================================================
    // Multi-threading for momentum loops and matrix assembly.
    // Best performance with number of PHYSICAL cores (not hyperthreads).
    
    int numThreads = 8;  // Number of OpenMP threads (recommend: physical core count)
    
    // =========================================================================
    // PRESSURE CORRECTION SOLVER OPTIONS
    // =========================================================================
    // Four options for solving the pressure-correction (Poisson) equation:
    //
    //   pressureSolverType = 0: SOR (Successive Over-Relaxation)
    //      - Red-black ordering, partially parallelized
    //      - Slowest iterative option
    //
    //   pressureSolverType = 1: Parallel CG (Jacobi preconditioned)
    //      - Fully parallelized (SpMV, dot products, AXPY)
    //      - Good performance, no external dependencies
    //
    //   pressureSolverType = 2: AMGCL (Algebraic Multigrid + CG) [FASTEST for large]
    //      - AMG preconditioner with CG Krylov solver
    //      - Best for large systems (>10k DOFs)
    //      - Requires compile with -DUSE_AMGCL flag
    //      - Include path: -I"ThermalSolver/amgcl"
    //
    //   pressureSolverType = 3: Direct LDLT (Eigen SimplicialLDLT)
    //      - Direct sparse Cholesky factorization
    //      - SERIAL (single-threaded) but no iterations
    //      - Best for small systems (<5k DOFs)
    //      - Pattern cached for speed across iterations
    //
    //   pressureSolverType = 4: AMGCL-CUDA (GPU version of AMGCL) [RECOMMENDED GPU]
    //      - Same algorithm as CPU AMGCL but runs on GPU
    //      - Requires NVIDIA GPU and CUDA Toolkit
    //      - Compile with build_with_amgcl_cuda.bat
    //      - Uses single precision (float) for 2x speed on GPU
    //      - BEST OPTION for GPU: Same convergence as CPU, but faster!
    //
    int pressureSolverType = 4;    // 0=SOR, 1=PCG, 2=AMGCL(CPU), 3=LDLT, 4=AMGCL-CUDA(GPU)
    bool parallelizeDirectSolver = true;  // Parallelize matrix assembly (recommended)
    
    // Solver iteration parameters:
    // Note: AMGCL typically converges in 5-30 iterations (AMG is very effective!)
    //       Parallel CG needs 50-200 iterations (Jacobi preconditioner is weaker)
    //       If AMGCL hits maxIter, try relaxing tolerance or check matrix conditioning
    int   maxPressureIter = 30;    // Max iterations (30 for AMGCL, 200 for PCG/SOR)
    float pTol = 1e-6f;            // Convergence tolerance (relaxed to 1e-5 for float precision)
    
    // SOR-specific parameters (only used when pressureSolverType = 0):
    float sorOmega = 0.0f;         // SOR relaxation factor (0 = auto-compute optimal ω)

    // =========================================================================
    // MOMENTUM SOLVER OPTIONS (Jacobi/SOR)
    // =========================================================================
    // Controls how momentum equations (U, V) are solved each SIMPLE iteration.
    //
    //   momentumSolverType = 0: Explicit (Point Gauss-Seidel) [ORIGINAL]
    //      - Updates each cell explicitly using neighbors from current iteration
    //      - Simple but slow convergence, especially for stiff problems
    //
    //   momentumSolverType = 1: Implicit Jacobi/SOR [RECOMMENDED]
    //      - Assembles full momentum matrix and solves with Jacobi/SOR
    //      - GPU: cuSPARSE SpMV + CUDA kernel (fast, parallel)
    //      - CPU: OpenMP-parallelized Jacobi/SOR
    //
    int momentumSolverType = 1;    // 0=Explicit, 1=Implicit Jacobi/SOR
    
    // Momentum solver tolerance and iterations
    // Momentum solver tolerance and iterations
    float  momentumTol = 1e-6f;      // Convergence tolerance for momentum solver
    int    maxMomentumIter = 400;    // Max iterations for Jacobi/SOR
    
    // SOR (Successive Over-Relaxation) for momentum solver
    // omega = 1.0: Pure Jacobi
    // omega > 1.0: SOR (faster convergence, 1.2-1.8 typical)
    // omega < 1.0: Under-relaxation (more stable)
    float  momentumSorOmega = 1.0f;  // Relaxation factor (1.0 = Jacobi, >1 = SOR)

    // =========================================================================
    // CONVECTION SCHEME
    // =========================================================================
    // Controls spatial discretization of convective (advective) terms:
    //   0 = First-Order Upwind (FOU): Simple, stable, diffusive
    //   1 = Second-Order Upwind (SOU): Higher accuracy, uses deferred correction
    //       to maintain diagonal dominance while improving flux approximation
    
    int convectionScheme = 0;  // 0 = FOU, 1 = SOU

    // =========================================================================
    // 2.5D REDUCED-ORDER MODEL
    // =========================================================================
    // For microchannel heat sinks with tall, thin fins, the 2.5D model captures
    // out-of-plane (z-direction) effects without solving full 3D equations:
    //   - Convection scaled by user factor (default 1.0 = off, e.g. 1.2 = 6/5)
    //   - Adds linear drag term: F = -(12μ/Ht²) * u  [classical depth-averaged parallel-plate friction]
    //
    // Reference: "Two-layer microchannel model" for topology optimization
    
    bool   enableTwoPointFiveD = false;    // Enable 2.5D reduced-order terms
    float  twoPointFiveDConvectionFactor = 1.2f; // Multiplier on convective fluxes when 2.5D is ON
    float  Ht_channel = 0.0f;             // Out-of-plane channel height [m] (from geometry file)
    float  twoPointFiveDSinkMultiplier = 1.0f; // Multiplier on classical base sink coefficient
        // Base = 12μ/Ht² (depth-averaged parallel-plate friction)

    // =========================================================================
    // DENSITY-BASED TOPOLOGY OPTIMIZATION (Brinkman Penalization)
    // =========================================================================
    // For topology optimization, geometry is described by a continuous density
    // field γ (gamma) rather than binary solid/fluid:
    //   - γ = 1.0 → pure fluid (no drag)
    //   - γ = 0.0 → pure solid (high drag = blocked flow)
    //   - 0 < γ < 1 → transition zone (gradual drag)
    //
    // Brinkman penalization adds a Darcy-like drag term to momentum:
    //   F = -α(γ) * u
    //
    // Per Haertel et al. (Eq. 17), α uses RAMP interpolation:
    //   α(γ) = α_max * (1 - γ) / (1 + q * γ)
    //
    // With q=1 (Haertel final value), this penalizes intermediate γ:
    //   - γ=1 (fluid): α = 0 (no drag)
    //   - γ=0.5 (buffer): α = 0.33 * α_max (strong drag)
    //   - γ=0 (solid): α = α_max (full drag)
    //
    // α_max should be tied to MATERIAL permeability (mesh/domain independent):
    //   α_max = μ / K_min
    // where K_min is the "solid-side" permeability [m^2].
    //
    // This is the physically clean SIMP/Brinkman form and remains consistent
    // across geometry sizes and mesh refinements.
    //
    // Reference: Haertel et al., "Topology optimization of microchannel heat sinks"
    
    // Default K_min keeps the same α_max as previous defaults:
    // Da = 1e-8 and L_ref = 1e-3 m -> K_min = 1e-14 m^2 lowr value=lower permeability
    float  brinkmanKMin = 1e-14f;       // Solid-side permeability K_min [m^2]
    float  brinkmanAlphaMax = 0.0f;     // Computed at runtime: μ / K_min
    float  brinkmanQ = 1.0f;            // RAMP convexity parameter (Haertel: 1-8, use 1 for strong penalization)

    // =========================================================================
    // PSEUDO-TRANSIENT TIME STEPPING
    // =========================================================================
    // Pseudo-transient continuation adds artificial time derivative to steady
    // equations: ∂u/∂t + [steady terms] = 0. This improves robustness for
    // difficult problems by allowing the solution to "evolve" toward steady state.
    //
    // The pseudo time step can be:
    //   - Global: same Δt everywhere (simpler, more stable)
    //   - Local (CFL-based): Δt = CFL * Δx / |u|  (faster convergence)
    //
    // CFL ramping increases CFL as residuals decrease, accelerating convergence
    // once the solution is close to steady state.
    
    float  timeStepMultiplier = 0.01f;    // Global Δt = multiplier * (Lx / U_inlet)
    float  timeStep = 0.0f;               // Computed global pseudo time step [s]
    bool   enablePseudoTimeStepping = false; // Master switch for pseudo-transient
    float  pseudoCFL = 0.1f;              // Current CFL number (updated by ramp)
    float  minPseudoSpeed = 0.05f;        // Minimum velocity for CFL calculation [m/s]
    bool   useLocalPseudoTime = true;     // true = local CFL, false = global Δt
    bool   logPseudoDtStats = false;      // Print min/max/avg pseudo-Δt each iteration
    
    float  pseudoRefLength = 0.0f;        // Reference length for CFL [m] (0 = auto)


    // Pseudo-time controller mode:
    //   0 = Legacy controls (fixed/Residual ramp/SER/line-search)
    //   1 = COMSOL-style manual CFL schedule (Eq. 3-66)
    //   2 = COMSOL-style multiplicative PID CFL controller (Eq. 20-7)
    int    pseudoControllerMode = 2;
    bool   pseudoUseMaxResidualMetric = true;  // true=max(U,V,Mass) RMS, false=Mass RMS
    bool   pseudoUseCFLRatioGate = true;       // Require CFL-ratio=1 for residual convergence
    float  pseudoCFLInfinity = 1.0e4f;         // COMSOL-like steady-state CFL_inf
    float  pseudoCFLRatio = 0.0f;              // min(log(CFL)/log(CFL_inf), 1)

    // COMSOL-style manual CFL schedule (Eq. 3-66):
    // CFL = a^min(n, s1)
    //     + if(n > o2, m2 * a^min(n-o2, s2), 0)
    //     + if(n > o3, m3 * a^min(n-o3, s3), 0)
    float  comsolManualBase = 1.3f;             // a
    int    comsolManualStage1Span = 9;          // s1
    int    comsolManualStage2Offset = 20;       // o2
    int    comsolManualStage2Span = 9;          // s2
    float  comsolManualStage2Multiplier = 9.0f; // m2
    int    comsolManualStage3Offset = 40;       // o3
    int    comsolManualStage3Span = 9;          // s3
    float  comsolManualStage3Multiplier = 90.0f;// m3

    // COMSOL-style multiplicative PID controller (Eq. 20-7)
    float  comsolPIDInitialCFL = 1.3f;   // Initial CFL (order one)
    float  comsolPIDTol = 1e-3f;         // Target nonlinear error estimate
    float  comsolPIDkP = 0.7f;           // Proportional exponent
    float  comsolPIDkI = 0.3f;           // Integral exponent
    float  comsolPIDkD = 0.0f;           // Derivative exponent
    float  comsolPIDGainMin = 0.2f;      // Clamp on multiplicative CFL gain
    float  comsolPIDGainMax = 2.5f;      // Clamp on multiplicative CFL gain
    float  comsolPIDCFLMin = 1.0f;       // Hard lower bound (CFL >= 1)
    float  comsolPIDCFLMax = 1.0e6f;     // Hard upper bound (safety)
    float  comsolErrorPrev = -1.0f;      // e_(n-1)
    float  comsolErrorPrevPrev = -1.0f;  // e_(n-2)

    // -------------------------------------------------------------------------
    // Switched Evolution Relaxation (SER) - Mulder & van Leer
    // -------------------------------------------------------------------------
    // SER adjusts CFL based on residual reduction ratio:
    //   CFL^k = min(CFL^(k-1) * |R^(k-1)| / |R^k|, CFL_max)
    // If residual decreases, CFL increases proportionally.
    // If residual increases, CFL decreases proportionally.
    // When enabled, this DISABLES other CFL ramping mechanisms.
    bool   enableSER        = false;       // Enable SER (disables other CFL ramps)
    float  serCFLMin        = 0.01f;       // Minimum CFL for SER
    float  serCFLMax        = 10.0f;      // Maximum CFL for SER (reduced for stability)
    float  serResidPrev     = 0.0f;       // Previous iteration L2 residual (for tracking)
    float  serSmooth        = 0.5f;       // Smoothing factor (0 = no smoothing, 1 = full new) - increased for stability
    int    serMinIter       = 10;         // Don't apply SER until this many iterations - increased for stability
    float  serMaxIncrease   = 1.1f;      // Max CFL increase per iteration (5%)
    float  serMaxDecrease   = 0.7f;       // Max CFL decrease per iteration (Lower value is more aggressive decrease by 30%)
    bool   serUseMaxResid   = true;       // true = use max(U,V,Mass), false = use Mass only
    
    // Line search for robustness (coupled with SER)
    // Ensures solution updates actually reduce residuals
    // If residual increases too much, temporarily reduce CFL to stabilize
    bool   enableLineSearch = false;       // Enable backtracking line search
    float  lsAlphaMin       = 0.1f;       // Minimum step size (don't go below this)
    float  lsAlphaReduce    = 0.3f;       // Reduction factor for backtracking (try α, α/3, α/9, ...)
    int    lsMaxTries       = 4;          // Max backtracking attempts
    float  lsResidIncreaseTol = 1.01f;    // Backtrack if residual increases by 1% (extreme sensitivity)
    float  lsCurrentAlpha   = 1.0f;       // Current line search step size (1.0 = full step)
    int    lsBacktrackCount = 0;          // Number of backtracks in current iteration
    
    // -------------------------------------------------------------------------
    // CFL ramping parameters (residual-based acceleration):
    // -------------------------------------------------------------------------
    // NOTE: Disabled when enableSER = true
    bool   enableCflRamp    = false;       // Enable automatic CFL increase
    float  pseudoCFLInitial = 0.1f;       // Starting CFL (0.1 is a more conservative value)
    float  pseudoCFLMax     = 30.0f;      // Maximum CFL (reduced for stability) 
    float  cflRampStartRes  = 1e-4f;      // Begin ramping later
    float  cflRampExponent  = 0.8f;       // CFL ∝ (Res_start / Res_current)^exponent
    float  cflRampSmooth    = 0.1f;       // Smoothing factor (0.1 = 90% old + 10% new)
    
    // Transient residual tracking (monitors convergence to true steady state):
    // At true steady state, these should approach zero
    float  transientResidU = 0.0f;   // max |ρV/Δt * (u_new - u_old)| for U-momentum
    float  transientResidV = 0.0f;   // max |ρV/Δt * (v_new - v_old)| for V-momentum

    // =========================================================================
    // GRID DIMENSIONS & SPACING
    // =========================================================================
    // Structured Cartesian grid with uniform spacing.
    // These values are read from ExportFiles/fluid_params.txt.
    
    // Internal padded counts:
    // - Physical cell counts from file are (M-1) x (N-1)
    // - Interior solver loops i=1..M-1, j=1..N-1 then cover all physical cells.
    int M = 0;       // Internal row count (physical Ny + 1)
    int N = 0;       // Internal column count (physical Nx + 1)
    int N_in_buffer = 0;   // Inlet buffer zone columns (for thermal cropping)
    int N_out_buffer = 0;  // Outlet buffer zone columns (for thermal cropping)
    float hy = 0.0f; // Cell height [m] (Δy)
    float hx = 0.0f; // Cell width [m] (Δx)

    // =========================================================================
    // FLUID PROPERTIES
    // =========================================================================
    // Properties for water at 20°C. These are HARDCODED here and NOT read from
    // the geometry file, ensuring consistent physics across geometry variations.
    
    float rho = 997.0f;       // Fluid density [kg/m³]
    float eta = 0.00089f;     // Dynamic viscosity [Pa·s] (μ)

    // =========================================================================
    // INLET VELOCITY & RAMPING
    // =========================================================================
    // Inlet ramping gradually increases velocity from 10% to 100% of target
    // over rampSteps iterations. This prevents numerical instabilities at startup.
    
    float  targetVel        = 0.0f;   // Final inlet velocity [m/s] (from geometry file)
    float  inletVelocity    = 0.0f;   // Current inlet velocity [m/s] (may be ramped)
    bool   enableInletRamp  = false;  // Enable gradual inlet velocity ramp
    int    rampSteps        = 1000;   // Iterations to reach full inlet velocity

    // =========================================================================
    // UNDER-RELAXATION FACTORS
    // =========================================================================
    // Under-relaxation stabilizes the iterative process by blending new values
    // with old: φ_new = α·φ_calculated + (1-α)·φ_old
    //
    // Lower values = more stable but slower convergence
    // Typical ranges: velocity 0.3-0.8, pressure 0.1-0.3
    
    float uvAlpha = 0.7f;    // Velocity under-relaxation factor (reduced for stability)
    float pAlpha  = 0.3f;    // Pressure under-relaxation factor (reduced for stability)

    // =========================================================================
    // RESIDUAL-BASED ARTIFICIAL VISCOSITY (Stabilization)
    // =========================================================================
    // Adds a small, localized viscosity in cells with large momentum residuals.
    // This is lagged by one iteration (uses previous residuals) to keep the
    // momentum solve stable and cheap. Intended for stabilization only.
    bool  enableResidualViscosity   = false;
    float residualViscCoeff         = 2.0f;  // μ_art = C * ρ * h * |R_u|  (dimensionless C)
    float residualViscMinVel        = 0.0f;   // Threshold [m/s] below which μ_art = 0
    float residualViscMaxFactor     = 100.0f;   // Clamp: μ_art ≤ maxFactor * μ
    int   residualViscSmoothIters   = 1;      // Residual smoothing passes (0 = off)
    float residualViscSmoothWeight  = 1.0f;   // Blend toward neighbor avg [0..1]

    // =========================================================================
    // RESIDUAL TRACKING
    // =========================================================================
    // Residuals measure how far the current solution is from satisfying the
    // governing equations. They should decrease each iteration toward epsilon.
    //
    // Reporting/Convergence:
    // - If enableNormalizedResiduals=true, we report RMS residuals normalized
    //   by their initial RMS values (common CFD practice).
    // - If false, we report raw RMS residuals (no normalization).

    bool  enableNormalizedResiduals = false;  // Master switch for RMS normalized residual reporting

    float residU    = 1.0f;  // Reported U residual (normalized RMS or raw RMS)
    float residV    = 1.0f;  // Reported V residual (normalized RMS or raw RMS)
    float residMass = 1.0f;  // Reported continuity residual (normalized RMS or raw RMS)

    // Raw max residuals (always computed)
    float residU_max    = 1.0f;
    float residV_max    = 1.0f;
    float residMass_max = 1.0f;

    // RMS residuals (always computed)
    float residU_RMS    = 0.0f;
    float residV_RMS    = 0.0f;
    float residMass_RMS = 0.0f;

    // Normalization references (set when residual exceeds a small floor)
    float residU_RMS0    = 0.0f;
    float residV_RMS0    = 0.0f;
    float residMass_RMS0 = 0.0f;
    bool  residU_RMS0_set = false;
    bool  residV_RMS0_set = false;
    bool  residMass_RMS0_set = false;
    float residNormFloorVel  = 1e-6f;   // Floor for RMS velocity residuals [m/s]
    float residNormFloorMass = 1e-12f;  // Floor for RMS mass residuals [kg/s]

    // =========================================================================
    // PSEUDO-TIME STEP STATISTICS
    // =========================================================================
    // Diagnostic structure for tracking local pseudo-Δt distribution.
    // Useful for debugging CFL-based time stepping.
    
    struct PseudoDtStats {
        float min = 0.0f;       // Minimum pseudo-Δt in the domain
        float max = 0.0f;       // Maximum pseudo-Δt in the domain
        float avg = 0.0f;       // Average pseudo-Δt
        long long samples = 0;  // Number of cells sampled
        bool valid = false;     // Whether stats were computed this iteration
    };

    PseudoDtStats pseudoStatsU;  // Stats for U-momentum pseudo-Δt
    PseudoDtStats pseudoStatsV;  // Stats for V-momentum pseudo-Δt

    // =========================================================================
    // STAGGERED VELOCITY FIELDS
    // =========================================================================
    // Velocities are stored at cell FACES (not centers) to ensure natural
    // coupling with pressure gradients and prevent checkerboard oscillations.
    //
    // u: horizontal velocity at VERTICAL faces, size (M+1) x N
    //    u(i,j) = velocity at the left face of cell (i-1,j) / right face of (i-1,j-1)
    //
    // v: vertical velocity at HORIZONTAL faces, size M x (N+1)
    //    v(i,j) = velocity at the bottom face of cell (i,j-1) / top face of (i-1,j-1)
    
    Eigen::MatrixXf u;       // Current x-velocity field
    Eigen::MatrixXf v;       // Current y-velocity field
    Eigen::MatrixXf uStar;   // Intermediate u (after momentum solve, before correction)
    Eigen::MatrixXf vStar;   // Intermediate v (after momentum solve, before correction)
    Eigen::MatrixXf uOld;    // Previous iteration u (for residual calculation)
    Eigen::MatrixXf vOld;    // Previous iteration v (for residual calculation)

    // =========================================================================
    // PRESSURE FIELD (Cell-Centered)
    // =========================================================================
    // Pressure is stored at cell CENTERS, with ghost cells at boundaries.
    // Size: (M+1) x (N+1) to accommodate ghost layers.
    //
    // pStar is the pressure CORRECTION (p' in SIMPLE algorithm), not the
    // intermediate pressure. Final pressure: p = p_old + α_p * p'
    
    Eigen::MatrixXf p;       // Pressure field [Pa]
    Eigen::MatrixXf pStar;   // Pressure correction field [Pa]

    // =========================================================================
    // MOMENTUM EQUATION COEFFICIENTS
    // =========================================================================
    // These "d" coefficients appear in the velocity correction equations:
    //   u' = dE * (p'_P - p'_E)  for U-momentum
    //   v' = dN * (p'_P - p'_N)  for V-momentum
    //
    // They represent the sensitivity of velocity to pressure gradient.
    
    Eigen::MatrixXf dE;      // d-coefficient for U-velocity correction
    Eigen::MatrixXf dN;      // d-coefficient for V-velocity correction
    Eigen::MatrixXf b;       // Work array (used internally)

    // =========================================================================
    // GEOMETRY FIELDS
    // =========================================================================
    // cellType: Raw geometry input (0=fluid, 1=solid, 0-1=buffer in V3)
    // gamma:    Porosity/density field (1=fluid, 0=solid) - inverted from cellType
    // alpha:    Brinkman penalization coefficient (high = solid, low = fluid)
    
    Eigen::MatrixXf cellType;  // Geometry from file (0=fluid, 1=solid)
    Eigen::MatrixXf alpha;     // Brinkman α field: α = α_max * (1 - γ)
    Eigen::MatrixXf gamma;     // Density field: γ = 1 (fluid), γ = 0 (solid)

    // Residual-based artificial viscosity fields
    Eigen::MatrixXf uResidualFace; // |uStar - uOld| on u-faces
    Eigen::MatrixXf vResidualFace; // |vStar - vOld| on v-faces
    Eigen::MatrixXf muArt;         // Artificial viscosity at cell centers

    // =========================================================================
    // CONSTRUCTOR & INITIALIZATION
    // =========================================================================
    
    SIMPLE();                                           // Constructor: loads params, geometry, initializes fields
    void loadParameters(const std::string& paramsFile); // Read grid size, spacing, inlet velocity from file
    void initializeMemory();                            // Allocate all field matrices

    // =========================================================================
    // MAIN SOLVER METHODS
    // =========================================================================
    
    void runIterations();                               // Main iteration loop: runs until convergence or max iterations
    float calculateStep(int& pressureIterations);       // Perform ONE SIMPLE iteration (momentum → pressure → correction)

    // =========================================================================
    // SOLVER SUBCOMPONENTS (implemented in Utilities/*.cpp)
    // =========================================================================
    
    bool solvePressureSystem(int& pressureIterations, float& localResidMass);  // Pressure correction solve
    void updateCflRamp(float currRes);                 // Update pseudo-CFL based on current residual

    void updateCflSER(float currentResidL2, int iteration);  // Switched Evolution Relaxation (Mulder & van Leer)
    void applyLineSearch(float currentResidL2);        // Backtracking line search for robustness
    void updatePseudoTimeController(float currentError, int completedIteration); // Unified pseudo-time controller
    void resetPseudoTimeControllerState();             // Reset PID/manual state at startup
    float computeComsolManualCFL(int nonlinearIteration) const; // COMSOL Eq. 3-66
    float computePseudoCFLRatio(float cfl) const;      // min(log(CFL)/log(CFL_inf), 1)
    float computePseudoDtFromSpeed(float speed) const; // Local/global pseudo dt utility
    float pseudoSpeedAtU(int i, int j) const;          // Velocity magnitude at u-face
    float pseudoSpeedAtV(int i, int j) const;          // Velocity magnitude at v-face

    // =========================================================================
    // GEOMETRY LOADING
    // =========================================================================
    
    void loadTopology(const std::string& fileName);     // Read cellType from geometry_fluid.txt
    void loadDensityField();                            // Convert cellType → gamma (invert 0↔1)
    void buildAlphaFromDensity();                       // Compute α = α_max * (1 - γ)
    void updateMomentumResiduals();                     // Compute per-face momentum residuals
    void updateResidualViscosity();                     // Build residual-based μ_art field

    // =========================================================================
    // BOUNDARY CONDITIONS
    // =========================================================================
    
    void setVelocityBoundaryConditions(Eigen::MatrixXf& uIn, Eigen::MatrixXf& vIn);  // Apply inlet/outlet/wall BCs
    void setPressureBoundaryConditions(Eigen::MatrixXf& pIn);                         // Apply pressure BCs (outlet = 0)
    // Inlined for performance - called ~4M times per iteration
    inline float checkBoundaries(int i, int j) const {
        if (i <= 0 || i >= M || j <= 0 || j >= N) return 1.0f;
        return 0.0f;
    }
    void paintBoundaries();                             // Debug: export boundary map to file

    // =========================================================================
    // OUTPUT METHODS
    // =========================================================================
    
    void saveAll();                                     // Save all fields (u, v, p, VTK) to ExportFiles/
    void saveMatrix(Eigen::MatrixXf inputMatrix, std::string fileName);  // Save single matrix to text file
    void initLogFiles(std::ofstream& residFile, std::ofstream& dpFile);  // Create residual/pressure log files
    void printIterationHeader() const;                  // Print column headers to console
    void writeIterationLogs(std::ofstream& residFile,   // Write iteration data to log files
                            std::ofstream& dpFile,
                            int iter,
                            float corePressureDrop,
                            float fullPressureDrop,
                            float coreStaticDrop,
                            float fullStaticDrop,
                            float pseudoCFL,
                            float pseudoCFLRatio);
    void printIterationRow(int iter,                    // Print one row of iteration data to console
                           float residMassVal,
                           float residUVal,
                           float residVVal,
                           float maxTransRes,
                           float corePressureDrop,
                           float fullPressureDrop,
                           float iterTimeMs,
                           int pressureIterations,
                           float pseudoCFL,
                           float pseudoCFLRatio) const;
    void printStaticDp(int iter,                        // Print static pressure drop (debug)
                       float coreStaticDrop,
                       float fullStaticDrop) const;

};

// =============================================================================
// INLINE HELPER FUNCTIONS
// =============================================================================
// These helpers provide fast lookup for fluid masks and interpolated values
// at staggered grid locations. They are used extensively in the momentum
// assembly loops (iterations.cpp) for efficiency.

// -----------------------------------------------------------------------------
// Fluid Mask Lookups
// -----------------------------------------------------------------------------
// Check if a given u/v/p location is in the fluid region.
// Uses precomputed boolean masks stored as flat vectors for cache efficiency.
// Returns true if the location should be solved, false if it's solid (skip).

// -----------------------------------------------------------------------------
// Alpha (Brinkman Penalization) Interpolation
// -----------------------------------------------------------------------------
// Since alpha is defined at cell CENTERS but u/v are at cell FACES, we need
// to interpolate alpha to the velocity locations using a simple average of
// the two adjacent cells.
//
// For u(i,j): average alpha from cells (i-1, j-1) and (i-1, j)  [left & right]
// For v(i,j): average alpha from cells (i-1, j-1) and (i, j-1)  [top & bottom]

// Get Brinkman alpha at U-velocity location (i,j)
// Returns the average of alpha from the two cells sharing this u-face
inline float alphaAtU(const SIMPLE& s, int i, int j) {
    const auto& alpha = s.alpha;
    const int M = s.M;
    const int N = s.N;
    
    // Boundary check: return 0 (no drag) if outside valid interior
    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 0.0;
    
    // Cell indices for the two cells sharing this u-face
    int ci = i - 1;      // Row index in cellType/alpha
    int cj1 = j - 1;     // Left cell column
    int cj2 = j;         // Right cell column
    
    // Additional bounds checks (defensive programming)
    if (ci < 0 || ci >= alpha.rows()) return 0.0f;
    if (cj1 < 0 || cj1 >= alpha.cols()) return 0.0f;
    if (cj2 < 0 || cj2 >= alpha.cols()) return alpha(ci, cj1);
    
    // Return arithmetic average of left and right cell alphas
    return 0.5f * (alpha(ci, cj1) + alpha(ci, cj2));
}

// Get Brinkman alpha at V-velocity location (i,j)
// Returns the average of alpha from the two cells sharing this v-face
inline float alphaAtV(const SIMPLE& s, int i, int j) {
    const auto& alpha = s.alpha;
    const int M = s.M;
    const int N = s.N;
    
    // Boundary check
    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 0.0;
    
    // Cell indices for the two cells sharing this v-face
    int ci1 = i - 1;     // Top cell row
    int ci2 = i;         // Bottom cell row
    int cj = j - 1;      // Column index
    
    // Additional bounds checks
    if (cj < 0 || cj >= alpha.cols()) return 0.0f;
    if (ci1 < 0 || ci1 >= alpha.rows()) return 0.0f;
    if (ci2 < 0 || ci2 >= alpha.rows()) return alpha(ci1, cj);
    
    // Return arithmetic average of top and bottom cell alphas
    return 0.5f * (alpha(ci1, cj) + alpha(ci2, cj));
}

// ---------------------------------------------------------------------------
// Residual-based artificial viscosity interpolation
// ---------------------------------------------------------------------------
// μ_art is defined at cell centers; interpolate to velocity faces similarly to α.
inline float muArtAtU(const SIMPLE& s, int i, int j) {
    const auto& muArt = s.muArt;
    const int M = s.M;
    const int N = s.N;

    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 0.0f;

    int ci = i - 1;
    int cj1 = j - 1;
    int cj2 = j;

    if (ci < 0 || ci >= muArt.rows()) return 0.0f;
    if (cj1 < 0 || cj1 >= muArt.cols()) return 0.0f;
    if (cj2 < 0 || cj2 >= muArt.cols()) return muArt(ci, cj1);

    return 0.5f * (muArt(ci, cj1) + muArt(ci, cj2));
}

inline float muArtAtV(const SIMPLE& s, int i, int j) {
    const auto& muArt = s.muArt;
    const int M = s.M;
    const int N = s.N;

    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 0.0f;

    int ci1 = i - 1;
    int ci2 = i;
    int cj = j - 1;

    if (cj < 0 || cj >= muArt.cols()) return 0.0f;
    if (ci1 < 0 || ci1 >= muArt.rows()) return 0.0f;
    if (ci2 < 0 || ci2 >= muArt.rows()) return muArt(ci1, cj);

    return 0.5f * (muArt(ci1, cj) + muArt(ci2, cj));
}

// -----------------------------------------------------------------------------
// Gamma (Density/Porosity) Interpolation
// -----------------------------------------------------------------------------
// Same logic as alpha interpolation, but for the gamma (porosity) field.
// Gamma is used in density-based topology optimization:
//   gamma = 1.0 → fluid (full permeability)
//   gamma = 0.0 → solid (zero permeability)
//   0 < gamma < 1 → transition zone

// Get porosity gamma at U-velocity location (i,j)
// Returns the average of gamma from the two cells sharing this u-face
// Default to 1.0 (fluid) at boundaries for stability
inline float gammaAtU(const SIMPLE& s, int i, int j) {
    const auto& gamma = s.gamma;
    const int M = s.M;
    const int N = s.N;
    
    // Default to fluid at boundaries (conservative for flow)
    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 1.0;
    
    int ci = i - 1;
    int cj1 = j - 1;
    int cj2 = j;
    
    if (ci < 0 || ci >= gamma.rows()) return 1.0f;
    if (cj1 < 0 || cj1 >= gamma.cols()) return 1.0f;
    if (cj2 < 0 || cj2 >= gamma.cols()) return gamma(ci, cj1);
    
    return 0.5f * (gamma(ci, cj1) + gamma(ci, cj2));
}

// Get porosity gamma at V-velocity location (i,j)
// Returns the average of gamma from the two cells sharing this v-face
inline float gammaAtV(const SIMPLE& s, int i, int j) {
    const auto& gamma = s.gamma;
    const int M = s.M;
    const int N = s.N;
    
    // Default to fluid at boundaries
    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 1.0;
    
    int ci1 = i - 1;
    int ci2 = i;
    int cj = j - 1;
    
    if (cj < 0 || cj >= gamma.cols()) return 1.0f;
    if (ci1 < 0 || ci1 >= gamma.rows()) return 1.0f;
    if (ci2 < 0 || ci2 >= gamma.rows()) return gamma(ci1, cj);
    
    return 0.5f * (gamma(ci1, cj) + gamma(ci2, cj));
}

// -----------------------------------------------------------------------------
// External Hard-Wall / Half-Cell Diffusion Detection
// -----------------------------------------------------------------------------
// Half-cell diffusion is applied FACE-LOCALLY (not side-global) for momentum
// control volumes adjacent to external no-slip boundaries.
//
// These helpers intentionally exclude Brinkman/solid regions by requiring the
// adjacent velocity location to be nearly pure fluid (gamma ~ 1).
//
// U-momentum: tangential to horizontal boundaries -> south/north checks.
// V-momentum: tangential to vertical boundaries   -> west/east checks.
//
// fluidTol default is strict so only near-pure-fluid locations get correction.
bool hasExternalNoSlipSouthForU(const SIMPLE& s, int i, int j, float fluidTol = 0.999f);
bool hasExternalNoSlipNorthForU(const SIMPLE& s, int i, int j, float fluidTol = 0.999f);
bool hasExternalNoSlipWestForV (const SIMPLE& s, int i, int j, float fluidTol = 0.999f);
bool hasExternalNoSlipEastForV (const SIMPLE& s, int i, int j, float fluidTol = 0.999f);

// Side-level summary (legacy logging helper) derived from face-local checks.
struct ExternalWallFlags {
    bool left = false;
    bool right = false;
    bool bottom = false;
    bool top = false;
};

ExternalWallFlags detectExternalNoSlipWalls(const SIMPLE& s);

// -----------------------------------------------------------------------------
// Second-Order Upwind (SOU) Deferred Correction Functions
// -----------------------------------------------------------------------------
// These compute the deferred correction source term for second-order upwind.
// The correction is: Sdc = Σ_face [ F * (φ_SOU - φ_FOU) ]
// This adds higher-order accuracy while maintaining the stable FOU matrix.
// Implemented in convection.cpp.

// Compute SOU deferred correction for U-momentum at location (i,j)
// Fe, Fw, Fn, Fs are the mass flux rates at east/west/north/south faces
float computeSOUCorrectionU(const SIMPLE& s, int i, int j,
                             float Fe, float Fw, float Fn, float Fs);

// Compute SOU deferred correction for V-momentum at location (i,j)
float computeSOUCorrectionV(const SIMPLE& s, int i, int j,
                             float Fe, float Fw, float Fn, float Fs);

// -----------------------------------------------------------------------------
// Pseudo-Time Statistics Logging
// -----------------------------------------------------------------------------
// Prints min/max/avg pseudo-Δt statistics for debugging CFL-based time stepping.
// Called once per iteration if logPseudoDtStats is enabled.

void logPseudoStats(const SIMPLE& solver, const char* label, const SIMPLE::PseudoDtStats& stats);

// =============================================================================
// POST-PROCESSING: Pressure Sampling at Physical Locations
// =============================================================================
// These structures and functions allow sampling pressure and velocity at
// arbitrary physical x-coordinates (not just cell boundaries). This enables
// accurate pressure-drop measurements independent of mesh alignment.
//
// Implemented in postprocessing.cpp.

// Structure to hold metrics sampled at a single vertical plane (constant x)
struct PlaneMetrics {
    float flowArea   = 0.0f;  // Total open (fluid) cross-sectional area [m²/m depth]
    float massFlux   = 0.0f;  // Sum of |mass flow| through plane [kg/s per m depth]
    float avgStatic  = 0.0f;  // Area-weighted average STATIC pressure [Pa]
    float avgDynamic = 0.0f;  // Area-weighted average DYNAMIC pressure [Pa] = ½ρV²
    float avgTotal   = 0.0f;  // Mass-flow-weighted TOTAL pressure [Pa] = static + dynamic
    bool   valid      = false;// True if at least one fluid cell was found at this x
};

// Sample pressure and velocity fields at a given physical x-coordinate [m]
// Uses linear interpolation between adjacent cell columns.
// Skips rows unless both adjacent cells are pure fluid (gamma ~= 1).
PlaneMetrics samplePlaneAtX(const SIMPLE& solver, float xPhysical);

// Sample metrics directly on a domain boundary patch (strict boundary-face integral).
// inletBoundary=true  -> x=0 inlet patch
// inletBoundary=false -> x=L outlet patch
// Uses boundary-face static pressure and boundary-face velocities.
// Includes only faces adjacent to pure-fluid cells (gamma ~= 1).
PlaneMetrics sampleBoundaryPatch(const SIMPLE& solver, bool inletBoundary);

// Debug helper: print detailed plane metrics to console
void printPlaneInfo(const char* name, float xPhysical, const PlaneMetrics& m);

// =============================================================================
// MOMENTUM SOLVER: Implicit AMG (STAR-CCM+ Style)
// =============================================================================
// These functions implement the implicit momentum solve using AMG with
// Gauss-Seidel relaxation, matching STAR-CCM+ methodology.
// Implemented in Utilities/momentum_solver.cpp.

// Solve U-momentum equation implicitly using AMG
// Returns: number of AMG iterations, updates solver.uStar in place
int solveUMomentumImplicit(SIMPLE& solver, float& residU);

// Solve V-momentum equation implicitly using AMG
// Returns: number of AMG iterations, updates solver.vStar in place
int solveVMomentumImplicit(SIMPLE& solver, float& residV);
