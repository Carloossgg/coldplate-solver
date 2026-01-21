// File: SIMPLE.h
// Author: Peter Tcherkezian
// Description: Core interface and parameters for the incompressible laminar SIMPLE/SIMPLEC solver on a structured grid.
//              Central registry of all solver controls (relaxation, CFL/pseudo-time ramps, scheme toggles), staggered
//              field storage, boundary/geometry inputs, and helper declarations used by Utilities/*. Adjust physics or
//              numerics (SIMPLE vs SIMPLEC, FOU vs SOU, direct vs SOR) here; implementations live in SIMPLE.cpp and
//              Utilities/.
#pragma once

#include <iostream>
#include <fstream>
#include <cmath>
#include <string>
#include <vector>
#include <chrono>
#include <deque>
#include <omp.h>

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <Eigen/SparseCholesky>

class SIMPLE {

public:

    // Convergence control
    double epsilon       = 1e-7;
    int    maxIterations = 100000;  // More iterations for complex geometries
    
    // Output control
    int    saveStateInterval = 600;  // Save state (checkpoint) every N iterations (0 = disable checkpoints)
    
    // Restart control
    bool   reuseInitialFields = false;  // Load previous solution as initial guess
    std::string restartDirectory = "ExportFiles";  // Directory containing u.txt, v.txt, p.txt

    // Pressure sampling planes (physical x-coordinates in meters)
    // User can configure these to sample at any x-location in the domain
    // Set to -1.0 to use automatic values (0.0 for inlet, domain end for outlet)
    double xPlaneCoreInlet  = 0.01;   // Core inlet plane [m] (default 10 mm = after typical inlet buffer)
    double xPlaneCoreOutlet = 0.050;   // Core outlet plane [m] (default 50 mm = before typical outlet buffer)
    // Full system planes are auto-calculated: inlet=0.0, outlet=N*hx (domain boundaries)

    // Pressure drop convergence (fluid region only)
    bool   usePressureDropConvergence = true;   // Enable pressure drop based convergence
    int    dpConvergenceWindow = 1500;           // Number of iterations to look back
    double dpConvergencePercent =1.0;         // Converged if change < this % over window
    bool   usePressureDropSlopeGate = false;    // Additional slope-based gate
    int    dpSlopeWindowIters = 1000;          // Window length for slope check
    double dpSlopeMaxDelta   = 10.0;           // Max allowed Δp change (Pa) over slope window
    
    // Parallelization (OpenMP)
    int numThreads = 8;  // Number of CPU cores to use (physical cores recommended)
    
    // Algorithm selection
    bool useSIMPLEC = false;       // If true, use SIMPLEC algorithm; if false, use standard SIMPLE algorithm
    
    // Pressure correction solver options
    bool useDirectPressureSolver = true;  // If true, use direct sparse matrix solver; if false, use iterative SOR
    bool parallelizeDirectSolver = true;  // If true, parallelize matrix assembly and mapping (recommended)
    int maxPressureIter = 50;      // Inner pressure sweeps per SIMPLE iteration (increase for fine mesh) - only used if useDirectPressureSolver = false
    double sorOmega = 0.0;         // SOR relaxation factor (0 = auto-compute optimal) - only used if useDirectPressureSolver = false
    double pTol = 1e-4;            // Absolute tolerance for pressure correction convergence - only used if useDirectPressureSolver = false

    // Convection scheme: 0 = first-order upwind, 1 = second-order upwind (SOU)
    int convectionScheme = 0;      // 0 = first-order upwind, 1 = second-order upwind  (SOU)

    // 2.5D reduced-order option (from two-layer microchannel model)
    bool   enableTwoPointFiveD = true; // TRUE ⇒ scale convection by 6/7 and add -(5μ/2Ht^2) u sink
    double Ht_channel = 0.0;            // Out-of-plane height [m] (read from geometry export; not hardcoded)
    double twoPointFiveDSinkMultiplier = 4.8; // =1 uses (5/2)μ/Ht^2; set ~4.8 to match parallel-plate 12μ/Ht^2

    // Pseudo-transient time step control
    double timeStepMultiplier = 0.01; // multiplier for Lx/U when computing pseudo dt (adjusting the global time step)
    double timeStep = 0.0;           // max pseudo time step (computed after geometry load)
    bool   enablePseudoTimeStepping = true; // FALSE ⇒ pseudoCFL/timeStep ignored everywhere
    double pseudoCFL = 0.1;         // current target CFL (overwritten by ramp if it is ON)
    double minPseudoSpeed = 0.05;   // minimum |velocity| used in CFL estimate
    bool   useLocalPseudoTime = true; // TRUE ⇒ use CFL-based local dt; FALSE ⇒ use global timeStep only
    bool   logPseudoDtStats = false;
    
    // Fixed reference length for CFL calculation (mesh-independent)
    double pseudoRefLength = 0.0;   // Reference length [m] (auto-computed from domain if 0)

    // CFL ramping control (residual-based)
    bool   enableCflRamp    = true; // TRUE ⇒ pseudoCFL is driven by the ramp logic below
    double pseudoCFLInitial = 0.1;   // starting CFL before ramp (used when enableCflRamp = true)
    double pseudoCFLMax     = 5;   // maximum allowed CFL during ramp
    double cflRampStartRes  = 5e-4;  // mass residual at which ramping starts
    double cflRampExponent  = 0.8;   // exponent in (RefRes / currRes)^exp
    double cflRampSmooth    = 0.1;   // smoothing factor (0.1 => 0.9*old + 0.1*new)
    
    // Transient term residual tracking (should → 0 at true steady state)
    double transientResidU = 0.0;   // max |transCoeff * (u - uOld)| for U-momentum
    double transientResidV = 0.0;   // max |transCoeff * (v - vOld)| for V-momentum

    // Grid dimensions
    int M = 0;   // rows (y-direction)
    int N = 0;   // columns (x-direction)

    // Buffer zones
    int N_in_buffer = 0;
    int N_out_buffer = 0;

    // Grid spacing
    double hy = 0.0;  // cell height
    double hx = 0.0;  // cell width

    // Fluid properties (water at 20°C)
    double rho = 997.0;       // density [kg/m³]
    double eta = 0.00089;     // dynamic viscosity [Pa·s]


    double targetVel        = 0.0;
    double inletVelocity    = 0.0;
    bool   enableInletRamp  = false;   // Toggle inlet velocity ramping
    int    rampSteps        = 1000;   // Ramp duration in SIMPLE iterations
    // Under-relaxation factors tuned for SIMPLE
    double uvAlpha = 0.5;    // velocity relaxation 
    double pAlpha  = 0.2;    // pressure relaxation 

    // Residuals
    double residU    = 1.0;
    double residV    = 1.0;
    double residMass = 1.0;

    struct PseudoDtStats {
        double min = 0.0;
        double max = 0.0;
        double avg = 0.0;
        long long samples = 0;
        bool valid = false;
    };

    PseudoDtStats pseudoStatsU;
    PseudoDtStats pseudoStatsV;

    // Velocity fields (staggered)
    Eigen::MatrixXd u;       // x-velocity at east faces
    Eigen::MatrixXd v;       // y-velocity at north faces
    Eigen::MatrixXd uStar;   // intermediate u
    Eigen::MatrixXd vStar;   // intermediate v
    Eigen::MatrixXd uOld;    // previous iteration u
    Eigen::MatrixXd vOld;    // previous iteration v

    // Pressure field (cell-centered)
    Eigen::MatrixXd p;       // pressure
    Eigen::MatrixXd pStar;   // pressure correction

    // Momentum equation coefficients for pressure-velocity coupling
    Eigen::MatrixXd dE;      // d coefficient for u
    Eigen::MatrixXd dN;      // d coefficient for v
    Eigen::MatrixXd b;       // work array

    // Geometry (cell types and porosity)
    Eigen::MatrixXi cellType;  // 0=fluid, 1=solid
    Eigen::MatrixXd alpha;     // Brinkman porosity term

    // Precomputed fluid masks (for speed optimization)
    std::vector<bool> isFluidU;   // true if u(i,j) is in fluid region
    std::vector<bool> isFluidV;   // true if v(i,j) is in fluid region
    std::vector<bool> isFluidP;   // true if p(i,j) is in fluid region
    void buildFluidMasks();       // Called after loading geometry

    // Constructor and initialization
    SIMPLE();
    void loadParameters(const std::string& paramsFile);
    void initializeMemory();

    // Main solver
    void runIterations();
    double calculateStep(int& pressureIterations);

    // Solver subcomponents (implemented in Utilities/*.cpp)
    bool solvePressureSystem(int& pressureIterations, double& localResidMass);
    void updateCflRamp(double currRes);

    // Geometry
    void loadTopology(const std::string& fileName);
    void buildAlphaField();

    // Boundary conditions
    void setVelocityBoundaryConditions(Eigen::MatrixXd& uIn, Eigen::MatrixXd& vIn);
    void setPressureBoundaryConditions(Eigen::MatrixXd& pIn);
    double checkBoundaries(int i, int j);
    void paintBoundaries();

    // Output
    void saveAll();
    void saveMatrix(Eigen::MatrixXd inputMatrix, std::string fileName);
    void initLogFiles(std::ofstream& residFile, std::ofstream& dpFile);
    void printIterationHeader() const;
    void writeIterationLogs(std::ofstream& residFile,
                            std::ofstream& dpFile,
                            int iter,
                            double corePressureDrop,
                            double fullPressureDrop,
                            double coreStaticDrop,
                            double fullStaticDrop);
    void printIterationRow(int iter,
                           double residMassVal,
                           double residUVal,
                           double residVVal,
                           double maxTransRes,
                           double corePressureDrop,
                           double fullPressureDrop,
                           double iterTimeMs,
                           int pressureIterations) const;
    void printStaticDp(int iter,
                       double coreStaticDrop,
                       double fullStaticDrop) const;

};

// Inline helpers (masks and alpha lookups)
inline bool fluidU(const SIMPLE& s, int i, int j) {
    return s.isFluidU[i * s.N + j];
}

inline bool fluidV(const SIMPLE& s, int i, int j) {
    return s.isFluidV[i * (s.N + 1) + j];
}

inline bool fluidP(const SIMPLE& s, int i, int j) {
    return s.isFluidP[i * (s.N + 1) + j];
}

inline double alphaAtU(const SIMPLE& s, int i, int j) {
    const auto& alpha = s.alpha;
    const int M = s.M;
    const int N = s.N;
    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 0.0;
    int ci = i - 1;
    int cj1 = j - 1;
    int cj2 = j;
    if (ci < 0 || ci >= alpha.rows()) return 0.0;
    if (cj1 < 0 || cj1 >= alpha.cols()) return 0.0;
    if (cj2 < 0 || cj2 >= alpha.cols()) return alpha(ci, cj1);
    return 0.5 * (alpha(ci, cj1) + alpha(ci, cj2));
}

inline double alphaAtV(const SIMPLE& s, int i, int j) {
    const auto& alpha = s.alpha;
    const int M = s.M;
    const int N = s.N;
    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 0.0;
    int ci1 = i - 1;
    int ci2 = i;
    int cj = j - 1;
    if (cj < 0 || cj >= alpha.cols()) return 0.0;
    if (ci1 < 0 || ci1 >= alpha.rows()) return 0.0;
    if (ci2 < 0 || ci2 >= alpha.rows()) return alpha(ci1, cj);
    return 0.5 * (alpha(ci1, cj) + alpha(ci2, cj));
}

// Convection (SOU deferred corrections)
double computeSOUCorrectionU(const SIMPLE& s, int i, int j,
                             double Fe, double Fw, double Fn, double Fs);
double computeSOUCorrectionV(const SIMPLE& s, int i, int j,
                             double Fe, double Fw, double Fn, double Fs);

// Time-control logging
void logPseudoStats(const SIMPLE& solver, const char* label, const SIMPLE::PseudoDtStats& stats);

// ============================================================================
// Post-processing: Physical location-based pressure sampling (postprocessing.cpp)
// ============================================================================
struct PlaneMetrics {
    double flowArea   = 0.0;  // Total open (fluid) area at the plane [m^2 per unit depth]
    double massFlux   = 0.0;  // Total mass flux through the plane [kg/s per unit depth]
    double avgStatic  = 0.0;  // Area-weighted average static pressure [Pa]
    double avgDynamic = 0.0;  // Area-weighted average dynamic pressure [Pa]
    double avgTotal   = 0.0;  // Mass-weighted average total pressure [Pa]
    bool   valid      = false; // True if at least one fluid cell was sampled
};

// Sample pressure and velocity at a given physical x-coordinate using linear interpolation
// Skips rows where either adjacent cell is solid
PlaneMetrics samplePlaneAtX(const SIMPLE& solver, double xPhysical);

// Debug helper to print plane sampling information
void printPlaneInfo(const char* name, double xPhysical, const PlaneMetrics& m);
