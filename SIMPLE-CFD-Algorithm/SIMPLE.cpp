#include "SIMPLE.h"
#include <cstdlib>
#include <iomanip>
#include <algorithm>
#include <chrono>
#include <deque>

// ------------------------------------------------------------------
// Constructor & Initialization
// ------------------------------------------------------------------
SIMPLE::SIMPLE() {
    loadParameters("ExportFiles/fluid_params.txt");
    initializeMemory();
    loadTopology("ExportFiles/geometry_fluid.txt");
    buildAlphaField();
    buildFluidMasks();  // Precompute fluid/solid masks for speed

    // Initialize with zero velocity field
    // The inlet BC will be applied during iterations
}

void SIMPLE::loadParameters(const std::string& fileName) {
    std::ifstream in(fileName);
    if (!in) {
        std::cerr << "Error: Could not open " << fileName << std::endl;
        std::cerr << "Run 'python GeometryGenerator.py' first!" << std::endl;
        exit(1);
    }

    if (!(in >> M >> N >> hy >> hx >> targetVel >> N_in_buffer >> N_out_buffer)) {
        std::cerr << "Error: Invalid parameter file format." << std::endl;
        exit(1);
    }

    double domainLength = N * hx;
    double velRef = std::max(targetVel, 1e-6);
    timeStep = timeStepMultiplier * domainLength / velRef;
    
    // Set pseudo reference length if not specified (mesh-independent!)
    // Based on domain length and inlet velocity, NOT cell size
    if (pseudoRefLength <= 0.0) {
        pseudoRefLength = domainLength / 100.0;  // 1% of domain length
    }

    // Initialize CFL ramping state
    if (enableCflRamp) {
        // Start from user-defined initial CFL
        pseudoCFL = pseudoCFLInitial;
    }
    
    // Calculate Reynolds number
    double L_char = M * hx;
    double Re = rho * targetVel * L_char / eta;

    // Set OpenMP threads
    omp_set_num_threads(numThreads);

    std::cout << "============================================" << std::endl;
    std::cout << "   SIMPLE CFD Solver - Laminar Flow        " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Algorithm: " << (useSIMPLEC ? "SIMPLEC" : "SIMPLE") << std::endl;
    std::cout << "Pressure solver: " << (useDirectPressureSolver ? "Direct (SimplicialLDLT)" : "Iterative (SOR)") << std::endl;
    std::cout << "OpenMP Threads: " << numThreads << std::endl;
    std::cout << "Grid: " << M << " x " << N << " (" << M * N << " cells)" << std::endl;
    std::cout << "Cell: " << hx * 1000 << " x " << hy * 1000 << " mm" << std::endl;
    std::cout << "Domain: " << N * hx * 1000 << " x " << M * hy * 1000 << " mm" << std::endl;
    std::cout << "Inlet: " << targetVel << " m/s" << std::endl;
    std::cout << "Re: " << std::fixed << std::setprecision(0) << Re << std::endl;
    std::cout << "Pseudo dt: " << std::scientific << timeStep << " s (manual)" << std::endl;
    if (enablePseudoTimeStepping) {
        std::cout << "Pseudo CFL: " << std::defaultfloat << pseudoCFL << std::endl;
        std::cout << "Pseudo ref length: " << std::scientific << pseudoRefLength << " m (mesh-independent)" << std::endl;
        std::cout << "Local pseudo time: " << (useLocalPseudoTime ? "ON" : "OFF") << std::endl;
        if (enableCflRamp) {
            std::cout << "CFL ramping: ON (CFL_init=" << pseudoCFLInitial
                      << ", CFL_max=" << pseudoCFLMax
                      << ", startRes=" << cflRampStartRes << ")\n";
        } else {
            std::cout << "CFL ramping: OFF\n";
        }
    } else {
        std::cout << "Pseudo time stepping: DISABLED" << std::endl;
    }
    std::cout << std::defaultfloat;
    std::cout << "============================================" << std::endl;
}

void SIMPLE::initializeMemory() {
    // Staggered grid sizes:
    // u: (M+1) rows, N columns (at vertical faces)
    // v: M rows, (N+1) columns (at horizontal faces)
    // p: (M+1) rows, (N+1) columns (at cell centers + ghost cells)

    u     = Eigen::MatrixXd::Zero(M + 1, N);
    v     = Eigen::MatrixXd::Zero(M, N + 1);
    p     = Eigen::MatrixXd::Zero(M + 1, N + 1);

    uStar = Eigen::MatrixXd::Zero(M + 1, N);
    vStar = Eigen::MatrixXd::Zero(M, N + 1);
    pStar = Eigen::MatrixXd::Zero(M + 1, N + 1);

    uOld  = Eigen::MatrixXd::Zero(M + 1, N);
    vOld  = Eigen::MatrixXd::Zero(M, N + 1);

    dE    = Eigen::MatrixXd::Zero(M + 1, N);
    dN    = Eigen::MatrixXd::Zero(M, N + 1);
    b     = Eigen::MatrixXd::Zero(M + 1, N + 1);

    cellType = Eigen::MatrixXi::Ones(M, N);  // Default: all fluid
    alpha    = Eigen::MatrixXd::Zero(M, N);

    if (reuseInitialFields) {
        auto loadMatrix = [&](const std::string& baseName, Eigen::MatrixXd& mat) {
            std::string fullPath = restartDirectory + "/" + baseName + ".txt";
            std::ifstream in(fullPath);
            if (!in) {
                std::cerr << "Failed to open restart file '" << fullPath
                          << "'. Initializing to zero." << std::endl;
                mat.setZero();
                return;
            }
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    if (!(in >> mat(i, j))) {
                        std::cerr << "Restart file '" << fullPath << "' is incomplete. "
                                  << "Falling back to zeros." << std::endl;
                        mat.setZero();
                        return;
                    }
                }
            }
        };
        loadMatrix("u", u);
        loadMatrix("v", v);
        loadMatrix("p", p);
        uStar = u;
        vStar = v;
        uOld = u;
        vOld = v;
        std::cout << "Initialized fields from restart files in '" << restartDirectory << "'." << std::endl;
    }
}

// ------------------------------------------------------------------
// Main Iteration Loop
// ------------------------------------------------------------------
void SIMPLE::runIterations() {
    int iter = 0;
    double error = 1.0;

    // Output file for residuals
    std::ofstream residFile("ExportFiles/residuals.txt");
    residFile << "Iter MassResid UResid VResid Core_dP_AfterInletBuffer(Pa) "
              << "Full_dP_FullSystem(Pa)" << std::endl;
    std::ofstream dpFile("ExportFiles/pressure_drop_history.txt");
    dpFile << "Iter Core_Total(Pa) Full_Total(Pa) Core_Static(Pa) Full_Static(Pa)" << std::endl;

    std::cout << "Starting simulation..." << std::endl;
    std::cout << std::setw(8) << "Iter"
              << std::setw(14) << "Mass"
              << std::setw(14) << "U-vel"
              << std::setw(14) << "V-vel"
              << std::setw(14) << "TransRes"
              << std::setw(16) << "Core dP (Pa)"
              << std::setw(16) << "Full dP (Pa)"
              << std::setw(10) << "Time (ms)"
              << std::setw(6) << "P-It" << std::endl;
    std::cout << std::string(122, '-') << std::endl;

    auto logPseudoStats = [&](const char* label, const PseudoDtStats& stats) {
        if (!enablePseudoTimeStepping || !useLocalPseudoTime || !stats.valid) return;
        std::cout << "        Pseudo-dt " << label << " [min/avg/max] = "
                  << std::scientific << std::setprecision(3)
                  << stats.min << " / " << stats.avg << " / " << stats.max
                  << "  (samples=" << stats.samples << ")" << std::defaultfloat << std::endl;
    };

    // ============================================================
    // Physical location-based pressure sampling planes
    // ============================================================
    // Domain physical dimensions
    const double domainXMax = N * hx;  // Total domain length in meters
    
    // Core planes: user-configurable physical x-coordinates
    const double xCoreIn  = xPlaneCoreInlet;
    const double xCoreOut = xPlaneCoreOutlet;
    
    // Full system planes: domain boundaries (auto-calculated)
    const double xFullIn  = 0.0;
    const double xFullOut = domainXMax;
    
    std::cout << "Pressure-drop planes (physical x-coordinates):" << std::endl;
    std::cout << "  Core inlet:   x = " << std::fixed << std::setprecision(4) 
              << xCoreIn * 1000.0 << " mm" << std::endl;
    std::cout << "  Core outlet:  x = " << xCoreOut * 1000.0 << " mm" << std::endl;
    std::cout << "  Full inlet:   x = " << xFullIn * 1000.0 << " mm (domain start)" << std::endl;
    std::cout << "  Full outlet:  x = " << xFullOut * 1000.0 << " mm (domain end)" << std::endl;
    std::cout << "  -> Core Δp = user-defined core region" << std::endl;
    std::cout << "  -> Full Δp = entire domain (inlet to outlet)" << std::endl;
    std::cout << std::defaultfloat;
    
    // Pressure drop convergence tracking (circular buffer for window)
    std::vector<double> dpHistory(dpConvergenceWindow, 0.0);
    int dpHistoryIdx = 0;
    bool dpHistoryFilled = false;
    bool dpConverged = false;
    std::deque<std::pair<int, double>> dpSlopeBuffer;
    auto computeDpSlope = [&]() -> double {
        if (dpSlopeBuffer.size() < 2) return 0.0;
        double x0 = static_cast<double>(dpSlopeBuffer.front().first);
        double sumX = 0.0;
        double sumY = 0.0;
        double sumXX = 0.0;
        double sumXY = 0.0;
        const double n = static_cast<double>(dpSlopeBuffer.size());
        for (const auto& sample : dpSlopeBuffer) {
            double x = static_cast<double>(sample.first) - x0;
            double y = sample.second;
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumXY += x * y;
        }
        double denom = n * sumXX - sumX * sumX;
        if (std::abs(denom) < 1e-12) return 0.0;
        return (n * sumXY - sumX * sumY) / denom;
    };
    
    // Total simulation time tracking
    auto simStart = std::chrono::high_resolution_clock::now();

    while (iter < maxIterations && error > epsilon && !dpConverged) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // Optional inlet velocity ramp (useful for aggressive start-up damping)
        if (enableInletRamp && rampSteps > 0 && iter < rampSteps) {
            inletVelocity = targetVel * (0.1 + 0.9 * double(iter) / rampSteps);
        } else {
            inletVelocity = targetVel;
        }

        // Perform one SIMPLE iteration
        int pressureIterations = 0;
        error = calculateStep(pressureIterations);

        // ------------------------------------------------------------------
        // CFL ramping logic (residual-based) - updates pseudoCFL in-place
        // ------------------------------------------------------------------
        if (enablePseudoTimeStepping && enableCflRamp) {
            // We use mass residual as the ramp driver
            double currRes = std::max(residMass, 1e-12);

            // Before ramp start: hold CFL at initial value
            if (currRes > cflRampStartRes) {
                pseudoCFL = pseudoCFLInitial;
            } else {
                // Ramp phase: increase CFL as residual drops
                double ratio    = cflRampStartRes / currRes;
                double rawFactor = std::pow(ratio, cflRampExponent);
                double targetCFL = std::min(pseudoCFLInitial * rawFactor, pseudoCFLMax);

                // Smooth update to avoid sudden jumps
                pseudoCFL = (1.0 - cflRampSmooth) * pseudoCFL
                          +  cflRampSmooth         * targetCFL;

                // Safety guard
                if (!std::isfinite(pseudoCFL) || pseudoCFL <= 0.0) {
                    pseudoCFL = pseudoCFLInitial;
                }
            }
        }
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        double iterTimeMs = std::chrono::duration<double, std::milli>(iterEnd - iterStart).count();

        // ============================================================
        // Calculate pressure drop (static and total)
        // Uses physical location-based sampling with linear interpolation
        // Static Pressure = p only
        // Total Pressure = p + 0.5 * rho * V^2
        // ============================================================
        PlaneMetrics inletCoreMetrics  = samplePlaneAtX(*this, xCoreIn);
        PlaneMetrics outletCoreMetrics = samplePlaneAtX(*this, xCoreOut);
        PlaneMetrics inletFullMetrics  = samplePlaneAtX(*this, xFullIn);
        PlaneMetrics outletFullMetrics = samplePlaneAtX(*this, xFullOut);
        
        // Calculate pressure drops (using total pressure: static + dynamic)
        double corePressureDrop = inletCoreMetrics.avgTotal - outletCoreMetrics.avgTotal;
        double fullPressureDrop = inletFullMetrics.avgTotal - outletFullMetrics.avgTotal;
        
        // Also calculate static-only for comparison/debugging
        double coreStaticDrop = inletCoreMetrics.avgStatic - outletCoreMetrics.avgStatic;
        double fullStaticDrop = inletFullMetrics.avgStatic - outletFullMetrics.avgStatic;
        if (usePressureDropSlopeGate && dpSlopeWindowIters > 0) {
            dpSlopeBuffer.emplace_back(iter, corePressureDrop);
            if (dpSlopeBuffer.size() > static_cast<size_t>(dpSlopeWindowIters)) {
                dpSlopeBuffer.pop_front();
            }
        }

        // Pressure drop convergence check (core/fluid region only)
        if (usePressureDropConvergence && iter >= rampSteps) {
            // Store current pressure drop in circular buffer
            dpHistory[dpHistoryIdx] = corePressureDrop;
            dpHistoryIdx = (dpHistoryIdx + 1) % dpConvergenceWindow;
            if (dpHistoryIdx == 0) dpHistoryFilled = true;
            
            // Check convergence only after buffer is filled
            if (dpHistoryFilled) {
                double sum = 0.0;
                for (int k = 0; k < dpConvergenceWindow; ++k) {
                    sum += dpHistory[k];
                }
                double mean = sum / double(dpConvergenceWindow);
                
                double varSum = 0.0;
                for (int k = 0; k < dpConvergenceWindow; ++k) {
                    double diff = dpHistory[k] - mean;
                    varSum += diff * diff;
                }
                double stdDev = std::sqrt(varSum / double(dpConvergenceWindow));
                
                double denom = std::max(std::abs(mean), 1.0);
                double stdPercent = (stdDev / denom) * 100.0;
                int lastIdx = (dpHistoryIdx - 1 + dpConvergenceWindow) % dpConvergenceWindow;
                double latest = dpHistory[lastIdx];
                double latestDeltaPercent = (std::abs(latest - mean) / denom) * 100.0;

                bool slopeGatePassed = true;
                if (usePressureDropSlopeGate &&
                    dpSlopeWindowIters > 0 &&
                    dpSlopeBuffer.size() >= static_cast<size_t>(dpSlopeWindowIters)) {
                    double slope = computeDpSlope();
                    double slopeTol = dpSlopeMaxDelta / static_cast<double>(dpSlopeWindowIters);
                    slopeGatePassed = std::abs(slope) <= slopeTol;
                }
                
                if (stdPercent < dpConvergencePercent &&
                    latestDeltaPercent < dpConvergencePercent &&
                    slopeGatePassed) {
                    dpConverged = true;
                }
            }
        }

        // Write to files
        residFile << iter << " "
                  << residMass << " "
                  << residU << " "
                  << residV << " "
                  << corePressureDrop << " "
                  << fullPressureDrop << std::endl;
        dpFile << iter << " "
               << corePressureDrop << " "
               << fullPressureDrop << " "
               << coreStaticDrop << " "
               << fullStaticDrop << std::endl;

        // Print to console every 10 iterations
        // TransRes = max(transientResidU, transientResidV) - should → 0 at true steady state
        double maxTransRes = std::max(transientResidU, transientResidV);
        if (iter % 10 == 0 || iter < 10) {
            std::cout << std::setw(8) << iter
                      << std::setw(14) << std::scientific << std::setprecision(3) << residMass
                      << std::setw(14) << residU
                      << std::setw(14) << residV
                      << std::setw(14) << maxTransRes
                      << std::setw(16) << std::fixed << std::setprecision(1) << corePressureDrop
                      << std::setw(16) << fullPressureDrop
                      << std::setw(10) << std::fixed << std::setprecision(1) << iterTimeMs
                      << std::setw(6) << pressureIterations
                      << std::endl;
            // Print static pressure drop on separate line for comparison (every 100 iterations)
            if (iter % 100 == 0 || iter < 10) {
                std::cout << "         Static Δp (Core/Full): " 
                          << std::setw(12) << std::fixed << std::setprecision(1) << coreStaticDrop << " / "
                          << std::setw(12) << fullStaticDrop << " Pa"
                          << std::endl;
            }
            logPseudoStats("U", pseudoStatsU);
            logPseudoStats("V", pseudoStatsV);
        }

        // Checkpoints
        if (saveStateInterval > 0 && iter > 0 && iter % saveStateInterval == 0) {
            saveAll();
            std::cout << "  [Saved checkpoint at iteration " << iter << "]" << std::endl;
        }

        iter++;
    }

    residFile.close();
    dpFile.close();
    saveAll();

    // ============================================================
    // Final summary with TOTAL pressure drop
    // Uses physical location-based sampling with linear interpolation
    // ============================================================
    PlaneMetrics coreInletFinal  = samplePlaneAtX(*this, xCoreIn);
    PlaneMetrics coreOutletFinal = samplePlaneAtX(*this, xCoreOut);
    PlaneMetrics fullInletFinal  = samplePlaneAtX(*this, xFullIn);
    PlaneMetrics fullOutletFinal = samplePlaneAtX(*this, xFullOut);
    
    double coreStaticDrop = coreInletFinal.avgStatic - coreOutletFinal.avgStatic;
    double coreTotalDrop = coreInletFinal.avgTotal - coreOutletFinal.avgTotal;
    double fullStaticDrop = fullInletFinal.avgStatic - fullOutletFinal.avgStatic;
    double fullTotalDrop = fullInletFinal.avgTotal - fullOutletFinal.avgTotal;

    // Calculate total simulation time
    auto simEnd = std::chrono::high_resolution_clock::now();
    double totalTimeSec = std::chrono::duration<double>(simEnd - simStart).count();
    int totalMin = static_cast<int>(totalTimeSec) / 60;
    double totalSec = totalTimeSec - totalMin * 60;

    std::cout << std::string(106, '=') << std::endl;
    if (dpConverged) {
        std::cout << "CONVERGED (pressure drop stable within " << dpConvergencePercent 
                  << "% over " << dpConvergenceWindow << " iterations) after " << iter << " iterations" << std::endl;
    } else if (error <= epsilon) {
        std::cout << "CONVERGED (residuals) after " << iter << " iterations" << std::endl;
    } else {
        std::cout << "Stopped at " << iter << " iterations (max reached)" << std::endl;
    }
    std::cout << "Total simulation time: " << totalMin << " min " 
              << std::fixed << std::setprecision(1) << totalSec << " sec (" 
              << std::setprecision(1) << totalTimeSec << " s)" << std::endl;
    std::cout << "Final residuals: Mass=" << std::scientific << residMass
              << " U=" << residU << " V=" << residV << std::endl;
    std::cout << std::fixed << std::setprecision(1);
    logPseudoStats("U (final)", pseudoStatsU);
    logPseudoStats("V (final)", pseudoStatsV);
    std::cout << "Core region (x=" << std::setprecision(2) << xCoreIn*1000.0 
              << " mm -> x=" << xCoreOut*1000.0 << " mm)" << std::endl;
    std::cout << "  Open inlet area:  " << coreInletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Open outlet area: " << coreOutletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Static Δp:        " << coreStaticDrop << " Pa (" 
              << coreStaticDrop/1000.0 << " kPa)" << std::endl;
    std::cout << "  Dynamic p (in/out): " << coreInletFinal.avgDynamic << " Pa | "
              << coreOutletFinal.avgDynamic << " Pa" << std::endl;
    std::cout << "  Total (mass-weighted, in/out): " << coreInletFinal.avgTotal << " Pa | "
              << coreOutletFinal.avgTotal << " Pa" << std::endl;
    std::cout << "  TOTAL Δp (core):  " << coreTotalDrop << " Pa (" 
              << coreTotalDrop/1000.0 << " kPa)" << std::endl;
    std::cout << std::endl;
    std::cout << "Full system (x=" << std::setprecision(2) << xFullIn*1000.0 
              << " mm -> x=" << xFullOut*1000.0 << " mm)" << std::endl;
    std::cout << "  Open inlet area:  " << fullInletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Open outlet area: " << fullOutletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Static Δp:        " << fullStaticDrop << " Pa (" 
              << fullStaticDrop/1000.0 << " kPa)" << std::endl;
    std::cout << "  Dynamic p (in/out): " << fullInletFinal.avgDynamic << " Pa | "
              << fullOutletFinal.avgDynamic << " Pa" << std::endl;
    std::cout << "  Total (mass-weighted, in/out): " << fullInletFinal.avgTotal << " Pa | "
              << fullOutletFinal.avgTotal << " Pa" << std::endl;
    std::cout << "  TOTAL Δp (full):  " << fullTotalDrop << " Pa (" 
              << fullTotalDrop/1000.0 << " kPa)" << std::endl;
    std::cout << "Results saved to ExportFiles/" << std::endl;
    std::cout << std::string(106, '=') << std::endl;
}

// ------------------------------------------------------------------
// Geometry Loading
// ------------------------------------------------------------------
void SIMPLE::loadTopology(const std::string& fileName) {
    std::ifstream in(fileName);
    if (!in) {
        std::cerr << "Warning: Could not open topology file '"
                  << fileName << "'. Assuming all fluid." << std::endl;
        // Default all to fluid
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                cellType(i, j) = 0;  // 0 = fluid
            }
        }
        return;
    }

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            int val;
            if (!(in >> val)) return;
            cellType(i, j) = val;  // Keep original values: 0=fluid, 1=solid
        }
    }
    
    // Count fluid/solid cells
    int fluidCells = 0, solidCells = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (cellType(i, j) == 0) fluidCells++;
            else solidCells++;
        }
    }
    std::cout << "Geometry: " << fluidCells << " fluid, " << solidCells << " solid cells" << std::endl;
}

void SIMPLE::buildAlphaField() {
    // Brinkman penalization: high alpha in solid regions creates drag
    const double alphaFluid = 0.0;
    const double alphaSolid = 0.0;  // Solid cells handled by masking (no Brinkman penalization)

    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            // 0 = fluid (low alpha), non-zero = solid (high alpha)
            alpha(i, j) = (cellType(i, j) == 0) ? alphaFluid : alphaSolid;
        }
    }
}

// ------------------------------------------------------------------
// Build Fluid Masks (Precomputed for Speed)
// ------------------------------------------------------------------
void SIMPLE::buildFluidMasks() {
    // Allocate masks
    const int uRows = M + 1;
    const int uCols = N;
    const int vRows = M;
    const int vCols = N + 1;
    const int pRows = M + 1;
    const int pCols = N + 1;

    isFluidU.resize(uRows * uCols, false);
    isFluidV.resize(vRows * vCols, false);
    isFluidP.resize(pRows * pCols, false);

    // Helper to check if a pressure cell (i,j) is solid
    // Pressure grid is (M+1) x (N+1), cellType is M x N
    // Pressure cell (i,j) corresponds to cellType(i-1, j-1)
    auto isSolid = [&](int i, int j) -> bool {
        int ci = i - 1;
        int cj = j - 1;
        if (ci < 0 || ci >= M || cj < 0 || cj >= N) return false;
        return cellType(ci, cj) != 0;
    };

    // Build U mask: u(i,j) is fluid if both adjacent pressure cells are fluid
    // u is at east face of cell (i,j), between p(i,j) and p(i,j+1)
    for (int i = 0; i < uRows; ++i) {
        for (int j = 0; j < uCols; ++j) {
            bool solid = isSolid(i, j) || isSolid(i, j + 1);
            isFluidU[i * uCols + j] = !solid;
        }
    }

    // Build V mask: v(i,j) is fluid if both adjacent pressure cells are fluid
    // v is at north face of cell (i,j), between p(i,j) and p(i+1,j)
    for (int i = 0; i < vRows; ++i) {
        for (int j = 0; j < vCols; ++j) {
            bool solid = isSolid(i, j) || isSolid(i + 1, j);
            isFluidV[i * vCols + j] = !solid;
        }
    }

    // Build P mask: p(i,j) is fluid if corresponding cellType is fluid
    for (int i = 0; i < pRows; ++i) {
        for (int j = 0; j < pCols; ++j) {
            isFluidP[i * pCols + j] = !isSolid(i, j);
        }
    }

    // Count for diagnostics
    int fluidU = 0, fluidV = 0, fluidP = 0;
    for (bool b : isFluidU) if (b) fluidU++;
    for (bool b : isFluidV) if (b) fluidV++;
    for (bool b : isFluidP) if (b) fluidP++;
    std::cout << "Fluid masks: U=" << fluidU << "/" << isFluidU.size()
              << ", V=" << fluidV << "/" << isFluidV.size()
              << ", P=" << fluidP << "/" << isFluidP.size() << std::endl;
}

// ------------------------------------------------------------------
// Main Entry Point
// ------------------------------------------------------------------
int main() {
    SIMPLE solver;
    solver.runIterations();
    return 0;
}
