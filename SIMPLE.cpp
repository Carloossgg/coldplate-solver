// File: SIMPLE.cpp
// Author: Peter Tcherkezian
// Description: Implements the SIMPLE steady laminar solver workflow:
//   - loads parameters/geometry, allocates staggered fields and masks
//   - runs the outer iteration: U/V momentum solve, pressure correction, velocity correction
//   - applies pseudo-time/CFL ramps, monitors residuals/pressure drops, logs/prints iteration data
//   - handles checkpoints and final summaries/field saves
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
    
    // Density-based topology optimization (always enabled)
    loadDensityField();  // Converts cellType to gamma
    buildAlphaFromDensity();  // Build Brinkman alpha from gamma field

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
    // Optional out-of-plane height for 2.5D model
    float htFromFile = 0.0f;
    if (in >> htFromFile) {
        Ht_channel = htFromFile;
    }
    // Ignore any extra columns (e.g., rho/nu, old buffer/convexity params) to enforce SIMPLE.h fluid properties
    float ignoredRho = 0.0f, ignoredNu = 0.0f;
    if (in >> ignoredRho >> ignoredNu) {
        std::cout << "Ignoring fluid properties in " << fileName
                  << "; using SIMPLE.h defaults (rho=" << rho
                  << ", eta=" << eta << ")." << std::endl;
    }
    if (enableTwoPointFiveD) {
        if (Ht_channel <= 0.0) {
            std::cerr << "Warning: Ht_channel not provided or <=0; falling back to hy (" << hy << " m)." << std::endl;
            Ht_channel = hy;
        }
    }
    
    // Compute Brinkman alpha_max from MATERIAL permeability:
    //   alpha_max = mu / K_min
    // This is mesh/domain independent and preferred for SIMP/Brinkman TO.
    float K_min = std::max(brinkmanKMin, 1e-20f);
    brinkmanAlphaMax = eta / K_min;

    float domainLength = N * hx;
    float velRef = std::max(targetVel, 1e-6f);
    timeStep = timeStepMultiplier * domainLength / velRef;
    
    // Set pseudo reference length if not specified (mesh-independent!)
    // Based on domain length and inlet velocity, NOT cell size
    if (pseudoRefLength <= 0.0f) {
        pseudoRefLength = domainLength / 100.0f;  // 1% of domain length
    }

    // Initialize CFL ramping state
    if (enableCflRamp) {
        // Start from user-defined initial CFL
        pseudoCFL = pseudoCFLInitial;
    }
    
    // Calculate Reynolds number
    float L_char = M * hx;
    float Re = rho * targetVel * L_char / eta;

    // Set OpenMP threads
    omp_set_num_threads(numThreads);

    std::cout << "============================================" << std::endl;
    std::cout << "   SIMPLE CFD Solver - Laminar Flow        " << std::endl;
    std::cout << "============================================" << std::endl;
    std::cout << "Algorithm: SIMPLE" << std::endl;
    if (enableTwoPointFiveD) {
        std::cout << "2.5D mode: ON (Ht = " << Ht_channel*1000.0 << " mm)" << std::endl;
    } else {
        std::cout << "2.5D mode: OFF" << std::endl;
    }
    std::cout << "Density-based TO: ON (K_min=" << std::scientific << K_min
              << " m^2, alpha_max=" << brinkmanAlphaMax << ")" << std::defaultfloat << std::endl;
    const char* solverNames[] = {"SOR (Red-Black)", "Parallel CG (Jacobi)", "AMGCL (AMG+CG)", "Direct LDLT (Serial)", "AMGCL-CUDA (GPU Float)"};
    int solverIdx = std::min(std::max(pressureSolverType, 0), 4);
#ifndef USE_AMGCL
    if (solverIdx == 2) {
        std::cout << "\n";
        std::cout << "========================================================================\n";
        std::cout << "                    *** AMGCL NOT COMPILED ***\n";
        std::cout << "========================================================================\n";
        std::cout << "  AMGCL solver was selected but not compiled into the executable.\n";
        std::cout << "  Falling back to Parallel CG (slower performance).\n";
        std::cout << "\n";
        std::cout << "  To enable AMGCL, recompile with:\n";
        std::cout << "    g++ -std=c++17 -O3 -fopenmp -DUSE_AMGCL -DAMGCL_NO_BOOST -I. \\\n";
        std::cout << "        -I\"ThermalSolver/amgcl\" -I\"Utilities/eigen-3.4.0\" \\\n";
        std::cout << "        SIMPLE.cpp Utilities/*.cpp -o simple.exe\n";
        std::cout << "========================================================================\n";
        std::cout << "\n";
        solverIdx = 1;
    }
#endif
#ifndef USE_AMGCL_CUDA
    if (solverIdx == 4) {
        std::cout << "\n";
        std::cout << "========================================================================\n";
        std::cout << "                *** AMGCL-CUDA (GPU) NOT COMPILED ***\n";
        std::cout << "========================================================================\n";
        std::cout << "  AMGCL-CUDA GPU solver was selected but not compiled.\n";
        std::cout << "  Falling back to AMGCL (CPU) or Parallel CG.\n";
        std::cout << "\n";
        std::cout << "  To enable AMGCL-CUDA, run: build_with_amgcl_cuda.bat\n";
        std::cout << "========================================================================\n";
        std::cout << "\n";
#ifdef USE_AMGCL
        solverIdx = 2;  // Fall back to AMGCL (CPU)
#else
        solverIdx = 1;  // Fall back to Parallel CG
#endif
    }
#endif
    pressureSolverType = solverIdx;
    std::cout << "Pressure solver: " << solverNames[pressureSolverType] << std::endl;
    
    // Momentum solver info (Jacobi/SOR)
    const char* momentumSolverNames[] = {"Explicit (Point Gauss-Seidel)", "Implicit Jacobi/SOR"};
    int momIdx = std::min(std::max(momentumSolverType, 0), 1);
    std::cout << "Momentum solver: " << momentumSolverNames[momIdx];
    if (momentumSolverType == 1) {
        std::cout << " (omega=" << momentumSorOmega << ", tol=" << momentumTol << ", maxIter=" << maxMomentumIter << ")";
    }
    std::cout << std::endl;
    
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

    u     = Eigen::MatrixXf::Zero(M + 1, N);
    v     = Eigen::MatrixXf::Zero(M, N + 1);
    p     = Eigen::MatrixXf::Zero(M + 1, N + 1);

    uStar = Eigen::MatrixXf::Zero(M + 1, N);
    vStar = Eigen::MatrixXf::Zero(M, N + 1);
    pStar = Eigen::MatrixXf::Zero(M + 1, N + 1);

    uOld  = Eigen::MatrixXf::Zero(M + 1, N);
    vOld  = Eigen::MatrixXf::Zero(M, N + 1);

    dE    = Eigen::MatrixXf::Zero(M + 1, N);
    dN    = Eigen::MatrixXf::Zero(M, N + 1);
    b     = Eigen::MatrixXf::Zero(M + 1, N + 1);

    cellType = Eigen::MatrixXf::Zero(M, N);  // Default: all fluid (0=fluid)
    alpha    = Eigen::MatrixXf::Zero(M, N);
    gamma    = Eigen::MatrixXf::Ones(M, N);  // Default: all fluid (gamma=1)

    if (reuseInitialFields) {
        auto loadMatrix = [&](const std::string& baseName, Eigen::MatrixXf& mat) {
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
    float error = 1.0f;

    std::ofstream residFile, dpFile;
    initLogFiles(residFile, dpFile);

    // ============================================================
    // Physical location-based pressure sampling planes
    // ============================================================
    // Domain physical dimensions
    const float domainXMax = N * hx;  // Total domain length in meters
    
    // Core planes: user-configurable physical x-coordinates
    const float xCoreIn  = xPlaneCoreInlet;
    const float xCoreOut = xPlaneCoreOutlet;
    
    // Full system planes: domain boundaries (auto-calculated)
    const float xFullIn  = 0.0f;
    const float xFullOut = domainXMax;
    
    std::cout << "Pressure-drop planes (physical x-coordinates):" << std::endl;
    std::cout << "  Core inlet:   x = " << std::fixed << std::setprecision(4) 
              << xCoreIn * 1000.0 << " mm" << std::endl;
    std::cout << "  Core outlet:  x = " << xCoreOut * 1000.0 << " mm" << std::endl;
    std::cout << "  Full inlet:   x = " << xFullIn * 1000.0 << " mm (domain start)" << std::endl;
    std::cout << "  Full outlet:  x = " << xFullOut * 1000.0 << " mm (domain end)" << std::endl;
    std::cout << "  -> Core dP = user-defined core region" << std::endl;
    std::cout << "  -> Full dP = entire domain (inlet to outlet)" << std::endl;
    std::cout << std::defaultfloat;
    
    // Pressure drop convergence tracking (circular buffer for window)
    std::vector<float> dpHistory(dpConvergenceWindow, 0.0f);
    int dpHistoryIdx = 0;
    bool dpHistoryFilled = false;
    bool dpConverged = false;
    std::deque<std::pair<int, float>> dpSlopeBuffer;
    auto computeDpSlope = [&]() -> float {
        if (dpSlopeBuffer.size() < 2) return 0.0f;
        float x0 = static_cast<float>(dpSlopeBuffer.front().first);
        float sumX = 0.0f;
        float sumY = 0.0f;
        float sumXX = 0.0f;
        float sumXY = 0.0f;
        const float n = static_cast<float>(dpSlopeBuffer.size());
        for (const auto& sample : dpSlopeBuffer) {
            float x = static_cast<float>(sample.first) - x0;
            float y = sample.second;
            sumX += x;
            sumY += y;
            sumXX += x * x;
            sumXY += x * y;
        }
        float denom = n * sumXX - sumX * sumX;
        if (std::abs(denom) < 1e-12f) return 0.0f;
        return (n * sumXY - sumX * sumY) / denom;
    };
    
    // Total simulation time tracking
    auto simStart = std::chrono::high_resolution_clock::now();

    while (iter < maxIterations && error > epsilon && !dpConverged) {
        auto iterStart = std::chrono::high_resolution_clock::now();

        // Optional inlet velocity ramp (useful for aggressive start-up damping)
        if (enableInletRamp && rampSteps > 0 && iter < rampSteps) {
            inletVelocity = (static_cast<float>(iter + 1) / rampSteps) * targetVel;
        } else {
            inletVelocity = targetVel;
        }

        if (iter % 100 == 0) {
            GlobalTimer::instance().printReport();
        }

        // Perform one SIMPLE iteration
        int pressureIterations = 0;
        {
            ScopedTimer t("Outer: calculateStep");
            error = calculateStep(pressureIterations);
        }

        // CFL ramping: SER takes precedence if enabled, otherwise use other methods
        {
            ScopedTimer t("Outer: CFL Ramping & Line Search");
            float monitoringResid = serUseMaxResid ? error : residMass;
            updateCflSER(monitoringResid, iter);      // Switched Evolution Relaxation (if enabled)
            applyLineSearch(monitoringResid);         // Line search for robustness (works with SER)
            updateCflRamp(monitoringResid);           // Residual-based ramp (disabled if SER on)
        }
        
        auto iterEnd = std::chrono::high_resolution_clock::now();
        float iterTimeMs = std::chrono::duration<float, std::milli>(iterEnd - iterStart).count();

        // ============================================================
        // Calculate pressure drop (static and total)
        // Uses physical location-based sampling with linear interpolation
        // Static Pressure = p only
        // Total Pressure = p + 0.5 * rho * V^2
        // ============================================================
        PlaneMetrics inletCoreMetrics, outletCoreMetrics, inletFullMetrics, outletFullMetrics;
        {
            ScopedTimer t("Outer: Monitoring & Sampling");
            inletCoreMetrics  = samplePlaneAtX(*this, xCoreIn);
            outletCoreMetrics = samplePlaneAtX(*this, xCoreOut);
            inletFullMetrics  = samplePlaneAtX(*this, xFullIn);
            outletFullMetrics = samplePlaneAtX(*this, xFullOut);
        }
        
        // Calculate pressure drops (using total pressure: static + dynamic)
        float corePressureDrop = inletCoreMetrics.avgTotal - outletCoreMetrics.avgTotal;
        float fullPressureDrop = inletFullMetrics.avgTotal - outletFullMetrics.avgTotal;
        

        
        // Also calculate static-only for comparison/debugging
        float coreStaticDrop = inletCoreMetrics.avgStatic - outletCoreMetrics.avgStatic;
        float fullStaticDrop = inletFullMetrics.avgStatic - outletFullMetrics.avgStatic;
        {
            ScopedTimer t("Outer: Pressure Drop Data");
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
                    
                    float denom = std::max(std::abs(mean), 1.0);
                    float stdPercent = (stdDev / denom) * 100.0f;
                    int lastIdx = (dpHistoryIdx - 1 + dpConvergenceWindow) % dpConvergenceWindow;
                    float latest = dpHistory[lastIdx];
                    float latestDeltaPercent = (std::abs(latest - mean) / denom) * 100.0f;

                    bool slopeGatePassed = true;
                    if (usePressureDropSlopeGate &&
                        dpSlopeWindowIters > 0 &&
                        dpSlopeBuffer.size() >= static_cast<size_t>(dpSlopeWindowIters)) {
                        float slope = computeDpSlope();
                        float slopeTol = dpSlopeMaxDelta / static_cast<float>(dpSlopeWindowIters);
                        slopeGatePassed = std::abs(slope) <= slopeTol;
                    }
                    
                    if (stdPercent < dpConvergencePercent &&
                        latestDeltaPercent < dpConvergencePercent &&
                        slopeGatePassed) {
                        dpConverged = true;
                    }
                }
            }
        }

        // Write to files
        {
            ScopedTimer t("Outer: Logging & Output");
            writeIterationLogs(residFile, dpFile, iter,
                               corePressureDrop, fullPressureDrop,
                               coreStaticDrop, fullStaticDrop,
                               pseudoCFL);

            // Print to console every iteration
            float maxTransRes = std::max(transientResidU, transientResidV);
            printIterationRow(iter, residMass, residU, residV, maxTransRes,
                              corePressureDrop, fullPressureDrop,
                              iterTimeMs, pressureIterations,
                               pseudoCFL);
            if (iter % 100 == 0 || iter < 10) {
                printStaticDp(iter, coreStaticDrop, fullStaticDrop);
            }
            logPseudoStats(*this, "U", pseudoStatsU);
            logPseudoStats(*this, "V", pseudoStatsV);

            // Checkpoints
            if (saveStateInterval > 0 && iter > 0 && iter % saveStateInterval == 0) {
                saveAll();
                std::cout << "  [Saved checkpoint at iteration " << iter << "]" << std::endl;
            }
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
    
    float coreStaticDrop = coreInletFinal.avgStatic - coreOutletFinal.avgStatic;
    float coreTotalDrop = coreInletFinal.avgTotal - coreOutletFinal.avgTotal;
    float fullStaticDrop = fullInletFinal.avgStatic - fullOutletFinal.avgStatic;
    float fullTotalDrop = fullInletFinal.avgTotal - fullOutletFinal.avgTotal;

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
    logPseudoStats(*this, "U (final)", pseudoStatsU);
    logPseudoStats(*this, "V (final)", pseudoStatsV);
    std::cout << "Core region (x=" << std::setprecision(2) << xCoreIn*1000.0 
              << " mm -> x=" << xCoreOut*1000.0 << " mm)" << std::endl;
    std::cout << "  Open inlet area:  " << coreInletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Open outlet area: " << coreOutletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Static dP:        " << coreStaticDrop << " Pa (" 
              << coreStaticDrop/1000.0 << " kPa)" << std::endl;
    std::cout << "  Dynamic p (in/out): " << coreInletFinal.avgDynamic << " Pa | "
              << coreOutletFinal.avgDynamic << " Pa" << std::endl;
    std::cout << "  Total (mass-flow-avg, in/out): " << coreInletFinal.avgTotal << " Pa | "
              << coreOutletFinal.avgTotal << " Pa" << std::endl;
    std::cout << "  TOTAL dP (core):  " << coreTotalDrop << " Pa (" 
              << coreTotalDrop/1000.0 << " kPa)" << std::endl;
    std::cout << std::endl;
    std::cout << "Full system (x=" << std::setprecision(2) << xFullIn*1000.0 
              << " mm -> x=" << xFullOut*1000.0 << " mm)" << std::endl;
    std::cout << "  Open inlet area:  " << fullInletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Open outlet area: " << fullOutletFinal.flowArea << " m^2 (per unit depth)" << std::endl;
    std::cout << "  Static dP:        " << fullStaticDrop << " Pa (" 
              << fullStaticDrop/1000.0 << " kPa)" << std::endl;
    std::cout << "  Dynamic p (in/out): " << fullInletFinal.avgDynamic << " Pa | "
              << fullOutletFinal.avgDynamic << " Pa" << std::endl;
    std::cout << "  Total (mass-flow-avg, in/out): " << fullInletFinal.avgTotal << " Pa | "
              << fullOutletFinal.avgTotal << " Pa" << std::endl;
    std::cout << "  TOTAL dP (full):  " << fullTotalDrop << " Pa (" 
              << fullTotalDrop/1000.0 << " kPa)" << std::endl;
    std::cout << "Results saved to ExportFiles/" << std::endl;
    std::cout << std::string(106, '=') << std::endl;

    // Final profiling report
    GlobalTimer::instance().printReport();
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
            float val;
            if (!(in >> val)) return;
            cellType(i, j) = val;  // 0=fluid, 1=solid (supports both int and float input)
        }
    }
    
    // Count fluid/solid/buffer cells
    int fluidCells = 0, solidCells = 0, bufferCells = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float v = cellType(i, j);
            if (v < 0.01f) fluidCells++;       // 0 = fluid
            else if (v > 0.99f) solidCells++;  // 1 = solid
            else bufferCells++;               // intermediate = buffer
        }
    }
    std::cout << "Geometry: " << fluidCells << " fluid, " << solidCells << " solid, " 
              << bufferCells << " buffer cells" << std::endl;
}

// ------------------------------------------------------------------
// Load Density Field from geometry (density method always enabled)
// ------------------------------------------------------------------
void SIMPLE::loadDensityField() {
    // geometry_fluid.txt contains either:
    // - Binary: 0 (fluid) or 1 (solid)
    // - Continuous: values between 0.0 (solid) and 1.0 (fluid)
    // We convert to gamma convention: gamma=1 (fluid), gamma=0 (solid)
    
    // cellType is already loaded from geometry_fluid.txt by loadTopology()
    // Just convert cellType values to gamma
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float val = cellType(i, j);
            // If binary (0 or 1): invert to gamma convention
            // If continuous (0.0-1.0): invert to gamma convention
            gamma(i, j) = 1.0f - val;  // cellType 0 (fluid) -> gamma 1.0, cellType 1 (solid) -> gamma 0.0
        }
    }
    
    // Count density statistics
    int pureFluid = 0, pureSolid = 0, buffer = 0;
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            if (gamma(i, j) > 0.99f) pureFluid++;
            else if (gamma(i, j) < 0.01f) pureSolid++;
            else buffer++;
        }
    }
    std::cout << "Geometry: " << pureFluid << " fluid, " << pureSolid 
              << " solid, " << buffer << " buffer cells" << std::endl;
}

// ------------------------------------------------------------------
// Build Alpha Field from Density (Brinkman penalization - Haertel RAMP)
// ------------------------------------------------------------------
void SIMPLE::buildAlphaFromDensity() {
    // RAMP interpolation per Haertel et al. (Eq. 17):
    //   I_α(γ) = (1 - γ) / (1 + q * γ)
    //   α(γ) = α_max * (1 - γ) / (1 + q * γ)
    //
    // Where q is the convexity parameter (Haertel uses 1-8, final value 1):
    //   - γ=1 (fluid): α = 0 (no drag)
    //   - γ=0 (solid): α = α_max (full drag)
    //   - γ=0.5 (buffer): α = α_max * 0.5 / (1 + 0.5*q) (significant drag!)
    //
    // With q=1: γ=0.5 gives α = 0.33 * α_max (strong penalization)
    // This prevents flow from "leaking" into intermediate density regions.
    
    float q = brinkmanQ;
    
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float g = gamma(i, j);
            // CORRECT Haertel RAMP: α = α_max * (1-γ) / (1 + q*γ)
            alpha(i, j) = brinkmanAlphaMax * (1.0f - g) / (1.0f + q * g + 1e-10f);
        }
    }
    
    // Keep Brinkman wall forcing symmetric with the active pressure/momentum rows.
    // Active cell rows are 0..M-2 (mapped from pressure rows 1..M-1).
    // Row M-1 is padding/unused in the momentum-pressure coupling, so do not
    // force it solid (avoids a fake extra wall band in exports).
    if (M >= 2) {
        const int wallBottomRow = 0;
        const int wallTopRow = M - 2;
        for (int j = 0; j < N; ++j) {
            alpha(wallBottomRow, j) = brinkmanAlphaMax;
            alpha(wallTopRow, j) = brinkmanAlphaMax;
            gamma(wallBottomRow, j) = 0.0f;
            gamma(wallTopRow, j) = 0.0f;
        }
    }
    
    std::cout << "Brinkman alpha field built (RAMP, q=" << brinkmanQ 
              << ", alpha_max=" << std::scientific << brinkmanAlphaMax 
              << ")" << std::defaultfloat << std::endl;
}

// ------------------------------------------------------------------
// Build Fluid Masks (Precomputed for Speed)
// ------------------------------------------------------------------
// ------------------------------------------------------------------
// Main Entry Point
// ------------------------------------------------------------------
int main() {
    SIMPLE solver;
    solver.runIterations();
    return 0;
}
