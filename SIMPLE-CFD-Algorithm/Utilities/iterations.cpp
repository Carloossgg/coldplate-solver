#include "SIMPLE.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline int idx(int i, int j, int cols) { return i * cols + j; }

// Fast mask lookup macros (inline for speed)
#define IS_FLUID_U(i, j) (isFluidU[(i) * N + (j)])
#define IS_FLUID_V(i, j) (isFluidV[(i) * (N + 1) + (j)])
#define IS_FLUID_P(i, j) (isFluidP[(i) * (N + 1) + (j)])

// Helper to get alpha at a u-velocity location (between pressure cells)
inline double getAlphaU(const Eigen::MatrixXd& alpha, int i, int j, int M, int N) {
    if (i < 1 || i >= M || j < 1 || j >= N - 1) return 0.0;
    int ci = i - 1;
    int cj1 = j - 1;
    int cj2 = j;
    if (ci < 0 || ci >= alpha.rows()) return 0.0;
    if (cj1 < 0 || cj1 >= alpha.cols()) return 0.0;
    if (cj2 < 0 || cj2 >= alpha.cols()) return alpha(ci, cj1);
    return 0.5 * (alpha(ci, cj1) + alpha(ci, cj2));
}

// Helper to get alpha at a v-velocity location
inline double getAlphaV(const Eigen::MatrixXd& alpha, int i, int j, int M, int N) {
    if (i < 1 || i >= M - 1 || j < 1 || j >= N) return 0.0;
    int ci1 = i - 1;
    int ci2 = i;
    int cj = j - 1;
    if (cj < 0 || cj >= alpha.cols()) return 0.0;
    if (ci1 < 0 || ci1 >= alpha.rows()) return 0.0;
    if (ci2 < 0 || ci2 >= alpha.rows()) return alpha(ci1, cj);
    return 0.5 * (alpha(ci1, cj) + alpha(ci2, cj));
}

double SIMPLE::calculateStep(int& pressureIterations)
{
    const double vol = hx * hy;
    const bool pseudoActive = enablePseudoTimeStepping;
    
    // Thread-local max residuals
    double localResidU = 0.0;
    double localResidV = 0.0;
    double localResidMass = 0.0;
    
    // Transient term residual tracking
    double localTransResidU = 0.0;
    double localTransResidV = 0.0;

    // Pseudo-transient for stability
    // Use cell size for CFL calculation (this is correct for stability)
    // The mesh-independence comes from running to true steady state (TransRes → 0)
    const double hChar = std::min(hx, hy);
    
    double uDtMin = timeStep, uDtMax = 0.0, uDtSum = 0.0;
    double vDtMin = timeStep, vDtMax = 0.0, vDtSum = 0.0;
    long long uDtCount = 0;
    long long vDtCount = 0;

    auto updateStats = [&](double dt, double& minDt, double& maxDt, double& sumDt, long long& count) {
        minDt = std::min(minDt, dt);
        maxDt = std::max(maxDt, dt);
        sumDt += dt;
        count++;
    };

    // Compute pseudo-dt using cell size for CFL (standard approach)
    auto computeLocalDt = [&](double normalVel) -> double {
        if (!pseudoActive) return std::numeric_limits<double>::infinity();
        if (!useLocalPseudoTime) return timeStep;
        double speed = std::max(std::abs(normalVel), minPseudoSpeed);
        double dtCfl = pseudoCFL * hChar / speed;
        return std::min(timeStep, dtCfl);
    };

    // Save old velocities
    uOld = u;
    vOld = v;

    // Maximum velocity for bounds checking (tighter for stability)
    const double maxVel = 3.0 * std::max(std::abs(inletVelocity), 0.1);
    
    // Second-order upwind (SOU) scheme flag
    const bool useSOU = (convectionScheme == 1);

    // =========================================================================
    // STEP 1: U-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    #pragma omp parallel for collapse(2) reduction(max:localResidU,localTransResidU) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            
            // Fast mask check instead of checkBoundaries + isSolidCell
            if (!IS_FLUID_U(i, j) || checkBoundaries(i, j) == 1.0) {
                uStar(i, j) = 0.0;
                dE(i, j) = 0.0;
                continue;
            }

            double De = eta * hy / hx;
            double Dn = eta * hx / hy;

            // Face velocities for flux calculation
            double ue = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j) + u(i, j + 1))));
            double uw = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j - 1) + u(i, j))));
            double vn = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i - 1, j) + v(i - 1, j + 1))));
            double vs = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i, j) + v(i, j + 1))));

            double Fe = rho * hy * ue;
            double Fw = rho * hy * uw;
            double Fn = rho * hx * vn;
            double Fs = rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            double aE = De + std::max(0.0, -Fe);
            double aW = De + std::max(0.0, Fw);
            double aN = Dn + std::max(0.0, -Fn);
            double aS = Dn + std::max(0.0, Fs);

            // Second-order upwind correction (deferred correction approach)
            // SOU adds a source term to correct first-order upwind to second-order
            double Sdc = 0.0;  // Deferred correction source term
            
            if (useSOU) {
                // East face: u_e using SOU
                // If Fe > 0 (flow to east): u_e = u_P + 0.5*(u_P - u_W) = 1.5*u_P - 0.5*u_W
                // If Fe < 0 (flow to west): u_e = u_E + 0.5*(u_E - u_EE) = 1.5*u_E - 0.5*u_EE
                if (Fe >= 0.0) {
                    // Upwind from west: need u(i,j-1) for second-order
                    if (j >= 2) {
                        double u_sou_e = 1.5 * u(i, j) - 0.5 * u(i, j - 1);
                        double u_fou_e = u(i, j);  // First-order upwind value
                        Sdc -= Fe * (u_sou_e - u_fou_e);
                    }
                } else {
                    // Upwind from east: need u(i,j+2) for second-order
                    if (j + 2 < N) {
                        double u_sou_e = 1.5 * u(i, j + 1) - 0.5 * u(i, j + 2);
                        double u_fou_e = u(i, j + 1);  // First-order upwind value
                        Sdc -= Fe * (u_sou_e - u_fou_e);
                    }
                }
                
                // West face: u_w using SOU
                if (Fw >= 0.0) {
                    // Upwind from west: need u(i,j-2) for second-order
                    if (j >= 2) {
                        double u_sou_w = 1.5 * u(i, j - 1) - 0.5 * u(i, j - 2);
                        double u_fou_w = u(i, j - 1);
                        Sdc += Fw * (u_sou_w - u_fou_w);
                    }
                } else {
                    // Upwind from east: need u(i,j+1) for second-order  
                    if (j + 1 < N) {
                        double u_sou_w = 1.5 * u(i, j) - 0.5 * u(i, j + 1);
                        double u_fou_w = u(i, j);
                        Sdc += Fw * (u_sou_w - u_fou_w);
                    }
                }
                
                // North face: u_n using SOU
                if (Fn >= 0.0) {
                    if (i >= 2) {
                        double u_sou_n = 1.5 * u(i - 1, j) - 0.5 * u(i - 2, j);
                        double u_fou_n = u(i - 1, j);
                        Sdc -= Fn * (u_sou_n - u_fou_n);
                    }
                } else {
                    if (i + 2 <= M) {
                        double u_sou_n = 1.5 * u(i, j) - 0.5 * u(i + 1, j);
                        double u_fou_n = u(i, j);
                        Sdc -= Fn * (u_sou_n - u_fou_n);
                    }
                }
                
                // South face: u_s using SOU
                if (Fs >= 0.0) {
                    if (i + 2 <= M) {
                        double u_sou_s = 1.5 * u(i + 1, j) - 0.5 * u(i + 2, j);
                        double u_fou_s = u(i + 1, j);
                        Sdc += Fs * (u_sou_s - u_fou_s);
                    }
                } else {
                    if (i >= 2) {
                        double u_sou_s = 1.5 * u(i, j) - 0.5 * u(i - 1, j);
                        double u_fou_s = u(i, j);
                        Sdc += Fs * (u_sou_s - u_fou_s);
                    }
                }
            }

            double sumA = aE + aW + aN + aS;
            double dtLocal = computeLocalDt(u(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, uDtMin, uDtMax, uDtSum, uDtCount);
            }
            double transCoeffLocal = rho * vol / dtLocal;

            double aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal;
            aP0 = std::max(aP0, 1e-10);

            // SIMPLE: dDenom = aP0 + sumA (neglects neighbor contributions)
            // SIMPLEC: dDenom = aP0 - sumA (accounts for neighbor contributions consistently)
            double dDenom = useSIMPLEC ? std::max(aP0 - sumA, 1e-12) : std::max(aP0 + sumA, 1e-12);
            dE(i, j) = hy / dDenom;

            double Sp = (p(i, j) - p(i, j + 1)) * hy;
            Sp += transCoeffLocal * uOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            double aP = aP0 / uvAlpha;
            Sp += (1.0 - uvAlpha) / uvAlpha * aP0 * u(i, j);

            double uNew = (aE * u(i, j + 1) + aW * u(i, j - 1) +
                          aN * u(i - 1, j) + aS * u(i + 1, j) + Sp) / aP;

            uNew = std::max(-maxVel, std::min(maxVel, uNew));

            double diff = std::abs(uNew - u(i, j));
            if (diff > localResidU) localResidU = diff;
            
            // Track transient term residual: |transCoeff * (uNew - u)|
            // At true steady state, this should be ~0
            // This measures how much the pseudo-transient term is affecting the solution
            double transTermU = transCoeffLocal * diff;
            if (transTermU > localTransResidU) localTransResidU = transTermU;
            
            uStar(i, j) = uNew;
        }
    }
    residU = localResidU;
    transientResidU = localTransResidU;

    // =========================================================================
    // STEP 2: V-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    localResidV = 0.0;
    
    #pragma omp parallel for collapse(2) reduction(max:localResidV,localTransResidV) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            
            // Fast mask check instead of checkBoundaries + isSolidCell
            if (!IS_FLUID_V(i, j) || checkBoundaries(i, j) == 1.0) {
                vStar(i, j) = 0.0;
                dN(i, j) = 0.0;
                continue;
            }

            double De = eta * hy / hx;
            double Dn = eta * hx / hy;

            // Face velocities for flux calculation
            double ue = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j) + u(i + 1, j))));
            double uw = std::max(-maxVel, std::min(maxVel, 0.5 * (u(i, j - 1) + u(i + 1, j - 1))));
            double vn = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i, j) + v(i + 1, j))));
            double vs = std::max(-maxVel, std::min(maxVel, 0.5 * (v(i - 1, j) + v(i, j))));

            double Fe = rho * hy * ue;
            double Fw = rho * hy * uw;
            double Fn = rho * hx * vn;
            double Fs = rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            double aE = De + std::max(0.0, -Fe);
            double aW = De + std::max(0.0, Fw);
            double aN = Dn + std::max(0.0, -Fn);
            double aS = Dn + std::max(0.0, Fs);

            // Second-order upwind correction (deferred correction approach)
            double Sdc = 0.0;
            
            if (useSOU) {
                // East face: v_e using SOU
                if (Fe >= 0.0) {
                    if (j >= 2) {
                        double v_sou_e = 1.5 * v(i, j) - 0.5 * v(i, j - 1);
                        double v_fou_e = v(i, j);
                        Sdc -= Fe * (v_sou_e - v_fou_e);
                    }
                } else {
                    if (j + 2 <= N) {
                        double v_sou_e = 1.5 * v(i, j + 1) - 0.5 * v(i, j + 2);
                        double v_fou_e = v(i, j + 1);
                        Sdc -= Fe * (v_sou_e - v_fou_e);
                    }
                }
                
                // West face: v_w using SOU
                if (Fw >= 0.0) {
                    if (j >= 2) {
                        double v_sou_w = 1.5 * v(i, j - 1) - 0.5 * v(i, j - 2);
                        double v_fou_w = v(i, j - 1);
                        Sdc += Fw * (v_sou_w - v_fou_w);
                    }
                } else {
                    if (j + 1 <= N) {
                        double v_sou_w = 1.5 * v(i, j) - 0.5 * v(i, j + 1);
                        double v_fou_w = v(i, j);
                        Sdc += Fw * (v_sou_w - v_fou_w);
                    }
                }
                
                // North face: v_n using SOU
                if (Fn >= 0.0) {
                    if (i >= 1) {
                        double v_sou_n = 1.5 * v(i, j) - 0.5 * v(i - 1, j);
                        double v_fou_n = v(i, j);
                        Sdc -= Fn * (v_sou_n - v_fou_n);
                    }
                } else {
                    if (i + 2 < M) {
                        double v_sou_n = 1.5 * v(i + 1, j) - 0.5 * v(i + 2, j);
                        double v_fou_n = v(i + 1, j);
                        Sdc -= Fn * (v_sou_n - v_fou_n);
                    }
                }
                
                // South face: v_s using SOU
                if (Fs >= 0.0) {
                    if (i + 2 < M) {
                        double v_sou_s = 1.5 * v(i + 1, j) - 0.5 * v(i + 2, j);
                        double v_fou_s = v(i + 1, j);
                        Sdc += Fs * (v_sou_s - v_fou_s);
                    }
                } else {
                    if (i >= 1) {
                        double v_sou_s = 1.5 * v(i, j) - 0.5 * v(i - 1, j);
                        double v_fou_s = v(i, j);
                        Sdc += Fs * (v_sou_s - v_fou_s);
                    }
                }
            }

            double sumA = aE + aW + aN + aS;
            double dtLocal = computeLocalDt(v(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, vDtMin, vDtMax, vDtSum, vDtCount);
            }
            double transCoeffLocal = rho * vol / dtLocal;

            double aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal;
            aP0 = std::max(aP0, 1e-10);

            // SIMPLE: dDenom = aP0 + sumA (neglects neighbor contributions)
            // SIMPLEC: dDenom = aP0 - sumA (accounts for neighbor contributions consistently)
            double dDenom = useSIMPLEC ? std::max(aP0 - sumA, 1e-12) : std::max(aP0 + sumA, 1e-12);
            dN(i, j) = hx / dDenom;

            double Sp = (p(i, j) - p(i + 1, j)) * hx;
            Sp += transCoeffLocal * vOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            double aP = aP0 / uvAlpha;
            Sp += (1.0 - uvAlpha) / uvAlpha * aP0 * v(i, j);

            double vNew = (aE * v(i, j + 1) + aW * v(i, j - 1) +
                          aN * v(i + 1, j) + aS * v(i - 1, j) + Sp) / aP;

            vNew = std::max(-maxVel, std::min(maxVel, vNew));

            double diff = std::abs(vNew - v(i, j));
            if (diff > localResidV) localResidV = diff;
            
            // Track transient term residual: |transCoeff * (vNew - v)|
            double transTermV = transCoeffLocal * diff;
            if (transTermV > localTransResidV) localTransResidV = transTermV;
            
            vStar(i, j) = vNew;
        }
    }
    residV = localResidV;
    transientResidV = localTransResidV;

    setVelocityBoundaryConditions(uStar, vStar);

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && uDtCount > 0) {
        pseudoStatsU.min = uDtMin;
        pseudoStatsU.max = uDtMax;
        pseudoStatsU.avg = uDtSum / double(uDtCount);
        pseudoStatsU.samples = uDtCount;
        pseudoStatsU.valid = true;
    } else {
        pseudoStatsU.valid = false;
    }

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && vDtCount > 0) {
        pseudoStatsV.min = vDtMin;
        pseudoStatsV.max = vDtMax;
        pseudoStatsV.avg = vDtSum / double(vDtCount);
        pseudoStatsV.samples = vDtCount;
        pseudoStatsV.valid = true;
    } else {
        pseudoStatsV.valid = false;
    }

    // =========================================================================
    // STEP 3: PRESSURE CORRECTION (Iterative SOR or Direct Sparse Solver)
    // =========================================================================
    pStar.setZero();
    
    // Calculate mass residual first (parallel with precomputed masks)
    localResidMass = 0.0;
    #pragma omp parallel for collapse(2) reduction(max:localResidMass) schedule(guided)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            if (!IS_FLUID_P(i, j) || checkBoundaries(i, j) == 1.0) continue;
            
            double massE = rho * uStar(i, j) * hy;
            double massW = rho * uStar(i, j - 1) * hy;
            double massN = rho * vStar(i, j) * hx;
            double massS = rho * vStar(i - 1, j) * hx;
            double b = std::abs(massW - massE + massS - massN);
            if (b > localResidMass) localResidMass = b;
        }
    }
    residMass = localResidMass;

    bool directSolverSucceeded = false;
    if (useDirectPressureSolver) {
        // =====================================================================
        // DIRECT SPARSE MATRIX SOLVER
        // =====================================================================
        // Build mapping from 2D grid indices (i,j) to linear indices
        // Only include fluid cells that are not at outlet boundary (j != N-1)
        std::vector<std::pair<int, int>> gridToLinear;  // maps linear index -> (i, j)
        std::vector<std::vector<int>> linearIndex(M + 1, std::vector<int>(N + 1, -1));  // maps (i, j) -> linear index
        
        int nFluidCells = 0;
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N - 1; ++j) {  // Exclude outlet boundary (j == N-1)
                if (IS_FLUID_P(i, j) && checkBoundaries(i, j) != 1.0) {
                    linearIndex[i][j] = nFluidCells;
                    gridToLinear.push_back({i, j});
                    nFluidCells++;
                }
            }
        }
        
        if (nFluidCells > 0) {
            // Build sparse matrix and RHS vector
            std::vector<Eigen::Triplet<double>> triplets;
            Eigen::VectorXd rhs = Eigen::VectorXd::Zero(nFluidCells);
            triplets.reserve(nFluidCells * 5);  // Estimate: 5 non-zeros per row (diagonal + 4 neighbors)
            
            if (parallelizeDirectSolver && nFluidCells > 1000) {
                // Parallel assembly: use thread-local storage
                std::vector<std::vector<Eigen::Triplet<double>>> threadTriplets(omp_get_max_threads());
                std::vector<Eigen::VectorXd> threadRhs(omp_get_max_threads());
                for (int t = 0; t < omp_get_max_threads(); ++t) {
                    threadTriplets[t].reserve(nFluidCells * 5 / omp_get_max_threads());
                    threadRhs[t] = Eigen::VectorXd::Zero(nFluidCells);
                }
                
                #pragma omp parallel
                {
                    int tid = omp_get_thread_num();
                    #pragma omp for schedule(static)
                    for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                        int i = gridToLinear[linIdx].first;
                        int j = gridToLinear[linIdx].second;
                        
                        // Compute coefficients
                        double aE_p = (j < N - 1 && IS_FLUID_P(i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                        double aW_p = (j > 1 && IS_FLUID_P(i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                        double aN_p = (i < M - 1 && IS_FLUID_P(i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                        double aS_p = (i > 1 && IS_FLUID_P(i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;
                        
                        double aP_p = aE_p + aW_p + aN_p + aS_p;
                        
                        if (aP_p < 1e-20) {
                            // Singular cell - set to zero
                            threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, linIdx, 1.0));
                            threadRhs[tid](linIdx) = 0.0;
                            continue;
                        }
                        
                        // Mass residual (RHS)
                        double massE = rho * uStar(i, j) * hy;
                        double massW = rho * uStar(i, j - 1) * hy;
                        double massN = rho * vStar(i, j) * hx;
                        double massS = rho * vStar(i - 1, j) * hx;
                        double b_rhs = massW - massE + massS - massN;
                        
                        // Diagonal term
                        threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, linIdx, aP_p));
                        
                        // Off-diagonal terms (neighbors)
                        // East neighbor
                        if (j < N - 1 && IS_FLUID_P(i, j + 1) && checkBoundaries(i, j + 1) != 1.0) {
                            int neighborLinIdx = linearIndex[i][j + 1];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aE_p));
                            }
                        }
                        
                        // West neighbor
                        if (j > 1 && IS_FLUID_P(i, j - 1) && checkBoundaries(i, j - 1) != 1.0) {
                            int neighborLinIdx = linearIndex[i][j - 1];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aW_p));
                            }
                        }
                        
                        // North neighbor
                        if (i < M - 1 && IS_FLUID_P(i + 1, j) && checkBoundaries(i + 1, j) != 1.0) {
                            int neighborLinIdx = linearIndex[i + 1][j];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aN_p));
                            }
                        }
                        
                        // South neighbor
                        if (i > 1 && IS_FLUID_P(i - 1, j) && checkBoundaries(i - 1, j) != 1.0) {
                            int neighborLinIdx = linearIndex[i - 1][j];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aS_p));
                            }
                        }
                        
                        threadRhs[tid](linIdx) = b_rhs;
                    }
                }
                
                // Merge thread-local triplets and RHS
                for (int t = 0; t < omp_get_max_threads(); ++t) {
                    triplets.insert(triplets.end(), threadTriplets[t].begin(), threadTriplets[t].end());
                    rhs += threadRhs[t];
                }
            } else {
                // Serial assembly (for small systems or when parallelization is disabled)
                for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                    int i = gridToLinear[linIdx].first;
                    int j = gridToLinear[linIdx].second;
                    
                    // Compute coefficients
                    double aE_p = (j < N - 1 && IS_FLUID_P(i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                    double aW_p = (j > 1 && IS_FLUID_P(i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                    double aN_p = (i < M - 1 && IS_FLUID_P(i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                    double aS_p = (i > 1 && IS_FLUID_P(i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;
                    
                    double aP_p = aE_p + aW_p + aN_p + aS_p;
                    
                    if (aP_p < 1e-20) {
                        // Singular cell - set to zero
                        triplets.push_back(Eigen::Triplet<double>(linIdx, linIdx, 1.0));
                        rhs(linIdx) = 0.0;
                        continue;
                    }
                    
                    // Mass residual (RHS)
                    double massE = rho * uStar(i, j) * hy;
                    double massW = rho * uStar(i, j - 1) * hy;
                    double massN = rho * vStar(i, j) * hx;
                    double massS = rho * vStar(i - 1, j) * hx;
                    double b_rhs = massW - massE + massS - massN;
                    
                    // Diagonal term
                    triplets.push_back(Eigen::Triplet<double>(linIdx, linIdx, aP_p));
                    
                    // Off-diagonal terms (neighbors)
                    // East neighbor
                    if (j < N - 1 && IS_FLUID_P(i, j + 1) && checkBoundaries(i, j + 1) != 1.0) {
                        int neighborLinIdx = linearIndex[i][j + 1];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aE_p));
                        }
                    } else if (j + 1 == N - 1 && aE_p > 0.0) {
                        // East neighbor is outlet boundary: p' = 0, so contribution is already zero in RHS
                        // No matrix entry needed
                    }
                    
                    // West neighbor
                    if (j > 1 && IS_FLUID_P(i, j - 1) && checkBoundaries(i, j - 1) != 1.0) {
                        int neighborLinIdx = linearIndex[i][j - 1];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aW_p));
                        }
                    }
                    
                    // North neighbor
                    if (i < M - 1 && IS_FLUID_P(i + 1, j) && checkBoundaries(i + 1, j) != 1.0) {
                        int neighborLinIdx = linearIndex[i + 1][j];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aN_p));
                        }
                    }
                    
                    // South neighbor
                    if (i > 1 && IS_FLUID_P(i - 1, j) && checkBoundaries(i - 1, j) != 1.0) {
                        int neighborLinIdx = linearIndex[i - 1][j];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aS_p));
                        }
                    }
                    
                    rhs(linIdx) = b_rhs;
                }
            }
            
            // Build sparse matrix (optimized: compressed storage)
            Eigen::SparseMatrix<double> A(nFluidCells, nFluidCells);
            A.setFromTriplets(triplets.begin(), triplets.end());
            A.makeCompressed();
            
            // Solve using direct sparse Cholesky LDLT decomposition (for symmetric positive definite matrix)
            // Uses AMD ordering internally for optimal fill-in reduction
            Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            solver.compute(A);
            
            if (solver.info() != Eigen::Success) {
                std::cerr << "Warning: Direct pressure solver factorization failed. Falling back to iterative solver." << std::endl;
            } else {
                Eigen::VectorXd pCorr = solver.solve(rhs);
                
                if (solver.info() != Eigen::Success) {
                    std::cerr << "Warning: Direct pressure solver solution failed. Falling back to iterative solver." << std::endl;
                } else {
                    // Map solution back to pStar matrix (parallelized)
                    if (parallelizeDirectSolver && nFluidCells > 1000) {
                        #pragma omp parallel for schedule(static)
                        for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                            int i = gridToLinear[linIdx].first;
                            int j = gridToLinear[linIdx].second;
                            pStar(i, j) = std::max(-10000.0, std::min(10000.0, pCorr(linIdx)));  // Bounds
                        }
                        // Set outlet boundary condition (p' = 0) - parallelized
                        #pragma omp parallel for schedule(static)
                        for (int i = 1; i < M; ++i) {
                            pStar(i, N - 1) = 0.0;
                        }
                    } else {
                        // Serial mapping for small systems
                        for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                            int i = gridToLinear[linIdx].first;
                            int j = gridToLinear[linIdx].second;
                            pStar(i, j) = std::max(-10000.0, std::min(10000.0, pCorr(linIdx)));  // Bounds
                        }
                        // Set outlet boundary condition (p' = 0)
                        for (int i = 1; i < M; ++i) {
                            pStar(i, N - 1) = 0.0;
                        }
                    }
                    pressureIterations = 1;  // Direct solver does it in one "iteration"
                    directSolverSucceeded = true;
                }
            }
        }
    }
    
    // If direct solver failed or was disabled, use iterative SOR
    if (!directSolverSucceeded) {
        // Compute optimal SOR omega if not set (Chebyshev acceleration for 2D Poisson)
        // For a rectangular grid: omega_opt = 2 / (1 + sin(pi * h / L))
        // where h is the mesh spacing and L is the domain size
        // For large grids this approaches 2.0
        double omega = sorOmega;
        if (omega <= 0.0 || omega >= 2.0) {
            // Optimal omega for 2D Laplacian with Dirichlet BCs
            int maxDim = std::max(M, N);
            double spectralRadius = std::cos(M_PI / maxDim);
            omega = 2.0 / (1.0 + std::sqrt(1.0 - spectralRadius * spectralRadius));
            // Clamp for safety (should be between 1.0 and 2.0)
            omega = std::max(1.0, std::min(1.95, omega));
        }

        // Red-Black SOR pressure correction
        pressureIterations = 0;
        for (int pIter = 0; pIter < maxPressureIter; ++pIter) {
            double maxChange = 0.0;
            pressureIterations = pIter + 1;
            
            // RED phase (i+j even) - use guided scheduling for geometries with many solids
            #pragma omp parallel for collapse(2) reduction(max:maxChange) schedule(guided)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    if ((i + j) % 2 != 0) continue;  // Skip black cells
                    
                    // Fast mask check
                    if (!IS_FLUID_P(i, j) || checkBoundaries(i, j) == 1.0) {
                        pStar(i, j) = 0.0;
                        continue;
                    }

                    double aE_p = (j < N - 1 && IS_FLUID_P(i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                    double aW_p = (j > 1 && IS_FLUID_P(i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                    double aN_p = (i < M - 1 && IS_FLUID_P(i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                    double aS_p = (i > 1 && IS_FLUID_P(i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;

                    double aP_p = aE_p + aW_p + aN_p + aS_p;
                    
                    if (j == N - 1 || aP_p < 1e-20) {
                        pStar(i, j) = 0.0;
                        continue;
                    }

                    double massE = rho * uStar(i, j) * hy;
                    double massW = rho * uStar(i, j - 1) * hy;
                    double massN = rho * vStar(i, j) * hx;
                    double massS = rho * vStar(i - 1, j) * hx;
                    double b_rhs = massW - massE + massS - massN;

                    double pE = (j < N - 1) ? pStar(i, j + 1) : 0.0;
                    double pW = (j > 1) ? pStar(i, j - 1) : 0.0;
                    double pN = (i < M - 1) ? pStar(i + 1, j) : 0.0;
                    double pS = (i > 1) ? pStar(i - 1, j) : 0.0;

                    double pGS = (aE_p * pE + aW_p * pW + aN_p * pN + aS_p * pS + b_rhs) / aP_p;
                    double pNew = pStar(i, j) + omega * (pGS - pStar(i, j));
                    pNew = std::max(-10000.0, std::min(10000.0, pNew));  // Bounds

                    double change = std::abs(pNew - pStar(i, j));
                    if (change > maxChange) maxChange = change;
                    pStar(i, j) = pNew;
                }
            }
            
            // BLACK phase (i+j odd) - use guided scheduling
            #pragma omp parallel for collapse(2) reduction(max:maxChange) schedule(guided)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    if ((i + j) % 2 != 1) continue;  // Skip red cells
                    
                    // Fast mask check
                    if (!IS_FLUID_P(i, j) || checkBoundaries(i, j) == 1.0) {
                        pStar(i, j) = 0.0;
                        continue;
                    }

                    double aE_p = (j < N - 1 && IS_FLUID_P(i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                    double aW_p = (j > 1 && IS_FLUID_P(i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                    double aN_p = (i < M - 1 && IS_FLUID_P(i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                    double aS_p = (i > 1 && IS_FLUID_P(i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;

                    double aP_p = aE_p + aW_p + aN_p + aS_p;
                    
                    if (j == N - 1 || aP_p < 1e-20) {
                        pStar(i, j) = 0.0;
                        continue;
                    }

                    double massE = rho * uStar(i, j) * hy;
                    double massW = rho * uStar(i, j - 1) * hy;
                    double massN = rho * vStar(i, j) * hx;
                    double massS = rho * vStar(i - 1, j) * hx;
                    double b_rhs = massW - massE + massS - massN;

                    double pE = (j < N - 1) ? pStar(i, j + 1) : 0.0;
                    double pW = (j > 1) ? pStar(i, j - 1) : 0.0;
                    double pN = (i < M - 1) ? pStar(i + 1, j) : 0.0;
                    double pS = (i > 1) ? pStar(i - 1, j) : 0.0;

                    double pGS = (aE_p * pE + aW_p * pW + aN_p * pN + aS_p * pS + b_rhs) / aP_p;
                    double pNew = pStar(i, j) + omega * (pGS - pStar(i, j));
                    pNew = std::max(-10000.0, std::min(10000.0, pNew));  // Bounds

                    double change = std::abs(pNew - pStar(i, j));
                    if (change > maxChange) maxChange = change;
                    pStar(i, j) = pNew;
                }
            }

            if (maxChange < pTol) break;
        }
    }

    // =========================================================================
    // STEP 4: CORRECT PRESSURE (Parallelized with precomputed masks)
    // =========================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            if (IS_FLUID_P(i, j)) {
                p(i, j) += pAlpha * pStar(i, j);
            }
        }
    }

    // Set pressure boundary conditions (outlet reference pressure = 0.0)
    // No normalization needed - boundary condition handles reference pressure
    setPressureBoundaryConditions(p);

    // =========================================================================
    // STEP 5: CORRECT VELOCITIES (Parallelized)
    // =========================================================================
    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            if (checkBoundaries(i, j) == 1.0) continue;
            if (getAlphaU(alpha, i, j, M, N) > 1e6) continue;
            
            double du = dE(i, j) * (pStar(i, j) - pStar(i, j + 1));
            du = std::max(-maxVel, std::min(maxVel, du));
            u(i, j) = uStar(i, j) + du;
            u(i, j) = std::max(-maxVel, std::min(maxVel, u(i, j)));
        }
    }

    #pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            if (checkBoundaries(i, j) == 1.0) continue;
            if (getAlphaV(alpha, i, j, M, N) > 1e6) continue;
            
            double dv = dN(i, j) * (pStar(i, j) - pStar(i + 1, j));
            dv = std::max(-maxVel, std::min(maxVel, dv));
            v(i, j) = vStar(i, j) + dv;
            v(i, j) = std::max(-maxVel, std::min(maxVel, v(i, j)));
        }
    }

    setVelocityBoundaryConditions(u, v);

    // =========================================================================
    // STEP 6: MASS BALANCE CORRECTION (Serial - small loop)
    // =========================================================================
    double massIn = 0.0, massOut = 0.0;
    for (int i = 1; i < M; ++i) {
        massIn += rho * u(i, 0) * hy;
        massOut += rho * u(i, N - 1) * hy;
    }

    if (std::abs(massOut) > 1e-10 && std::abs(massIn) > 1e-10) {
        double ratio = massIn / massOut;
        if (ratio > 0.5 && ratio < 2.0) {
            for (int i = 1; i < M; ++i) {
                u(i, N - 1) *= ratio;
            }
        }
    }

    return std::max({residU, residV, residMass});
}
