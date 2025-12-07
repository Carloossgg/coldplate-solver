// File: Utilities/pressure_solver.cpp
// Author: Peter Tcherkezian
// Description: Pressure correction assembly and solve:
//   - builds sparse matrix/RHS from staggered-grid pressure correction equations (respecting fluid masks/boundaries)
//   - supports sparse direct solve (Eigen SimplicialLDLT) or SOR iterative solve
//   - caches grid-to-linear mappings and symbolic factorization to accelerate repeated solves across iterations
#include "SIMPLE.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <omp.h>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Solve pressure correction (direct or SOR), updating pStar and residMass.
// Returns true if direct solver succeeded.
namespace {

struct PressureMapping {
    std::vector<std::pair<int, int>> gridToLinear;  // linear -> (i,j)
    std::vector<std::vector<int>> linearIndex;      // (i,j) -> linear
    int nFluidCells = 0;
};

PressureMapping buildPressureMapping(SIMPLE& s) {
    PressureMapping map;
    map.linearIndex.assign(s.M + 1, std::vector<int>(s.N + 1, -1));
    int n = 0;
    for (int i = 1; i < s.M; ++i) {
        for (int j = 1; j < s.N - 1; ++j) {
            if (fluidP(s, i, j) && s.checkBoundaries(i, j) != 1.0) {
                map.linearIndex[i][j] = n;
                map.gridToLinear.push_back({i, j});
                n++;
            }
        }
    }
    map.nFluidCells = n;
    return map;
}

} // namespace

bool SIMPLE::solvePressureSystem(int& pressureIterations, double& localResidMass) {
    pStar.setZero();
    localResidMass = 0.0;

    // Calculate mass residual first (parallel with precomputed masks)
    #pragma omp parallel for collapse(2) reduction(max:localResidMass) schedule(guided)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N; ++j) {
            if (!fluidP(*this, i, j) || checkBoundaries(i, j) == 1.0) continue;

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
        static PressureMapping mapping;
        static int cachedM = -1;
        static int cachedN = -1;

        bool rebuildMapping = (cachedM != M) || (cachedN != N) || mapping.gridToLinear.empty();
        if (rebuildMapping) {
            mapping = buildPressureMapping(*this);
            cachedM = M;
            cachedN = N;
        }

        const int nFluidCells = mapping.nFluidCells;

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
                        int i = mapping.gridToLinear[linIdx].first;
                        int j = mapping.gridToLinear[linIdx].second;

                        double aE_p = (j < N - 1 && fluidP(*this, i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                        double aW_p = (j > 1 && fluidP(*this, i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                        double aN_p = (i < M - 1 && fluidP(*this, i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                        double aS_p = (i > 1 && fluidP(*this, i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;

                        double aP_p = aE_p + aW_p + aN_p + aS_p;

                        if (aP_p < 1e-20) {
                            threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, linIdx, 1.0));
                            threadRhs[tid](linIdx) = 0.0;
                            continue;
                        }

                        double massE = rho * uStar(i, j) * hy;
                        double massW = rho * uStar(i, j - 1) * hy;
                        double massN = rho * vStar(i, j) * hx;
                        double massS = rho * vStar(i - 1, j) * hx;
                        double b_rhs = massW - massE + massS - massN;

                        threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, linIdx, aP_p));

                        if (j < N - 1 && fluidP(*this, i, j + 1) && checkBoundaries(i, j + 1) != 1.0) {
                            int neighborLinIdx = mapping.linearIndex[i][j + 1];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aE_p));
                            }
                        }

                        if (j > 1 && fluidP(*this, i, j - 1) && checkBoundaries(i, j - 1) != 1.0) {
                            int neighborLinIdx = mapping.linearIndex[i][j - 1];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aW_p));
                            }
                        }

                        if (i < M - 1 && fluidP(*this, i + 1, j) && checkBoundaries(i + 1, j) != 1.0) {
                            int neighborLinIdx = mapping.linearIndex[i + 1][j];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aN_p));
                            }
                        }

                        if (i > 1 && fluidP(*this, i - 1, j) && checkBoundaries(i - 1, j) != 1.0) {
                            int neighborLinIdx = mapping.linearIndex[i - 1][j];
                            if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                                threadTriplets[tid].push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aS_p));
                            }
                        }

                        threadRhs[tid](linIdx) = b_rhs;
                    }
                }

                for (int t = 0; t < omp_get_max_threads(); ++t) {
                    triplets.insert(triplets.end(), threadTriplets[t].begin(), threadTriplets[t].end());
                    rhs += threadRhs[t];
                }
            } else {
                for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                    int i = mapping.gridToLinear[linIdx].first;
                    int j = mapping.gridToLinear[linIdx].second;

                    double aE_p = (j < N - 1 && fluidP(*this, i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                    double aW_p = (j > 1 && fluidP(*this, i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                    double aN_p = (i < M - 1 && fluidP(*this, i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                    double aS_p = (i > 1 && fluidP(*this, i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;

                    double aP_p = aE_p + aW_p + aN_p + aS_p;

                    if (aP_p < 1e-20) {
                        triplets.push_back(Eigen::Triplet<double>(linIdx, linIdx, 1.0));
                        rhs(linIdx) = 0.0;
                        continue;
                    }

                    double massE = rho * uStar(i, j) * hy;
                    double massW = rho * uStar(i, j - 1) * hy;
                    double massN = rho * vStar(i, j) * hx;
                    double massS = rho * vStar(i - 1, j) * hx;
                    double b_rhs = massW - massE + massS - massN;

                    triplets.push_back(Eigen::Triplet<double>(linIdx, linIdx, aP_p));

                    if (j < N - 1 && fluidP(*this, i, j + 1) && checkBoundaries(i, j + 1) != 1.0) {
                        int neighborLinIdx = mapping.linearIndex[i][j + 1];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aE_p));
                        }
                    } else if (j + 1 == N - 1 && aE_p > 0.0) {
                        // outlet boundary contribution already zero
                    }

                    if (j > 1 && fluidP(*this, i, j - 1) && checkBoundaries(i, j - 1) != 1.0) {
                        int neighborLinIdx = mapping.linearIndex[i][j - 1];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aW_p));
                        }
                    }

                    if (i < M - 1 && fluidP(*this, i + 1, j) && checkBoundaries(i + 1, j) != 1.0) {
                        int neighborLinIdx = mapping.linearIndex[i + 1][j];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aN_p));
                        }
                    }

                    if (i > 1 && fluidP(*this, i - 1, j) && checkBoundaries(i - 1, j) != 1.0) {
                        int neighborLinIdx = mapping.linearIndex[i - 1][j];
                        if (neighborLinIdx >= 0 && neighborLinIdx < nFluidCells) {
                            triplets.push_back(Eigen::Triplet<double>(linIdx, neighborLinIdx, -aS_p));
                        }
                    }

                    rhs(linIdx) = b_rhs;
                }
            }

            Eigen::SparseMatrix<double> A(nFluidCells, nFluidCells);
            A.setFromTriplets(triplets.begin(), triplets.end());
            A.makeCompressed();

            static Eigen::SimplicialLDLT<Eigen::SparseMatrix<double>> solver;
            static bool patternAnalyzed = false;
            static int lastNFluid = 0;
            if (lastNFluid != nFluidCells) {
                patternAnalyzed = false;
                lastNFluid = nFluidCells;
            }

            if (!patternAnalyzed) {
                solver.analyzePattern(A);
                if (solver.info() == Eigen::Success) {
                    patternAnalyzed = true;
                } else {
                    std::cerr << "Warning: Direct solver analyzePattern failed. Falling back to iterative solver." << std::endl;
                }
            }

            if (patternAnalyzed) {
                solver.factorize(A);
            }

            if (!patternAnalyzed || solver.info() != Eigen::Success) {
                std::cerr << "Warning: Direct pressure solver factorization failed. Falling back to iterative solver." << std::endl;
            } else {
                Eigen::VectorXd pCorr = solver.solve(rhs);

                if (solver.info() != Eigen::Success) {
                    std::cerr << "Warning: Direct pressure solver solution failed. Falling back to iterative solver." << std::endl;
                } else {
                    if (parallelizeDirectSolver && nFluidCells > 1000) {
                        #pragma omp parallel for schedule(static)
                        for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                            int i = mapping.gridToLinear[linIdx].first;
                            int j = mapping.gridToLinear[linIdx].second;
                            pStar(i, j) = std::max(-10000.0, std::min(10000.0, pCorr(linIdx)));
                        }
                        #pragma omp parallel for schedule(static)
                        for (int i = 1; i < M; ++i) {
                            pStar(i, N - 1) = 0.0;
                        }
                    } else {
                        for (int linIdx = 0; linIdx < nFluidCells; ++linIdx) {
                            int i = mapping.gridToLinear[linIdx].first;
                            int j = mapping.gridToLinear[linIdx].second;
                            pStar(i, j) = std::max(-10000.0, std::min(10000.0, pCorr(linIdx)));
                        }
                        for (int i = 1; i < M; ++i) {
                            pStar(i, N - 1) = 0.0;
                        }
                    }
                    pressureIterations = 1;
                    directSolverSucceeded = true;
                }
            }
        }
    }

    if (!directSolverSucceeded) {
        double omega = sorOmega;
        if (omega <= 0.0 || omega >= 2.0) {
            int maxDim = std::max(M, N);
            double spectralRadius = std::cos(M_PI / maxDim);
            omega = 2.0 / (1.0 + std::sqrt(1.0 - spectralRadius * spectralRadius));
            omega = std::max(1.0, std::min(1.95, omega));
        }

        pressureIterations = 0;
        for (int pIter = 0; pIter < maxPressureIter; ++pIter) {
            double maxChange = 0.0;
            pressureIterations = pIter + 1;

            #pragma omp parallel for collapse(2) reduction(max:maxChange) schedule(guided)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    if ((i + j) % 2 != 0) continue;
                    if (!fluidP(*this, i, j) || checkBoundaries(i, j) == 1.0) {
                        pStar(i, j) = 0.0;
                        continue;
                    }

                    double aE_p = (j < N - 1 && fluidP(*this, i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                    double aW_p = (j > 1 && fluidP(*this, i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                    double aN_p = (i < M - 1 && fluidP(*this, i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                    double aS_p = (i > 1 && fluidP(*this, i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;

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
                    pNew = std::max(-10000.0, std::min(10000.0, pNew));

                    double change = std::abs(pNew - pStar(i, j));
                    if (change > maxChange) maxChange = change;
                    pStar(i, j) = pNew;
                }
            }

            #pragma omp parallel for collapse(2) reduction(max:maxChange) schedule(guided)
            for (int i = 1; i < M; ++i) {
                for (int j = 1; j < N; ++j) {
                    if ((i + j) % 2 != 1) continue;
                    if (!fluidP(*this, i, j) || checkBoundaries(i, j) == 1.0) {
                        pStar(i, j) = 0.0;
                        continue;
                    }

                    double aE_p = (j < N - 1 && fluidP(*this, i, j + 1)) ? rho * dE(i, j) * hy : 0.0;
                    double aW_p = (j > 1 && fluidP(*this, i, j - 1)) ? rho * dE(i, j - 1) * hy : 0.0;
                    double aN_p = (i < M - 1 && fluidP(*this, i + 1, j)) ? rho * dN(i, j) * hx : 0.0;
                    double aS_p = (i > 1 && fluidP(*this, i - 1, j)) ? rho * dN(i - 1, j) * hx : 0.0;

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
                    pNew = std::max(-10000.0, std::min(10000.0, pNew));

                    double change = std::abs(pNew - pStar(i, j));
                    if (change > maxChange) maxChange = change;
                    pStar(i, j) = pNew;
                }
            }

            if (maxChange < pTol) break;
        }
    }

    return directSolverSucceeded;
}

