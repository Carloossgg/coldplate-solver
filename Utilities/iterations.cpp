// ============================================================================
// File: Utilities/iterations.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Implements a single SIMPLE iteration on the staggered Cartesian grid.
//   This is the heart of the CFD solver, executing the following steps each iteration:
//
//   1. SAVE OLD VELOCITIES
//      Store u, v for residual calculation and pseudo-transient term
//
//   2. SOLVE U-MOMENTUM (x-direction)
//      - Assemble coefficient matrix with diffusion + convection (FOU or SOU)
//      - Add pseudo-transient term: ρV/Δt * (u - u_old) for stability
//      - Add 2.5D sink term: -(5μ/2Ht²) * u for out-of-plane friction
//      - Apply under-relaxation: u* = α*u_new + (1-α)*u_old
//      - Compute d-coefficient for velocity correction
//
//   3. SOLVE V-MOMENTUM (y-direction)
//      Same process as U-momentum but for vertical velocity
//
//   4. SET VELOCITY BOUNDARY CONDITIONS
//      Apply inlet (fixed velocity), outlet (zero gradient), wall (no-slip)
//
//   5. SOLVE PRESSURE CORRECTION (Poisson equation)
//      - Assemble sparse matrix from continuity equation
//      - Solve using direct LDLT or iterative SOR
//      - Returns mass residual (continuity error)
//
//   6. CORRECT PRESSURE
//      p = p + α_p * p'  (relaxed pressure update)
//
//   7. CORRECT VELOCITIES
//      u = u* + d_E * (p'_P - p'_E)  where d = Δx / a_P
//      v = v* + d_N * (p'_P - p'_N)
//
//   8. MASS BALANCE CORRECTION
//      Scale outlet velocity to enforce global mass conservation
//
//   9. UPDATE CFL RAMP
//      Increase pseudo-CFL as residuals decrease for faster convergence
//
//
// Brinkman Penalization:
//   For topology optimization, solid regions are penalized rather than blocked:
//   - Add source term: F = -α(γ) * u, where α = α_max * (1 - γ)
//   - γ = 1 (fluid): no drag, γ = 0 (solid): maximum drag
//   - This allows gradient-based optimization with continuous γ values
//
// Pseudo-Transient Time Stepping:
//   Adds artificial unsteady term to steady equations for stability:
//   - ρV/Δt * (u - u_old) added to momentum source
//   - Δt computed from CFL: Δt = CFL * Δx / |u|
//   - CFL increases as solution converges (residual-based ramping)
//   - At true steady state, (u - u_old) → 0, so term vanishes
//
// ============================================================================
#include "SIMPLE.h"
#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <omp.h>

// Define PI if not already defined (for SOR optimal omega calculation)
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ============================================================================
// calculateStep: Perform one complete SIMPLE iteration
// ============================================================================
// This function orchestrates a single outer iteration of the SIMPLE algorithm.
// It solves momentum equations, pressure correction, and velocity correction.
//
// Parameters:
//   pressureIterations [out]: Number of inner pressure sweeps (for logging)
//
// Returns:
//   Maximum residual (max of |Δu|, |Δv|, mass imbalance)
// ============================================================================
float SIMPLE::calculateStep(int& pressureIterations)
{
    // -------------------------------------------------------------------------
    // Cell volume and pseudo-transient setup
    // -------------------------------------------------------------------------
    const float vol = hx * hy;                        // Cell volume [m³/m depth]
    const bool pseudoActive = enablePseudoTimeStepping; // Is pseudo-transient enabled?
    
    // Thread-local reduction variables for OpenMP parallelization
    float localResidU = 0.0f;        // Max |u_new - u_old|
    float localResidV = 0.0f;        // Max |v_new - v_old|
    float localResidMass = 0.0f;     // Max mass imbalance
    
    // Transient term residual tracking (monitors convergence to true steady state)
    // At true steady state, these should approach zero
    float localTransResidU = 0.0f;   // Max |ρV/Δt * (u_new - u_old)|
    float localTransResidV = 0.0f;   // Max |ρV/Δt * (v_new - v_old)|

    // -------------------------------------------------------------------------
    // Velocity bounds for stability (prevent unbounded growth)
    // -------------------------------------------------------------------------
    // Moved here so it's available for both implicit and explicit paths
    const float maxVel = 3.0f * std::max(std::abs(inletVelocity), 0.1f);

    // Keep BCs explicitly synchronized before assembling momentum equations.
    // This also enables robust runtime detection of external no-slip walls.
    setVelocityBoundaryConditions(u, v);
    setPressureBoundaryConditions(p);
    const ExternalWallFlags externalWalls = detectExternalNoSlipWalls(*this);
    static bool printedHalfCellActivation = false;
    if (!printedHalfCellActivation) {
        std::cout << "Half-cell diffusion activation:"
                  << " bottom=" << (externalWalls.bottom ? "ON" : "OFF")
                  << " top=" << (externalWalls.top ? "ON" : "OFF")
                  << " left=" << (externalWalls.left ? "ON" : "OFF")
                  << " right=" << (externalWalls.right ? "ON" : "OFF")
                  << std::endl;
        printedHalfCellActivation = true;
    }

    // =========================================================================
    // MOMENTUM SOLVER: Select Implicit (AMG) or Explicit (Point Gauss-Seidel)
    // =========================================================================
    if (momentumSolverType == 1) {
        // -----------------------------------------------------------------
        // IMPLICIT AMG MOMENTUM SOLVER (STAR-CCM+ Style)
        // -----------------------------------------------------------------
        // Assembles full momentum matrix and solves with AMG using
        // Gauss-Seidel relaxation, matching STAR-CCM+ methodology.
        // -----------------------------------------------------------------
        
        // Save old velocities for residual calculation
        uOld = u;
        vOld = v;
        
        // Solve U and V momentum (sequential for stability)
        int maxIters;
        {
            ScopedTimer t("Iter: Momentum Solve (Implicit)");
            solveUMomentumImplicit(*this, localResidU); solveVMomentumImplicit(*this, localResidV);
        }
        residU = localResidU;
        residV = localResidV;
          // Suppress unused variable warning
        
        // Apply boundary conditions to intermediate velocities
        {
            ScopedTimer t("Iter: Momentum BCs");
            setVelocityBoundaryConditions(uStar, vStar);
        }
        
        // Compute pseudo-transient residual for implicit solver
        // Fully disabled when pseudo-time stepping is off.
        {
            ScopedTimer t("Iter: Transient Residual Calculation");
            localTransResidU = 0.0f;
            localTransResidV = 0.0f;

            if (enablePseudoTimeStepping) {
                const float vol = hx * hy;
                const float dt = std::max(timeStep, 1e-20f);
                const float transCoeff = rho * vol / dt;

                #pragma omp parallel for reduction(max:localTransResidU) schedule(static)
                for (int i = 1; i < M; ++i) {
                    for (int j = 1; j < N - 1; ++j) {
                        float diff = std::abs(uStar(i, j) - uOld(i, j));
                        float transTermU = transCoeff * diff;
                        if (transTermU > localTransResidU) localTransResidU = transTermU;
                    }
                }

                #pragma omp parallel for reduction(max:localTransResidV) schedule(static)
                for (int i = 1; i < M - 1; ++i) {
                    for (int j = 1; j < N; ++j) {
                        float diff = std::abs(vStar(i, j) - vOld(i, j));
                        float transTermV = transCoeff * diff;
                        if (transTermV > localTransResidV) localTransResidV = transTermV;
                    }
                }
            }

            transientResidU = localTransResidU;
            transientResidV = localTransResidV;
        }
        
        // Invalidate pseudo-stats (not computed in implicit mode)
        pseudoStatsU.valid = false;
        pseudoStatsV.valid = false;
        
    } else {
        ScopedTimer t("Iter: Momentum Solve (Explicit)");
    // =========================================================================
    // EXPLICIT MOMENTUM SOLVER (Original Point Gauss-Seidel)
    // =========================================================================

    // -------------------------------------------------------------------------
    // Pseudo-transient Δt calculation setup
    // -------------------------------------------------------------------------
    // -------------------------------------------------------------------------
    // Pseudo-transient Δt calculation setup
    // -------------------------------------------------------------------------
    // Use minimum cell size for CFL stability
    const float hChar = std::min(hx, hy);
    
    // Statistics for pseudo-Δt (for diagnostic logging)
    float uDtMin = timeStep, uDtMax = 0.0f, uDtSum = 0.0f;
    float vDtMin = timeStep, vDtMax = 0.0f, vDtSum = 0.0f;
    long long uDtCount = 0;
    long long vDtCount = 0;

    // Lambda to update min/max/avg statistics for pseudo-Δt
    auto updateStats = [&](float dt, float& minDt, float& maxDt, float& sumDt, long long& count) {
        minDt = std::min(minDt, dt);
        maxDt = std::max(maxDt, dt);
        sumDt += dt;
        count++;
    };

    // -------------------------------------------------------------------------
    // Lambda: Compute local pseudo-Δt based on CFL condition
    // -------------------------------------------------------------------------
    // Δt = CFL * Δx / |u|, clamped to global maximum timeStep
    // This allows faster convergence in low-velocity regions while maintaining
    // stability in high-velocity regions.
    auto computeLocalDt = [&](float normalVel) -> float {
        if (!pseudoActive) return std::numeric_limits<float>::infinity();
        if (!useLocalPseudoTime) return timeStep;  // Use global Δt if local is disabled
        float speed = std::max(std::abs(normalVel), minPseudoSpeed);  // Prevent divide-by-zero
        float dtCfl = pseudoCFL * hChar / speed;  // CFL-based local Δt
        return std::min(timeStep, dtCfl);  // Clamp to global maximum
    };

    // -------------------------------------------------------------------------
    // Step 0: Save old velocities for residual calculation
    // -------------------------------------------------------------------------
    uOld = u;
    vOld = v;
    
    // -------------------------------------------------------------------------
    // Convection scheme setup
    // -------------------------------------------------------------------------
    const bool useSOU = (convectionScheme == 1);  // Second-order upwind?
    
    // 2.5D model: scale convection by 6/7 (accounts for parabolic profile).
    // Master switch behavior: when 2.5D is OFF, convection scaling is OFF.
    const float convScale = (enableTwoPointFiveD && enableConvectionScaling) ? (6.0f / 7.0f) : 1.0f;
    
    // 2.5D sink coefficient: -(5μ/2Ht²) * multiplier for parallel-plate friction
    // Base is (5/2)μ/Ht²; multiplier ≈4.8 gives 12μ/Ht² (Poiseuille friction)
    const float sinkCoeff = (enableTwoPointFiveD && Ht_channel > 0.0f)
                                 ? twoPointFiveDSinkMultiplier * (5.0f * eta / (2.0f * Ht_channel * Ht_channel))
                                 : 0.0f;
    const float sinkDiag = sinkCoeff * vol;  // Diagonal contribution from sink term
    
    // =========================================================================
    // STEP 1: U-MOMENTUM (Parallelized with precomputed masks)
    // =========================================================================
    // NOTE: For topology optimization with adjoint sensitivities, we MUST solve
    // ALL cells without shortcuts. The Brinkman term α·u naturally drives
    // velocity to zero in solid regions. No hard thresholds allowed - they
    // create discontinuities in ∂R/∂γ that break gradient computation.
    // =========================================================================
    
    #pragma omp parallel for reduction(max:localResidU,localTransResidU) schedule(static)
    for (int i = 1; i < M; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            
            // Domain boundary check (physical boundaries, not internal solids)
            if (checkBoundaries(i, j) == 1.0f) {
                uStar(i, j) = 0.0f;
                dE(i, j) = 0.0f;
                continue;
            }

            // Get alpha for Brinkman penalization (smooth RAMP interpolation)
            float alphaLocal = alphaAtU(*this, i, j);

            float muLocal = eta;
            if (enableResidualViscosity) {
                muLocal += muArtAtU(*this, i, j);
            }
            float De = muLocal * hy / hx;
            float Dn = muLocal * hx / hy;

            // Face velocities for flux calculation
            float ue = std::max(-maxVel, std::min(maxVel, 0.5f * (u(i, j) + u(i, j + 1))));
            float uw = std::max(-maxVel, std::min(maxVel, 0.5f * (u(i, j - 1) + u(i, j))));
            float vn = std::max(-maxVel, std::min(maxVel, 0.5f * (v(i, j) + v(i, j + 1))));
            float vs = std::max(-maxVel, std::min(maxVel, 0.5f * (v(i - 1, j) + v(i - 1, j + 1))));

            // Convective fluxes (scaled by 6/7 for 2.5D model)
            float Fe = convScale * rho * hy * ue;
            float Fw = convScale * rho * hy * uw;
            float Fn = convScale * rho * hx * vn;
            float Fs = convScale * rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            float aE = De + std::max(0.0f, -Fe);
            float aW = De + std::max(0.0f, Fw);
            float aN = Dn + std::max(0.0f, -Fn);
            float aS = Dn + std::max(0.0f, Fs);

            // Half-cell wall diffusion for tangential velocity at external
            // horizontal no-slip walls (u-momentum).
            if (externalWalls.bottom && (i - 1 == 0)) {
                aS += Dn;
            }
            if (externalWalls.top && (i + 1 == M)) {
                aN += Dn;
            }

            float Sdc = useSOU ? computeSOUCorrectionU(*this, i, j, Fe, Fw, Fn, Fs) : 0.0f;

            float sumA = aE + aW + aN + aS;
            float dtLocal = computeLocalDt(u(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, uDtMin, uDtMax, uDtSum, uDtCount);
            }
            float transCoeffLocal = rho * vol / dtLocal;
            
            // Brinkman penalization: alpha * vol adds drag in porous/solid regions
            // alphaLocal already computed above for fast-path check
            float brinkmanDrag = alphaLocal * vol;

            // Full diagonal for momentum solve (includes scaled convection + body forces)
            float aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal + sinkDiag + brinkmanDrag;
            aP0 = std::max(aP0, 1e-10f);

            float dDenom = std::max(aP0, 1e-12f);
            dE(i, j) = hy / dDenom;

            float Sp = (p(i, j) - p(i, j + 1)) * hy;
            Sp += transCoeffLocal * uOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            float aP = aP0 / uvAlpha;
            Sp += (1.0f - uvAlpha) / uvAlpha * aP0 * u(i, j);

            float uNew = (aE * u(i, j + 1) + aW * u(i, j - 1) +
                          aN * u(i + 1, j) + aS * u(i - 1, j) + Sp) / aP;

            uNew = std::max(-maxVel, std::min(maxVel, uNew));

            float diff = std::abs(uNew - u(i, j));
            if (diff > localResidU) localResidU = diff;
            
            // Track transient term residual: |transCoeff * (uNew - u)|
            // At true steady state, this should be ~0
            // This measures how much the pseudo-transient term is affecting the solution
            float transTermU = transCoeffLocal * diff;
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
    
    #pragma omp parallel for reduction(max:localResidV,localTransResidV) schedule(static)
    for (int i = 1; i < M - 1; ++i) {
        for (int j = 1; j < N; ++j) {
            
            // Domain boundary check (physical boundaries, not internal solids)
            if (checkBoundaries(i, j) == 1.0f) {
                vStar(i, j) = 0.0f;
                dN(i, j) = 0.0f;
                continue;
            }

            // Get alpha for Brinkman penalization (smooth RAMP interpolation)
            float alphaLocalV = alphaAtV(*this, i, j);

            float muLocalV = eta;
            if (enableResidualViscosity) {
                muLocalV += muArtAtV(*this, i, j);
            }
            float De = muLocalV * hy / hx;
            float Dn = muLocalV * hx / hy;

            // Face velocities for flux calculation
            float ue = std::max(-maxVel, std::min(maxVel, 0.5f * (u(i, j) + u(i + 1, j))));
            float uw = std::max(-maxVel, std::min(maxVel, 0.5f * (u(i, j - 1) + u(i + 1, j - 1))));
            float vn = std::max(-maxVel, std::min(maxVel, 0.5f * (v(i, j) + v(i + 1, j))));
            float vs = std::max(-maxVel, std::min(maxVel, 0.5f * (v(i - 1, j) + v(i, j))));

            // Convective fluxes (scaled by 6/7 for 2.5D model)
            float Fe = convScale * rho * hy * ue;
            float Fw = convScale * rho * hy * uw;
            float Fn = convScale * rho * hx * vn;
            float Fs = convScale * rho * hx * vs;

            // Base coefficients (diffusion + first-order upwind)
            float aE = De + std::max(0.0f, -Fe);
            float aW = De + std::max(0.0f, Fw);
            float aN = Dn + std::max(0.0f, -Fn);
            float aS = Dn + std::max(0.0f, Fs);

            // Half-cell wall diffusion for tangential velocity at external
            // vertical no-slip walls (v-momentum).
            if (externalWalls.left && (j - 1 == 0)) {
                aW += De;
            }
            if (externalWalls.right && (j + 1 == N)) {
                aE += De;
            }

            float Sdc = useSOU ? computeSOUCorrectionV(*this, i, j, Fe, Fw, Fn, Fs) : 0.0f;

            float sumA = aE + aW + aN + aS;
            float dtLocal = computeLocalDt(v(i, j));
            if (pseudoActive && useLocalPseudoTime && logPseudoDtStats) {
                updateStats(dtLocal, vDtMin, vDtMax, vDtSum, vDtCount);
            }
            float transCoeffLocal = rho * vol / dtLocal;
            
            // Brinkman penalization: alpha * vol adds drag in porous/solid regions
            // alphaLocalV already computed above for fast-path check
            float brinkmanDrag = alphaLocalV * vol;

            // Full diagonal for momentum solve (includes scaled convection + body forces)
            float aP0 = sumA + (Fe - Fw + Fn - Fs) + transCoeffLocal + sinkDiag + brinkmanDrag;
            aP0 = std::max(aP0, 1e-10f);

            float dDenom = std::max(aP0, 1e-12f);
            dN(i, j) = hx / dDenom;

            float Sp = (p(i, j) - p(i + 1, j)) * hx;
            Sp += transCoeffLocal * vOld(i, j);
            Sp += Sdc;  // Add second-order upwind deferred correction

            float aP = aP0 / uvAlpha;
            Sp += (1.0f - uvAlpha) / uvAlpha * aP0 * v(i, j);

            float vNew = (aE * v(i, j + 1) + aW * v(i, j - 1) +
                          aN * v(i + 1, j) + aS * v(i - 1, j) + Sp) / aP;

            vNew = std::max(-maxVel, std::min(maxVel, vNew));

            float diff = std::abs(vNew - v(i, j));
            if (diff > localResidV) localResidV = diff;
            
            // Track transient term residual: |transCoeff * (vNew - v)|
            float transTermV = transCoeffLocal * diff;
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
        pseudoStatsU.avg = uDtSum / float(uDtCount);
        pseudoStatsU.samples = uDtCount;
        pseudoStatsU.valid = true;
    } else {
        pseudoStatsU.valid = false;
    }

    if (pseudoActive && useLocalPseudoTime && logPseudoDtStats && vDtCount > 0) {
        pseudoStatsV.min = vDtMin;
        pseudoStatsV.max = vDtMax;
        pseudoStatsV.avg = vDtSum / float(vDtCount);
        pseudoStatsV.samples = vDtCount;
        pseudoStatsV.valid = true;
    } else {
        pseudoStatsV.valid = false;
    }

    }  // End of else block (explicit momentum solver)

    // -------------------------------------------------------------------------
    // Momentum residual RMS (computed for reporting/normalization)
    // -------------------------------------------------------------------------
    residU_max = residU;
    residV_max = residV;
    {
        double sumU = 0.0;
        double sumV = 0.0;
        long long countU = 0;
        long long countV = 0;

        #pragma omp parallel for reduction(+:sumU,countU) schedule(static)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                if (checkBoundaries(i, j) == 1.0f) continue;
                float diff = uStar(i, j) - uOld(i, j);
                sumU += static_cast<double>(diff) * static_cast<double>(diff);
                countU++;
            }
        }

        #pragma omp parallel for reduction(+:sumV,countV) schedule(static)
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1; j < N; ++j) {
                if (checkBoundaries(i, j) == 1.0f) continue;
                float diff = vStar(i, j) - vOld(i, j);
                sumV += static_cast<double>(diff) * static_cast<double>(diff);
                countV++;
            }
        }

        residU_RMS = (countU > 0)
            ? static_cast<float>(std::sqrt(sumU / static_cast<double>(countU)))
            : 0.0f;
        residV_RMS = (countV > 0)
            ? static_cast<float>(std::sqrt(sumV / static_cast<double>(countV)))
            : 0.0f;
    }

    // -------------------------------------------------------------------------
    // Residual-based artificial viscosity update (lagged)
    // -------------------------------------------------------------------------
    if (enableResidualViscosity) {
        updateMomentumResiduals();
    }
    updateResidualViscosity();

    // =========================================================================
    // STEP 3: PRESSURE CORRECTION (Iterative SOR or Direct Sparse Solver)
    // =========================================================================
    {
        ScopedTimer t("Iter: Pressure Correction Solve");
        bool directSolverSucceeded = solvePressureSystem(pressureIterations, localResidMass);
        (void)directSolverSucceeded;  // Suppress unused variable warning
    }
    residMass = localResidMass;

    // =========================================================================
    // STEP 4: CORRECT PRESSURE (all cells)
    // =========================================================================
    {
        ScopedTimer t("Iter: Pressure Update");
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N; ++j) {
                p(i, j) += pAlpha * pStar(i, j);
            }
        }

        // Set pressure boundary conditions (outlet reference pressure = 0.0)
        // No normalization needed - boundary condition handles reference pressure
        setPressureBoundaryConditions(p);
    }

    // =========================================================================
    // STEP 5: CORRECT VELOCITIES (Parallelized)
    // =========================================================================
    {
        ScopedTimer t("Iter: Velocity Correction");
        #pragma omp parallel for schedule(static)
        for (int i = 1; i < M; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                if (checkBoundaries(i, j) == 1.0f) continue;
                
                float du = dE(i, j) * (pStar(i, j) - pStar(i, j + 1));
                du = std::max(-maxVel, std::min(maxVel, du));
                u(i, j) = uStar(i, j) + du;
                u(i, j) = std::max(-maxVel, std::min(maxVel, u(i, j)));
            }
        }

        #pragma omp parallel for schedule(static)
        for (int i = 1; i < M - 1; ++i) {
            for (int j = 1; j < N; ++j) {
                if (checkBoundaries(i, j) == 1.0f) continue;
                
                float dv = dN(i, j) * (pStar(i, j) - pStar(i + 1, j));
                dv = std::max(-maxVel, std::min(maxVel, dv));
                v(i, j) = vStar(i, j) + dv;
                v(i, j) = std::max(-maxVel, std::min(maxVel, v(i, j)));
            }
        }

        setVelocityBoundaryConditions(u, v);
    }

    // =========================================================================
    // STEP 6: MASS BALANCE CORRECTION (Serial - small loop) [DISABLED FOR INVESTIGATION]
    // =========================================================================
    /*
    {
        ScopedTimer t("Iter: Mass Balance Correction");
        float massIn = 0.0f, massOut = 0.0f;
        for (int i = 1; i < M; ++i) {
            massIn += rho * u(i, 0) * hy;
            massOut += rho * u(i, N - 1) * hy;
        }

        if (std::abs(massOut) > 1e-10f && std::abs(massIn) > 1e-10f) {
            float ratio = massIn / massOut;
            if (ratio > 0.5f && ratio < 2.0f) {
                for (int i = 1; i < M; ++i) {
                    u(i, N - 1) *= ratio;
                }
            }
        }
    }
    */

    // -------------------------------------------------------------------------
    // Residual reporting/normalization (does NOT affect local stabilization)
    // -------------------------------------------------------------------------
    const float velFloor = std::max(residNormFloorVel, 1e-30f);
    const float massFloor = std::max(residNormFloorMass, 1e-30f);

    if (!residU_RMS0_set && residU_RMS > velFloor) {
        residU_RMS0 = residU_RMS;
        residU_RMS0_set = true;
    }
    if (!residV_RMS0_set && residV_RMS > velFloor) {
        residV_RMS0 = residV_RMS;
        residV_RMS0_set = true;
    }
    if (!residMass_RMS0_set && residMass_RMS > massFloor) {
        residMass_RMS0 = residMass_RMS;
        residMass_RMS0_set = true;
    }

    if (!residU_RMS0_set) residU_RMS0 = velFloor;
    if (!residV_RMS0_set) residV_RMS0 = velFloor;
    if (!residMass_RMS0_set) residMass_RMS0 = massFloor;

    if (enableNormalizedResiduals) {
        residU = residU_RMS / residU_RMS0;
        residV = residV_RMS / residV_RMS0;
        residMass = residMass_RMS / residMass_RMS0;
    } else {
        residU = residU_RMS;
        residV = residV_RMS;
        residMass = residMass_RMS;
    }

    return std::max({residU, residV, residMass});
}


