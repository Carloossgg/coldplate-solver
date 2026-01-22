// ============================================================================
// File: Utilities/convection.cpp
// Author: Peter Tcherkezian
// ============================================================================
// Description:
//   Implements Second-Order Upwind (SOU) deferred correction for improved
//   accuracy in convective flux discretization.
//
// DEFERRED CORRECTION APPROACH:
//   The basic idea is to solve the First-Order Upwind (FOU) system for stability
//   (it's diagonally dominant), but add a correction term to improve accuracy:
//
//   a_P * φ_P = Σ(a_nb * φ_nb) + S_dc
//
//   where S_dc = Σ_faces [ F * (φ_SOU - φ_FOU) ]
//
//   This means:
//   - Matrix coefficients use FOU values (for stability and diagonal dominance)
//   - Source term includes the difference between SOU and FOU face values
//   - As iterations progress, the correction term converges to the SOU solution
//
// SECOND-ORDER UPWIND FORMULA:
//   For a face with flow from left to right (F > 0):
//     φ_face_FOU = φ_left                    (first-order: upwind value)
//     φ_face_SOU = 1.5*φ_left - 0.5*φ_leftleft  (second-order: linear extrapolation)
//
//   The correction is: S_dc -= F * (φ_SOU - φ_FOU)
//
// FLOW DIRECTION HANDLING:
//   - Each face (E, W, N, S) checks the sign of the mass flux F
//   - Positive F means flow in the positive coordinate direction
//   - The upwind stencil is selected accordingly
//   - Near boundaries, falls back to FOU (no correction) if stencil unavailable
//
// ============================================================================
#include "SIMPLE.h"
#include <algorithm>

// ============================================================================
// computeSOUCorrectionU: Deferred correction for U-momentum
// ============================================================================
// Computes the source term correction that upgrades FOU to SOU for horizontal
// velocity component. Called during U-momentum assembly if convectionScheme == 1.
//
// Parameters:
//   s: SIMPLE solver instance (for accessing velocity field and grid info)
//   i, j: U-velocity grid location
//   Fe, Fw, Fn, Fs: Mass flux rates at east/west/north/south faces [kg/s/m]
//
// Returns:
//   S_dc contribution to add to the momentum source term
// ============================================================================
double computeSOUCorrectionU(const SIMPLE& s, int i, int j,
                             double Fe, double Fw, double Fn, double Fs) {
    if (s.convectionScheme != 1) return 0.0;  // first-order upwind
    const int M = s.M;
    const int N = s.N;
    const auto& u = s.u;
    double Sdc = 0.0;

    if (Fe >= 0.0) {
        if (j >= 2) {
            double u_sou_e = 1.5 * u(i, j) - 0.5 * u(i, j - 1);
            double u_fou_e = u(i, j);
            Sdc -= Fe * (u_sou_e - u_fou_e);
        }
    } else {
        if (j + 2 < N) {
            double u_sou_e = 1.5 * u(i, j + 1) - 0.5 * u(i, j + 2);
            double u_fou_e = u(i, j + 1);
            Sdc -= Fe * (u_sou_e - u_fou_e);
        }
    }

    if (Fw >= 0.0) {
        if (j >= 2) {
            double u_sou_w = 1.5 * u(i, j - 1) - 0.5 * u(i, j - 2);
            double u_fou_w = u(i, j - 1);
            Sdc += Fw * (u_sou_w - u_fou_w);
        }
    } else {
        if (j + 1 < N) {
            double u_sou_w = 1.5 * u(i, j) - 0.5 * u(i, j + 1);
            double u_fou_w = u(i, j);
            Sdc += Fw * (u_sou_w - u_fou_w);
        }
    }

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

    return Sdc;
}

// Deferred correction term for V-momentum (second-order upwind)
double computeSOUCorrectionV(const SIMPLE& s, int i, int j,
                             double Fe, double Fw, double Fn, double Fs) {
    if (s.convectionScheme != 1) return 0.0;  // first-order upwind
    const int M = s.M;
    const int N = s.N;
    const auto& v = s.v;
    double Sdc = 0.0;

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

    return Sdc;
}

