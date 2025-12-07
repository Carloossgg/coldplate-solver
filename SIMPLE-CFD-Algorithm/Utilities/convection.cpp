// File: Utilities/convection.cpp
// Author: Peter Tcherkezian
// Description: Second-order upwind (SOU) deferred correction for U and V momentum:
//   computes higher-order flux corrections added to FOU matrices (deferred) to improve accuracy while preserving
//   diagonal dominance and stability of the underlying FOU system.
#include "SIMPLE.h"
#include <algorithm>

// Deferred correction term for U-momentum (second-order upwind)
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

