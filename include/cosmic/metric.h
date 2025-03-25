#include <vector>
#include <cmath>
#include <iostream>

class boyer_lindquist_metric {
    public:
        boyer_lindquist_metric(double a0, double M0){
            a = a0;
            M = M0;
            
        }

        void compute_metric(double r, double th){
            // Define General Terms
            double r2 = r * r;
            double a2 = a * a;
            double sin2 = std::sin(th) * std::sin(th);
            double cos2 = std::cos(th) * std::cos(th);
            double p2 = r2 + (a2 * cos2);
            double delta = r2 + a2 - (2.0 * M * r);
            double SUM = ((r2 + a2) * (r2 + a2)) - (a2 * delta * sin2);

            alpha = std::sqrt((p2 * delta) / SUM);
            beta3 = (-2.0 * M * a * r) / SUM;
            gamma11 = delta / p2;
            gamma22 = 1.0 / p2;
            gamma33 = p2 / (SUM * std::sin(th) * std::sin(th));
            g_00 = ((2.0 * M * r) / p2) - 1.0;
            g_03 = ((-2.0 * M * a * r) / p2) * std::sin(th) * std::sin(th);
            g_11 = p2 / delta;
            g_22 = p2;
            g_33 = (SUM / p2) * std::sin(th) * std::sin(th);

            // Derivatives w.r.t r
            double dp2_dr = 2.0 * r;
            double dp2inv_dr = -2.0 * r / p2 / p2;
            double ddelta_dr = dp2_dr - (2.0 * M);
            double dSUMinv_dr =  ((4.0 * r * (r2 + a2)) - (ddelta_dr * a2 * sin2)) / -SUM / SUM;
            d_gamma11_dr = (delta * dp2inv_dr) + (ddelta_dr / p2);
            d_gamma22_dr = dp2inv_dr;
            d_gamma33_dr = (dp2_dr / (SUM * sin2)) + (dSUMinv_dr * p2 / sin2);
            d_alpha_dr = (0.5 / alpha) * ((dp2_dr * delta / SUM) + (ddelta_dr * p2 / SUM) + (dSUMinv_dr * p2 * delta));
            d_beta3_dr = -2.0 * M * a * (1.0 / SUM + r * dSUMinv_dr);

            // Derivatives w.r.t theta
            double dp2_dth = -2.0 * a * a * std::sin(th) * std::cos(th);
            double dp2inv_dth = -dp2_dth / p2 / p2;
            double ddelta_dth = 0;
            double dSUMinv_dth = 2.0 * a2 * delta * std::sin(th) * std::cos(th) / SUM / SUM;
            d_gamma11_dth = delta * dp2inv_dth;
            d_gamma22_dth = dp2inv_dth;
            d_gamma33_dth = (dp2_dth / SUM / sin2) + (dSUMinv_dth * p2 / sin2) - (2.0 * std::cos(th) / std::sin(th) / std::sin(th) / std::sin(th) * p2 / SUM);
            d_alpha_dth = 0.5 / alpha * delta * (dp2_dth / SUM + dSUMinv_dth * p2);
            d_beta3_dth = -2.0 * M * a * r * dSUMinv_dth;
        }

        double a, M;
        double alpha, beta3;
        double gamma11, gamma22, gamma33; // components of upper gamma^ij
        double g_00, g_03, g_11, g_22, g_33; // components of lower g_\mu\nu
        double d_alpha_dr, d_beta3_dr, d_gamma11_dr, d_gamma22_dr, d_gamma33_dr;
        double d_alpha_dth, d_beta3_dth, d_gamma11_dth, d_gamma22_dth, d_gamma33_dth;
};