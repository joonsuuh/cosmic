#ifndef COSMIC_METRIC_H
#define COSMIC_METRIC_H

#include <cmath>

class BoyerLindquistMetric {
public:
    // Black hole parameters
    const double a; // Spin
    const double M; // Mass
    
    // Metric components
    double alpha, beta3;
    double gamma11, gamma22, gamma33; // components of upper gamma^ij
    double g_00, g_03, g_11, g_22, g_33; // components of lower g_\mu\nu
    
    // Metric derivatives
    double d_alpha_dr, d_beta3_dr, d_gamma11_dr, d_gamma22_dr, d_gamma33_dr;
    double d_alpha_dth, d_beta3_dth, d_gamma11_dth, d_gamma22_dth, d_gamma33_dth;

    BoyerLindquistMetric(double spin, double mass) : a(spin), M(mass) {}

    inline void computeMetric(double r, double theta) {
        // Define General Terms
        const double r2 = r * r;
        const double a2 = a * a;
        const double sin2 = std::sin(theta) * std::sin(theta);
        const double cos2 = std::cos(theta) * std::cos(theta);
        const double p2 = r2 + (a2 * cos2);
        const double delta = r2 + a2 - (2.0 * M * r);
        const double sigma = ((r2 + a2) * (r2 + a2)) - (a2 * delta * sin2);

        // Compute metric components
        computeMetricComponents(r, r2, a2, sin2, cos2, p2, delta, sigma);
        
        // Compute metric derivatives
        computeMetricDerivatives(r, r2, a2, sin2, cos2, p2, delta, sigma, theta);
    }

    inline void computeDerivatives(double* y, double* k) {
        computeMetric(y[0], y[1]);
        
        const double u_uppert = std::sqrt((gamma11 * y[3] * y[3]) +
                                        (gamma22 * y[4] * y[4]) +
                                        (gamma33 * y[5] * y[5])) /
                               alpha;
    
        k[0] = gamma11 * y[3] / u_uppert;
        k[1] = gamma22 * y[4] / u_uppert;
        k[2] = (gamma33 * y[5] / u_uppert) - beta3;
    
        const double temp1 = (y[3] * y[3] * d_gamma11_dr) +
                           (y[4] * y[4] * d_gamma22_dr) +
                           (y[5] * y[5] * d_gamma33_dr);
                            
        k[3] = (-alpha * u_uppert * d_alpha_dr) +
               (y[5] * d_beta3_dr) - (temp1 / (2.0 * u_uppert));
    
        const double temp2 = (y[3] * y[3] * d_gamma11_dth) +
                           (y[4] * y[4] * d_gamma22_dth) +
                           (y[5] * y[5] * d_gamma33_dth);
                            
        k[4] = (-alpha * u_uppert * d_alpha_dth) +
               (y[5] * d_beta3_dth) - temp2 / (2.0 * u_uppert);
    
        k[5] = 0.0;
    }

private:
    // Extract computation of metric components for better organization
    inline void computeMetricComponents(double r, double r2, double a2, double sin2, double cos2, 
                                double p2, double delta, double SUM) {
        alpha = std::sqrt((p2 * delta) / SUM);
        beta3 = (-2.0 * M * a * r) / SUM;
        gamma11 = delta / p2;
        gamma22 = 1.0 / p2;
        gamma33 = p2 / (SUM * sin2);
        g_00 = ((2.0 * M * r) / p2) - 1.0;
        g_03 = ((-2.0 * M * a * r) / p2) * sin2;
        g_11 = p2 / delta;
        g_22 = p2;
        g_33 = (SUM / p2) * sin2;
    }

    // Extract computation of derivatives for better organization
    inline void computeMetricDerivatives(double r, double r2, double a2, double sin2, double cos2, 
                                 double p2, double delta, double sigma, double theta) {
        // Derivatives w.r.t r
        const double dp2_dr = 2.0 * r;
        const double dp2inv_dr = -2.0 * r / p2 / p2;
        const double ddelta_dr = dp2_dr - (2.0 * M);
        const double dSUMinv_dr = ((4.0 * r * (r2 + a2)) - (ddelta_dr * a2 * sin2)) / -sigma / sigma;
        
        d_gamma11_dr = (delta * dp2inv_dr) + (ddelta_dr / p2);
        d_gamma22_dr = dp2inv_dr;
        d_gamma33_dr = (dp2_dr / (sigma * sin2)) + (dSUMinv_dr * p2 / sin2);
        d_alpha_dr = (0.5 / alpha) * ((dp2_dr * delta / sigma) + (ddelta_dr * p2 / sigma) + (dSUMinv_dr * p2 * delta));
        d_beta3_dr = -2.0 * M * a * (1.0 / sigma + r * dSUMinv_dr);

        // Derivatives w.r.t theta
        const double dp2_dth = -2.0 * a2 * std::sin(theta) * std::cos(theta);
        const double dp2inv_dth = -dp2_dth / p2 / p2;
        const double ddelta_dth = 0;
        const double dSUMinv_dth = 2.0 * a2 * delta * std::sin(theta) * std::cos(theta) / sigma / sigma;
        
        d_gamma11_dth = delta * dp2inv_dth;
        d_gamma22_dth = dp2inv_dth;
        d_gamma33_dth = (dp2_dth / sigma / sin2) + (dSUMinv_dth * p2 / sin2) - 
                       (2.0 * std::cos(theta) / std::sin(theta) / sin2 * p2 / sigma);
        d_alpha_dth = 0.5 / alpha * delta * (dp2_dth / sigma + dSUMinv_dth * p2);
        d_beta3_dth = -2.0 * M * a * r * dSUMinv_dth;
    }
};

#endif // COSMIC_METRIC_H