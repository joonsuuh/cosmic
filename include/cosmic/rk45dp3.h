#pragma once

#include <cmath>      // for sqrt, abs, pow, isnan, isinf
#include <algorithm>  // for clamp, max
#include <memory>    // for unique_ptr

class rk45_dormand_prince {
public:
    rk45_dormand_prince(int num_equations, double tolerance_abs, double tolerance_rel) 
      : N(num_equations)
      , atol(tolerance_abs)
      , rtol(tolerance_rel)
      , brightness(false)
      , k1(std::make_unique<double[]>(N))
      , k2(std::make_unique<double[]>(N))
      , k3(std::make_unique<double[]>(N))
      , k4(std::make_unique<double[]>(N))
      , k5(std::make_unique<double[]>(N))
      , k6(std::make_unique<double[]>(N))
      , k7(std::make_unique<double[]>(N))
      , y_err(std::make_unique<double[]>(N))
      , y_next(std::make_unique<double[]>(N))
    {}

    template <typename F, typename StopCond1, typename StopCond2>
    void integrate(const F &f, const StopCond1 &diskHit, const StopCond2 &noHit, 
                         std::unique_ptr<double[]>& y0) {        
        double h = 0.1;
        double err = 1.0;
        double err_prev = 1.0;

        f(y0.get(), k1.get());  // Initial k1 calculation
        
        while (!diskHit(y0.get()) && !noHit(y0.get())) {
            step(f, h, y0.get());
            err = std::max(error(), hmin);
            
            if (err < 1.0) {
                // Accept step and update state
                // Swap pointers for O(1) transfer
                y0.swap(y_next);
                k1.swap(k7);  // same as f(y.get(),k1.get()) 
                err_prev = err;
                
            }
            if (err_prev < 1.0) {
                // PI controller step size adjustment for good step size
                h = h * S * std::pow(err, -alpha) * std::pow(err_prev, beta);
                err_prev = err;
            } else {
                // Reject step and reduce step size
                h = std::min(h * S / std::pow(err, 0.2), h);
            }
            h = std::clamp(h, hmin, hmax);
        
        }
        
        brightness = diskHit(y0.get());
        // return y;
    }

    template <typename F>
    void step(const F& f, double h, const double* y) {
        // step 2 (step 1 is already done outside)
        for(int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * a21 * k1[i];
        }
        f(y_next.get(), k2.get());
        
        // step 3
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        }
        f(y_next.get(), k3.get());
        
        //step 4
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        f(y_next.get(), k4.get());

        // step 5
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        f(y_next.get(), k5.get());

        // step 6
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        f(y_next.get(), k6.get());

        // step7 finally calculate y_next
        for(int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + 
                                   a75 * k5[i] + a76 * k6[i]);
        }
        // Then use y_next to calculate k7
        f(y_next.get(), k7.get());

        // find the error
        for(int i = 0; i < N; i++) {
            y_err[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + 
                            e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        }
    }

    inline double error() {
        double err_sum = 0.0;
        for (int i = 0; i < N; i++) {
            double e = y_err[i] / (atol + std::max(std::abs(y_next[i]), std::abs(y_next[i])) * rtol);
            err_sum += e * e;
        }
        return std::sqrt(err_sum / N);
    }

    inline bool get_brightness() const {
        return brightness;
    }
private:
    // initial stuff
    const int N;
    double atol, rtol;
    bool brightness;

    // Replace raw pointers with unique_ptr
    std::unique_ptr<double[]> k1, k2, k3, k4, k5, k6, k7;
    std::unique_ptr<double[]> y_err, y_next;  // Removed y_tmp

    // clamp values
    static constexpr double hmin = 1.0e-10;
    static constexpr double hmax = 1.0;

    // Adaptive step size constants
    static constexpr double S = 0.9;  // Safety factor
    static constexpr double alpha = 0.7/5.0;  // PI controller parameters
    static constexpr double beta = 0.4/5.0;

    // Butcher tableau constants - made static to avoid recalculation
    static constexpr double c2 = 1.0 / 5.0;
    static constexpr double c3 = 3.0 / 10.0;
    static constexpr double c4 = 4.0 / 5.0;
    static constexpr double c5 = 8.0 / 9.0;

    static constexpr double a21 = 1.0 / 5.0;
    static constexpr double a31 = 3.0 / 40.0;
    static constexpr double a32 = 9.0 / 40.0;
    static constexpr double a41 = 44.0 / 45.0;
    static constexpr double a42 = -56.0 / 15.0;
    static constexpr double a43 = 32.0 / 9.0;
    static constexpr double a51 = 19372.0 / 6561.0;
    static constexpr double a52 = -25360.0 / 2187.0;
    static constexpr double a53 = 64448.0 / 6561.0;
    static constexpr double a54 = -212.0 / 729.0;
    static constexpr double a61 = 9017.0 / 3168.0;
    static constexpr double a62 = -355.0 / 33.0;
    static constexpr double a63 = 46732.0 / 5247.0;
    static constexpr double a64 = 49.0 / 176.0;
    static constexpr double a65 = -5103.0 / 18656.0;

    static constexpr double a71 = 35.0 / 384.0;
    static constexpr double a72 = 0.0;
    static constexpr double a73 = 500.0 / 1113.0;
    static constexpr double a74 = 125.0 / 192.0;
    static constexpr double a75 = -2187.0 / 6784.0;
    static constexpr double a76 = 11.0 / 84.0;

    static constexpr double e1 = 71.0 / 57600.0;
    static constexpr double e2 = 0.0;
    static constexpr double e3 = -71.0 / 16695.0;
    static constexpr double e4 = 71.0 / 1920.0;
    static constexpr double e5 = -17253.0 / 339200.0;
    static constexpr double e6 = 22.0 / 525.0;
    static constexpr double e7 = -1.0 / 40.0;
};