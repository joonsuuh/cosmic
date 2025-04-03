#ifndef COSMIC_DORMAND_PRINCE_H
#define COSMIC_DORMAND_PRINCE_H

#include <cmath>      // for sqrt, abs, pow
#include <cstring>    // for memcpy

// Modern C++ approach using template interfaces
class DormandPrinceRK45 {
public:
    DormandPrinceRK45(int num_equations, double tolerance_abs, double tolerance_rel) 
      : N(num_equations)
      , atol(tolerance_abs)
      , rtol(tolerance_rel)
    {
      // Raw pointer allocation
      k1 = new double[static_cast<size_t>(N)]{};
      k2 = new double[static_cast<size_t>(N)]{};
      k3 = new double[static_cast<size_t>(N)]{};
      k4 = new double[static_cast<size_t>(N)]{};
      k5 = new double[static_cast<size_t>(N)]{};
      k6 = new double[static_cast<size_t>(N)]{};
      k7 = new double[static_cast<size_t>(N)]{};
      y_err = new double[static_cast<size_t>(N)]{};
      y_next = new double[static_cast<size_t>(N)]{};
    }

    // Destructor to clean up allocated memory
    ~DormandPrinceRK45() {
      delete[] k1;
      delete[] k2;
      delete[] k3;
      delete[] k4;
      delete[] k5;
      delete[] k6;
      delete[] k7;
      delete[] y_err;
      delete[] y_next;
    }

    // Disable copy construction/assignment (Rule of 5)
    DormandPrinceRK45(const DormandPrinceRK45&) = delete;
    DormandPrinceRK45& operator=(const DormandPrinceRK45&) = delete;
    DormandPrinceRK45(DormandPrinceRK45&&) = delete;
    DormandPrinceRK45& operator=(DormandPrinceRK45&&) = delete;

    // Custom inline functions for performance
    inline double max_val(double a, double b) {
        return a > b ? a : b;
    }
    
    inline double clamp_val(double value, double min_val, double max_val) {
        return value < min_val ? min_val : (value > max_val ? max_val : value);
    }

    inline double min_val(double a, double b) {
        return a < b ? a : b;
    }

    // 
    // integrates ODEs and checks for hit on the disk
    // returns if hit detected
    template<typename DerivativeFunctor, typename HitFunctor, typename BoundaryFunctor>
    bool integrate(DerivativeFunctor& derivFunctor, 
                  HitFunctor& hitFunctor,
                  BoundaryFunctor& boundaryFunctor,
                  double* y) {
        double h = 0.1;
        double err = 1.0;
        double err_prev = 1.0;
        // bool hit_detected = false; // Local variable instead of class member

        // First same as last (FSAL) so k1 <- k7 
        derivFunctor(y, k1);
        
        while (!boundaryFunctor(y)) {
            step(derivFunctor, h, y);
            err = max_val(error(), hmin);
            
            if (err < 1.0) {
                // Direct loop is more efficient than std::memcpy for small arrays
                for (int i = 0; i < N; ++i) {
                    y[i] = y_next[i];
                }
                
                // raw pointer swap for k1 <- k7
                double* temp = k1;
                k1 = k7;
                k7 = temp;

                // for PI_controller
                err_prev = err;
                
                // Check if we hit the disk after this step
                if (hitFunctor(y)) {
                    return true; // Return immediately if hit detected
                }
            }
            if (err_prev < 1.0) {
                // PI controller step size adjustment for good step size
                h = h * S * std::pow(err, -alpha) * std::pow(err_prev, beta);
            } else {
                // Reject step and reduce step size
                h = min_val(h * S / std::pow(err, 0.2), h);
            }
            h = clamp_val(h, hmin, hmax);
        }
        
        return false; // Return false if no hit detected
    }

    template <typename DerivativeFunction>
    void step(DerivativeFunction& derivFunc, double h, const double* y) {  // Removed const from h
        // step 2 (step 1 is already done outside)
        for(int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * a21 * k1[i];
        }
        derivFunc(y_next, k2);
        
        // step 3
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        }
        derivFunc(y_next, k3);
        
        // step 4
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        derivFunc(y_next, k4);

        // step 5
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        derivFunc(y_next, k5);

        // step 6
        for (int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        derivFunc(y_next, k6);

        // step 7 finally calculate y_next
        for(int i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + 
                                   a75 * k5[i] + a76 * k6[i]);
        }
        // Then use y_next to calculate k7
        derivFunc(y_next, k7);

        // find the error
        for(int i = 0; i < N; i++) {
            y_err[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + 
                            e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        }
    }

    inline double error() {
        double err_sum = 0.0;
        for (int i = 0; i < N; i++) {
            double e = y_err[i] / (atol + max_val(std::abs(y_next[i]), std::abs(y_next[i])) * rtol);
            err_sum += e * e;
        }
        return std::sqrt(err_sum / N);
    }

private:
    // initial stuff
    const int N;
    const double atol, rtol;

    // dynamic array allocation 
    double* k1{}; // same as k1{nullptr} for safety
    double* k2{};
    double* k3{};
    double* k4{};
    double* k5{};
    double* k6{};
    double* k7{};
    double* y_err{};
    double* y_next{};

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

#endif // COSMIC_DORMAND_PRINCE_H