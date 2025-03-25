#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <memory>  // for unique_ptr
#include <atomic>  // for atomic

class rk45_dormand_prince {
public:
    enum class StopReason {
        NONE,
        CONDITION1,
        CONDITION2,
        MAX_ITERATIONS,
        ERROR
    };

    rk45_dormand_prince(int num_equations, double tolerance_abs, double tolerance_rel) 
        : N(num_equations), atol(tolerance_abs), rtol(tolerance_rel),
          // Initialize all unique_ptrs in constructor initialization list
          k1(std::make_unique<double[]>(N)),
          k2(std::make_unique<double[]>(N)),
          k3(std::make_unique<double[]>(N)),
          k4(std::make_unique<double[]>(N)),
          k5(std::make_unique<double[]>(N)),
          k6(std::make_unique<double[]>(N)),
          k7(std::make_unique<double[]>(N)),
          y_tmp(std::make_unique<double[]>(N)),
          y_err(std::make_unique<double[]>(N)),
          y_next(std::make_unique<double[]>(N))
    {
        if (N == 0) {
            throw std::invalid_argument("State size must be positive");
        }
    }

    // Add getter for stop reason
    StopReason get_stop_reason() const { return stop_reason; }

    template <typename F, typename StopCond1, typename StopCond2>
    std::unique_ptr<double[]> integrate(const F &f, const StopCond1 &stop1, const StopCond2 &stop2, 
                                        std::unique_ptr<double[]> y0)
    {        
        double h = 0.1;
        // double x = x0;
        double err = 1.0;
        double err_prev = 1.0;
        // Create working copies and result array
        auto result = std::make_unique<double[]>(N);
        auto y = std::make_unique<double[]>(N);
        
        // Ensure proper initialization of arrays
        for(size_t i = 0; i < N; i++) {
            y[i] = y0[i];
            result[i] = y0[i];
        }
        
        int iteration = 0;
        const int max_iter = 5000; // Increase max iterations
        
        try {
            f(y.get(), k1.get());
            
            while (iteration < max_iter) {
                if(stop1(y.get())) {
                    stop_reason = StopReason::CONDITION1;
                    brightness = true;
                    break;
                }
                if(stop2(y.get())) {
                    stop_reason = StopReason::CONDITION2;
                    brightness = false;
                    break;
                }

                iteration++;
                
                // Compute next step
                step(f, h, y.get());
                err = std::max(error(), hmin);
                
                const double S = 0.9;  // Safety 
                if (err < 1.0) {
                    // x += h;
                    // Careful array copying
                    for(size_t i = 0; i < N; i++) {
                        y[i] = y_next[i];
                        k1[i] = k7[i];
                    }
                    err_prev = err;
                    
                     
                    // More conservative step size adjustment
                    // h = std::max(hmin, std::min(h * 0.95 * std::pow(err, -0.2), hmax));
                    
                    // if (err_prev < 1.0) {
                    h = S * h * std::pow(err, -0.7/5.0) * std::pow(err_prev, 0.4/5.0);
                    
                } else {
                    
                // } else 
                    std::min(h, S * h * std::pow(err, -0.2));
                }
                h = std::max(h, hmin);
                    h = std::min(h, hmax);
                
                // } else {
                //     // Reduce step size more aggressively on failure
                //     h *= 0.5;
                //     if (h < hmin) {
                //         throw std::runtime_error("Step size too small");
                //     }
                // }
                // if (err_prev < o
                
            }

            if(iteration >= max_iter) {
                stop_reason = StopReason::MAX_ITERATIONS;
                std::cerr << "Warning: Maximum iterations reached\n";
            }
            
            #pragma omp critical
            {
                int current_max = max_iterations.load();
                max_iterations.store(std::max(current_max, iteration));
                total_integrations++;
            }

            // Ensure final state is properly copied
            for(size_t i = 0; i < N; i++) {
                result[i] = y[i];
            }
            
            brightness = stop1(y.get());
            
        } catch (const std::exception& e) {
            stop_reason = StopReason::ERROR;
            std::cerr << "Integration failed at step " << iteration << ": " << e.what() << std::endl;
            // On failure, return last valid state
            for(size_t i = 0; i < N; i++) {
                result[i] = y[i];
            }
        }

        if (stop_reason == StopReason::MAX_ITERATIONS || stop_reason == StopReason::ERROR) {
            std::cout << "Integration stopped due to: " 
                  << (stop_reason == StopReason::MAX_ITERATIONS ? "Max iterations" : "Error") 
                  << std::endl;
        }

        return result;
    }

    template <typename F>
    void step(const F& f, double h, const double* y) {
        // Stage 2-7 calculations
        for(size_t i = 0; i < N; i++) {
            y_tmp[i] = y[i] + h * a21 * k1[i];
        }
        f(y_tmp.get(), k2.get());
        
        for (size_t i = 0; i < N; i++) {
            y_tmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
        }
        f(y_tmp.get(), k3.get());
        
        for (size_t i = 0; i < N; i++) {
            y_tmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
        }
        f(y_tmp.get(), k4.get());

        for (size_t i = 0; i < N; i++) {
            y_tmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
        }
        f(y_tmp.get(), k5.get());

        for (size_t i = 0; i < N; i++) {
            y_tmp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
        }
        f(y_tmp.get(), k6.get());

        // First calculate y_next
        for(size_t i = 0; i < N; i++) {
            y_next[i] = y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + 
                                   a75 * k5[i] + a76 * k6[i]);
        }
        
        // Then use y_next to calculate k7
        f(y_next.get(), k7.get());

        // Update y_next with k7 contribution
        for(size_t i = 0; i < N; i++) {
            // y_next[i] += h * k7[i];
            y_err[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + 
                           e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
        }

        // Check for NaN or inf in y_next
        for(size_t i = 0; i < N; i++) {
            if (std::isnan(y_next[i]) || std::isinf(y_next[i])) {
                throw std::runtime_error("Integration step produced invalid values");
            }
        }
    }

    double error() {
        double err_sum = 0.0;
        for (size_t i = 0; i < N; i++) {  // Fix int to size_t
            double e = y_err[i] / (atol + std::max(std::abs(y_next[i]), std::abs(y_next[i])) * rtol);
            err_sum += e * e;
        }
        return std::sqrt(err_sum / N);
    }

    bool get_brightness() const {
        return brightness;
    }

    static int get_max_iterations() { return max_iterations; }
    static int get_total_integrations() { return total_integrations; }

private:
    // // Add helper method for step size calculation
    // double compute_step_factor(double err, double err_prev, double h) {
        
    // }

    const int N;  // Store size as member
    double atol, rtol;
    const double hmin = 1.0e-10;
    const double hmax = 1.0;
    bool brightness;

    // Add member variable
    StopReason stop_reason = StopReason::NONE;

    // Replace raw pointers with unique_ptr
    std::unique_ptr<double[]> k1, k2, k3, k4, k5, k6, k7;
    std::unique_ptr<double[]> y_tmp, y_err, y_next;

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

    static std::atomic<int> max_iterations;
    static std::atomic<int> total_integrations;
};

// Initialize static members
std::atomic<int> rk45_dormand_prince::max_iterations{0};
std::atomic<int> rk45_dormand_prince::total_integrations{0};