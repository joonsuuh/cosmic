#pragma once

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <atomic>

class rk45_dormand_prince {
public:
  rk45_dormand_prince(int num_equations, double tolerance_abs, double tolerance_rel) 
    : n_eq(num_equations), atol(tolerance_abs), rtol(tolerance_rel)
  {
    // Pre-allocate all vectors once in constructor
    y_tmp.resize(n_eq);
    y_err.resize(n_eq);
    y_next.resize(n_eq);

    k1.resize(n_eq);
    k2.resize(n_eq);
    k3.resize(n_eq);
    k4.resize(n_eq);
    k5.resize(n_eq);
    k6.resize(n_eq);
    k7.resize(n_eq);
  }

  template <typename F, typename StopCond1, typename StopCond2>
  std::vector<double> integrate(const F &f, const StopCond1 &stop1, const StopCond2 &stop2, double x0, 
                                const std::vector<double> &y0)
  {
    result.clear();

    double h = 0.1;
    double x = x0;
    std::copy(y0.begin(), y0.end(), y_next.begin());
    std::vector<double>& y = y_next;
    
    double err = 1.0;
    double err_prev = 1.0;
    // k1 = f(x, y);
    k1 = f(y);

    // Store initial conditions 
    result.push_back(y);

    try {
      int iteration = 0;
      const int max_iterations = 5000;

      while (!stop1(x, y) && !stop2(x, y)) {
        if (iteration++ > max_iterations) {
          throw std::runtime_error("Max iterations exceeded");
        }
        step(f, h, x, y);
        err = std::max(error(), 1.0e-10);

        if (err < 1.0) {
            x += h;
            // hs.push_back(h);
            // xs.push_back(x);
            result.push_back(y_next);
            
            std::swap(k1, k7);
            err_prev = err;
            std::swap(y, y_next);

            // Step size control
            double S = 0.9;
            double h_factor;
            if (err_prev < 1.0) {
                h_factor = S * std::pow(err, -0.7/5.0) * std::pow(err_prev, 0.4/5.0);
            } else {
                h_factor = S * std::pow(err, -0.2);
            }
            h *= h_factor;
            h = std::max(hmin, std::min(h, hmax));
        }
      }

      #pragma omp critical
      {
        max_iterations_count = std::max(max_iterations_count.load(), iteration);
        total_integrations++;
      }

      // print iteration count
      std::cout << "Iterations: " << iteration << "\n";
    } catch (const std::exception& e) {
      std::cerr << "Caught exception: " << e.what() << std::endl;
    }
    
    
    
    if (stop1(x,y)) {
      brightness = true;
    } else {
      brightness = false;
    }

    y.clear();
    return y_next;
  }

  template <typename F>
  void step(const F& f, double h, double x, const std::vector<double> &y) {
    // Stage 1: k1 is already computed in the calling function

    // Stage 2
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * a21 * k1[i];
    }
    // k2 = f(x + c2 * h, y_tmp);
    k2 = f(y_tmp);
    
    // Stage 3
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a31 * k1[i] + a32 * k2[i]);
    }
    // k3 = f(x + c3 * h, y_tmp);
    k3 = f(y_tmp);
    
    // Stage 4
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a41 * k1[i] + a42 * k2[i] + a43 * k3[i]);
    }
    // k4 = f(x + c4 * h, y_tmp);
    k4 = f(y_tmp);
    
    // Stage 5
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a51 * k1[i] + a52 * k2[i] + a53 * k3[i] + a54 * k4[i]);
    }
    // k5 = f(x + c5 * h, y_tmp);
    k5 = f(y_tmp);
    
    // Stage 6
    for (int i = 0; i < n_eq; i++) {
      y_tmp[i] = y[i] + h * (a61 * k1[i] + a62 * k2[i] + a63 * k3[i] + a64 * k4[i] + a65 * k5[i]);
    }
    // k6 = f(x + h, y_tmp);
    k6 = f(y_tmp);
    
    // Stage 7 (final result)
    for (int i = 0; i < n_eq; i++) {
      y_next[i] = y[i] + h * (a71 * k1[i] + a73 * k3[i] + a74 * k4[i] + a75 * k5[i] + a76 * k6[i]);
  
    }
    
    // Compute k7 for next step and error estimation
    // k7 = f(x + h, y_next);
    k7 = f(y_next);
    
    // Complete error calculation
    for (int i = 0; i < n_eq; i++) {
      y_err[i] = h * (e1 * k1[i] + e3 * k3[i] + e4 * k4[i] + e5 * k5[i] + e6 * k6[i] + e7 * k7[i]);
    }
  }

  double error() {
    double err_sum = 0.0;
    for (int i = 0; i < n_eq; i++) {
      double scale = atol + std::max(std::abs(y_next[i]), std::abs(y_next[i])) * rtol;
      double e = y_err[i] / scale;
      err_sum += e * e;
    }
    return std::sqrt(err_sum / n_eq);
  }

  // Add static methods to get counters
  static int get_max_iterations() { return max_iterations_count; }
  static int get_total_integrations() { return total_integrations; }

  // Class members with better organization
  const int n_eq;
  const double atol, rtol;

  // Minimum and maximum step size
  const double hmin = 1.0e-10;
  const double hmax = 1.0;

  // Temporary vectors used during computation
  std::vector<double> k1, k2, k3, k4, k5, k6, k7;
  std::vector<double> y_tmp, y_err, y_next;

  // Result storage
  // std::vector<double> hs;
  // std::vector<double> xs;
  bool brightness;
  std::vector<std::vector<double>> result;

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

  // Add static atomic counters
  static std::atomic<int> max_iterations_count;
  static std::atomic<int> total_integrations;
};

// Initialize static members
std::atomic<int> rk45_dormand_prince::max_iterations_count{0};
std::atomic<int> rk45_dormand_prince::total_integrations{0};