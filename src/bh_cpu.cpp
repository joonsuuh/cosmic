// STL
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <utility>
#include <vector>

// External Libraries
#include <omp.h>

// Project Headers
#include "clock.h"
#include "metric.h"
#include "rk45dp3.h"

#ifndef PPM_DIR
  #define PPM_DIR "data/"
#endif

struct BlackHole {
  double a {0.99};        // spin parameter
  double M {1.0};        // mass
  double D {500.0};        // distance
  double theta0 {85.0 * M_PI / 180.0};   // initial theta
  double phi0 {0.0};     // initial phi
  double r_in {5.0 * M};     // inner radius
  double r_out {20.0 * M};    // outer radius
  double epsilon {1.0e-5};  // tolerance
  double r_H {M + std::sqrt(M * M - a * a)};
  double r_H_tol {1.01 * r_H};
  double r_far {r_out * 50.0};
};

struct Image {
  int ratiox {16};
    int ratioy {9};
    int resScale {10};
    int resx {ratiox * resScale};
    int resy {ratioy * resScale};
    double scaling {1.5};
    double x_sc {-ratiox * scaling};
    double y_sc {-ratioy * scaling};
    double stepx {std::abs(2.0 * x_sc / resx)};
    double stepy {std::abs(2.0 * y_sc / resy)};
};

// Function declaration
inline void compute_derivatives(boyer_lindquist_metric& metric, double* y, double* k);

int main() {
  std::cout << "CPU version running\n";
  // Black hole parameters
  double a = 0.99;
  double M = 1.0;
  double D = 500.0;
  double theta0 = 85.0 * M_PI / 180.0;
  double phi0 = 0.0;
  double r_in = 5.0 * M;
  double r_out = 20.0 * M;
  double epsilon = 1.0e-5;
  double r_H = M + std::sqrt(M * M - a * a);
  double r_H_tol = 1.01 * r_H;
  double r_far = r_out * 50.0;

  int ratiox = 16;
  int ratioy = 9;
  int resScale = 10;
  int resx = ratiox * resScale;
  int resy = ratioy * resScale;
  std::cout << "Resolution: " << resx << "x" << resy << std::endl;

  double SCALING = 1.5;
  double x_sc = -16.0 * SCALING;
  double y_sc = -9.0 * SCALING;
  double stepx = std::abs(2.0 * x_sc / resx);
  x_sc += 0.5 * stepx;
  double stepy = std::abs(2.0 * y_sc / resy);

  // Create a 1D vector to store the final output
  std::vector<double> final_screen(resx * resy, 0.0);
  std::cout << "Vector size in bytes: " << final_screen.size() * sizeof(double) << std::endl;

  // Get number of available threads
  int max_threads = omp_get_max_threads();
  int num_threads = max_threads; // Set desired number of threads here
  omp_set_num_threads(num_threads);
  std::cout << "Available Threads: " << max_threads << "\n"
            << "Using Threads: " << num_threads << std::endl;

  // Parallel region
  {
    Clock clock;

#pragma omp parallel
    {
      // Each thread needs its own metric instance
      boyer_lindquist_metric thread_metric(a, M);

#pragma omp for collapse(2) schedule(dynamic) nowait
      for (int i = 0; i < resx; i++) {
        for (int j = 0; j < resy; j++) {
          // Reset metric for this iteration
          // thread_metric = boyer_lindquist_metric(a, M);
          
          double local_x_sc = x_sc + (i * stepx);
          double local_y_sc = y_sc + (j * stepy);

          double beta = local_x_sc / D;
          double alpha = local_y_sc / D;
          double r = std::sqrt((D * D) + (local_x_sc * local_x_sc) +
                          (local_y_sc * local_y_sc));
          double theta = theta0 - alpha;
          double phi = beta;
          thread_metric.compute_metric(r, theta);

          auto dydx = [&thread_metric](double* y, double* k) {
            compute_derivatives(thread_metric, y, k);
          };

          double u_r = -std::sqrt(thread_metric.g_11) * std::cos(beta) * std::cos(alpha);
          double u_theta = -std::sqrt(thread_metric.g_22) * std::sin(alpha);
          double u_phi = std::sqrt(thread_metric.g_33) * std::sin(beta) * std::cos(alpha);

          // heap allocation with unique_ptr // auto is std::unique_ptr<double[]>
          auto y0 = std::make_unique<double[]>(6);
          y0[0] = r;
          y0[1] = theta;
          y0[2] = phi;
          y0[3] = u_r;
          y0[4] = u_theta;
          y0[5] = u_phi;

          // diskHit
          auto stop1 = [&r_in, &r_out](const double* y) {
              return ((y[0] >= r_in && y[0] <= r_out) &&
                      (std::abs(y[1] - M_PI / 2.0) < 0.01));
          };

          // fall into hole or really far out
          auto stop2 = [&r_H_tol, &r_far](const double* y) {
              return (y[0] < r_H_tol || y[0] > r_far);
          };

          rk45_dormand_prince rk45(6, 1.0e-12, 1.0e-12);
          rk45.integrate(dydx, stop1, stop2, y0);

          double Iobs = 0.0;
          if (rk45.get_brightness()) {
            double rf = y0[0];
            double u_rf = -y0[3];
            double u_thf = -y0[4];
            double u_phif = -y0[5];

            double u_uppertf = std::sqrt((thread_metric.gamma11 * u_rf * u_rf) +
                                    (thread_metric.gamma22 * u_thf * u_thf) +
                                    (thread_metric.gamma33 * u_phif * u_phif)) /
                               thread_metric.alpha;
            double u_lower_tf =
                (-thread_metric.alpha * thread_metric.alpha * u_uppertf) +
                (u_phif * thread_metric.beta3);
            double omega = 1.0 / (a + (std::pow(rf, 3.0 / 2.0) / std::sqrt(M)));
            double oneplusz =
                (1.0 + (omega * u_phif / u_lower_tf)) /
                std::sqrt(-thread_metric.g_00 - (omega * omega * thread_metric.g_33) -
                     (2 * omega * thread_metric.g_03));
            Iobs = 1.0 / (oneplusz * oneplusz * oneplusz);
          }

          final_screen[(resy - j - 1) * resx + i] = Iobs;

          // #pragma omp critical
          // {
          //   std::cout << "Thread " << omp_get_thread_num() << " completed pixel (" << i << "," << j << ")\n";
          // }
        }
      }
    }
  } // Print EXECUTION TIME

  // NORMALIZE IMAGE DATA
  double max_intensity = 0.0;
  // Update array access in normalization
  for (auto &intensity : final_screen) {
    max_intensity = std::max(max_intensity, intensity);
  }
  
  for (auto &intensity : final_screen) {
    intensity /= max_intensity;
  }

  // APPLY HOT colormap from matlab/matplotlib
  // https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py
  // _hot_data = {'red':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),
            //  'green': ((0., 0., 0.),(0.365079, 0.000000, 0.000000),
            //            (0.746032, 1.000000, 1.000000),(1.0, 1.0, 1.0)),
            //  'blue':  ((0., 0., 0.),(0.746032, 0.000000, 0.000000),(1.0, 1.0, 1.0))} 

  // Write the final_screen to a ppm file
  std::ofstream output_file(PPM_DIR "bh_array.ppm");
  output_file << "P3\n";
  output_file << resx << " " << resy << "\n";
  output_file << "255\n";

  for (int i = 0; i < resy; i++) {
    for (int j = 0; j < resx; j++) {
      double value = final_screen[i * resx + j];
      int r, g, b;
      
      // Implement hot colormap
      if (value < 0.365079) {
        // Black to red
        r = static_cast<int>((value / 0.365079) * 255);
        g = 0;
        b = 0;
      } else if (value < 0.746032) {
        // Red to yellow
        r = 255;
        g = static_cast<int>(((value - 0.365079) / (0.746032 - 0.365079)) * 255);
        b = 0;
      } else {
        // Yellow to white
        r = 255;
        g = 255;
        b = static_cast<int>(((value - 0.746032) / (1.0 - 0.746032)) * 255);
      }
      
      output_file << r << " " << g << " " << b << " ";
    }
    output_file << "\n";
  }
  output_file.close();

  return 0;
}

// Function implementation
inline void compute_derivatives(boyer_lindquist_metric& metric, double* y, double* k) {
    metric.compute_metric(y[0], y[1]);
    double r = y[0];
    double th = y[1];
    double phi = y[2];
    double u_r = y[3];
    double u_th = y[4];
    double u_phi = y[5];

    double u_uppert = std::sqrt((metric.gamma11 * u_r * u_r) +
                               (metric.gamma22 * u_th * u_th) +
                               (metric.gamma33 * u_phi * u_phi)) /
                      metric.alpha;

    double drdt = metric.gamma11 * u_r / u_uppert;
    double dthdt = metric.gamma22 * u_th / u_uppert;
    double dphidt =
        (metric.gamma33 * u_phi / u_uppert) - metric.beta3;

    double temp1 = (u_r * u_r * metric.d_gamma11_dr) +
                   (u_th * u_th * metric.d_gamma22_dr) +
                   (u_phi * u_phi * metric.d_gamma33_dr);
    double durdt =
        (-metric.alpha * u_uppert * metric.d_alpha_dr) +
        (u_phi * metric.d_beta3_dr) - (temp1 / (2.0 * u_uppert));

    double temp2 = (u_r * u_r * metric.d_gamma11_dth) +
                   (u_th * u_th * metric.d_gamma22_dth) +
                   (u_phi * u_phi * metric.d_gamma33_dth);
    double duthdt =
        (-metric.alpha * u_uppert * metric.d_alpha_dth) +
        (u_phi * metric.d_beta3_dth) - temp2 / (2.0 * u_uppert);
    double duphidt = 0;

    k[0] = drdt;
    k[1] = dthdt;
    k[2] = dphidt;
    k[3] = durdt;
    k[4] = duthdt;
    k[5] = duphidt;
}
