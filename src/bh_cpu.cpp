#include "metric.h"
#include "rk45_dp.h"
#include "clock.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include <chrono>
#include <omp.h> // Add OpenMP header
using namespace std;

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
    double r_H = M + sqrt(M * M - a * a);
    double r_H_tol = 1.01 * r_H;
    double r_far = r_out * 50.0;
    boyer_lindquist_metric metric(a, M);

    int ratiox = 16;
    int ratioy = 9;
    int resScale = 120;
    int resx = ratiox * resScale;
    int resy = ratioy * resScale;
    std::cout << "Resolution: " << resx << "x" << resy << endl;

    double x_sc = -25.0;
    double y_sc = -12.5;  
    double y_sc0 = y_sc;
    double stepx = abs(2.0 * x_sc / resx);
    x_sc += 0.5 * stepx;
    double stepy = abs(2.0 * y_sc / resy);

    // Create a 2D vector to store the final output
    vector<vector<double>> final_screen(resy, vector<double>(resx, 0.0));

    // Get number of available threads
    int max_threads = omp_get_max_threads();
    int num_threads = 8; // Set desired number of threads here
    omp_set_num_threads(num_threads);
    cout << "Available Threads: " << max_threads << endl;
    cout << "Using Threads: " << num_threads << endl;

    // Parallel region
    {
        Clock clock;
        
        #pragma omp parallel
        {
            // Each thread needs its own metric instance to avoid race conditions
            boyer_lindquist_metric local_metric(a, M);

            // Parallelize the outer loop
            // collapse(2) schedule(dynamic) nowait: optional; is used for optimization 
            #pragma omp for collapse(2) schedule(dynamic) nowait
            for (int i = 0; i < resx; i++) {
                for (int j = 0; j < resy; j++) {
                    double local_x_sc = x_sc + (i * stepx);
                    double local_y_sc = y_sc0 + (j * stepy);

                    double beta = local_x_sc / D;
                    double alpha = local_y_sc / D;
                    double r = sqrt((D * D) + (local_x_sc * local_x_sc) + (local_y_sc * local_y_sc));
                    double theta = theta0 - alpha;
                    double phi = beta;
                    local_metric.compute_metric(r, theta);

                    auto dydx = [&] (double x, const vector<double> &y) {
                        local_metric.compute_metric(y[0], y[1]);
                        double r = y[0];
                        double th = y[1];
                        double phi = y[2];
                        double u_r = y[3];
                        double u_th = y[4];
                        double u_phi = y[5]; 

                        double u_uppert = sqrt((local_metric.gamma11 * u_r * u_r) + 
                                            (local_metric.gamma22 * u_th * u_th) + 
                                            (local_metric.gamma33 * u_phi * u_phi)) / local_metric.alpha;
                        
                        double drdt = local_metric.gamma11 * u_r / u_uppert;
                        double dthdt = local_metric.gamma22 * u_th / u_uppert;
                        double dphidt = (local_metric.gamma33 * u_phi / u_uppert) - local_metric.beta3;

                        double temp1 = (u_r * u_r * local_metric.d_gamma11_dr) + 
                                    (u_th * u_th * local_metric.d_gamma22_dr) + 
                                    (u_phi * u_phi * local_metric.d_gamma33_dr);
                        double durdt = (-local_metric.alpha * u_uppert * local_metric.d_alpha_dr) + 
                                    (u_phi * local_metric.d_beta3_dr) - (temp1 / (2.0 * u_uppert));

                        double temp2 = (u_r * u_r * local_metric.d_gamma11_dth) + 
                                    (u_th * u_th * local_metric.d_gamma22_dth) + 
                                    (u_phi * u_phi * local_metric.d_gamma33_dth);
                        double duthdt = (-local_metric.alpha * u_uppert * local_metric.d_alpha_dth) + 
                                    (u_phi * local_metric.d_beta3_dth) - temp2 / (2.0 * u_uppert);
                        double duphidt = 0;

                        return vector<double>{drdt, dthdt, dphidt, durdt, duthdt, duphidt};
                    };

                    double u_r = -sqrt(local_metric.g_11) * cos(beta) * cos(alpha);
                    double u_theta = -sqrt(local_metric.g_22) * sin(alpha);
                    double u_phi = sqrt(local_metric.g_33) * sin(beta) * cos(alpha);

                    vector<double> y0 = {r, theta, phi, u_r, u_theta, u_phi};

                    int n = 10'000;
                    double t0 = 0.0;
                    double t_end = 10'000;
                    double dt = (t_end - t0) / n;
                    vector<double> t_out(n);
                    for (int k = 0; k < n; k++) {
                        t_out[k] = t0 + k * dt;
                    }

                    auto stop1 = [&] (double x, const vector<double> &y) {
                        double r = y[0];
                        double theta = y[1];
                        return ((r >= r_in && r <= r_out) && (abs(theta - M_PI / 2.0) < 0.01));
                    };

                    auto stop2 = [&] (double x, const vector<double> &y) {
                        double r = y[0];
                        return (r < r_H_tol || r > r_far);
                    };

                    rk45_dormand_prince rk45(6, 1.0e-12, 1.0e-12);
                    rk45.integrate(dydx, stop1, stop2, 0.0, y0, true, t_out);

                    double Iobs = 0.0;
                    if (rk45.brightness[0] != 0) {
                        double rf = rk45.result.back()[0];
                        double u_rf = -rk45.result.back()[3];
                        double u_thf = -rk45.result.back()[4];
                        double u_phif = -rk45.result.back()[5];

                        double u_uppertf = sqrt((local_metric.gamma11 * u_rf * u_rf) + 
                                            (local_metric.gamma22 * u_thf * u_thf) + 
                                            (local_metric.gamma33 * u_phif * u_phif)) / local_metric.alpha;
                        double u_lower_tf = (-local_metric.alpha * local_metric.alpha * u_uppertf) + 
                                        (u_phif * local_metric.beta3);
                        double omega = 1.0 / (a + (pow(rf, 3.0 / 2.0) / sqrt(M)));
                        double oneplusz = (1.0 + (omega * u_phif / u_lower_tf)) / 
                                        sqrt(-local_metric.g_00 - (omega * omega * local_metric.g_33) - 
                                        (2 * omega * local_metric.g_03));
                        Iobs = 1.0 / (oneplusz * oneplusz * oneplusz);
                    }
                    
                    final_screen[resy - j - 1][i] = Iobs;
                    
                    // #pragma omp critical
                    // {
                    //     fprintf(stderr, "Thread %d is working on pixel (%d, %d)\n", 
                    //             omp_get_thread_num(), i, j);
                    // }
                }
            }
        }
    } // destructor of Clock

    // Write the final_screen to a csv file
    ofstream output_file2("../data/bh_image/BH_1920x1080_MP.csv");
    for (int i = 0; i < resy; i++) {
        for (int j = 0; j < resx; j++) {
            output_file2 << final_screen[i][j] << " ";
        }
        output_file2 << "\n";
    }
    output_file2.close();
     
    return 0;
}
