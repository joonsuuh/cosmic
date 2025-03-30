#include "timer.h"
#include "metric.h"
#include "rk45_dp2.h"
#include <chrono>
#include <cmath>
#include <fstream>
#include <algorithm>
// #include <iomanip>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

#ifndef PPM_DIR
  #define PPM_DIR "data/"
#endif

// OpenGL stuff
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader.h"

// Configuration struct to hold black hole parameters
struct BlackHoleConfig {
  double a;       // Black hole spin parameter
  double M;       // Black hole mass
  double D;       // Distance to observer
  double theta0;  // Observer inclination angle (degrees)
  double phi0;    // Observer azimuthal angle (degrees)
  double innerRadius;    // Inner radius of accretion disk
  double outerRadius;   // Outer radius of accretion disk
  double epsilon; // Integration tolerance
  double r_H;     // Event horizon radius
  double r_H_tol; // Tolerance factor for horizon detection
  double farRadius;   // Far field boundary
  int aspectWidth;     // Width aspect ratio
  int ratioy;     // Height aspect ratio
  int imageScale;   // Resolution scale factor
  int pixelWidth;
  int pixelHeight;
};

// Key callback function
void key_callback(GLFWwindow *window, int key, int scancode, int action,
                  int mods) {
  if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, GLFW_TRUE);
  }
}

// Function to calculate black hole image
std::vector<float> calculateBlackHoleImage(const BlackHoleConfig &config) {
  std::cout << "Calculating black hole image at resolution: " << config.pixelWidth << "x"
            << config.pixelHeight << std::endl;

  // Initialize 2D vector for the final image
  std::vector<float> final_screen(config.pixelHeight * config.pixelWidth, 0.0f);

  // Screen coordinates setup
  double offsetX = -25.0;
  double offsetY = -12.5;
  double y_sc0 = offsetY;
  double stepX = std::abs(2.0 * offsetX / config.pixelWidth);
  offsetX += 0.5 * stepX;
  double stepY = std::abs(2.0 * offsetY / config.pixelHeight);

  // Configure OpenMP
  int max_threads = omp_get_max_threads();
  int num_threads = max_threads;
  omp_set_num_threads(num_threads);
  
  int using_threads;  // Initialize outside parallel region
  #pragma omp parallel
  {
    #pragma omp master
    {
      using_threads = omp_get_num_threads();
    }
  }
  
  std::cout << "Available threads: " << max_threads << "\n"
            << "Used threads: " << using_threads << std::endl;

// Parallel region for ray tracing
#pragma omp parallel
  {
    // Each thread needs its own metric instance
    BoyerLindquistMetric local_metric(config.a, config.M);

// Calculate rays in parallel
#pragma omp for collapse(2) schedule(dynamic, 16) nowait
    for (int i = 0; i < config.pixelWidth; i++) {
      for (int j = 0; j < config.pixelHeight; j++) {
        // Calculate screen coordinates
        double local_x_sc = offsetX + (i * stepX);
        double local_y_sc = y_sc0 + (j * stepY);

        // Calculate initial ray parameters
        double beta = local_x_sc / config.D;
        double alpha = local_y_sc / config.D;
        double r = std::sqrt((config.D * config.D) + (local_x_sc * local_x_sc) +
                             (local_y_sc * local_y_sc));
        double theta = config.theta0 - alpha;
        double phi = beta;
        local_metric.computeMetric(r, theta);

        // Define differential equations for geodesic path
        // auto dydx = [&local_metric](double x, const std::vector<double> &y) {
        auto dydx = [&local_metric](const std::vector<double> &y) {
          local_metric.computeMetric(y[0], y[1]);
          double r = y[0];
          double th = y[1];
          double phi = y[2];
          double u_r = y[3];
          double u_th = y[4];
          double u_phi = y[5];

          // Calculate upper time component
          double u_uppert = std::sqrt((local_metric.gamma11 * u_r * u_r) +
                                      (local_metric.gamma22 * u_th * u_th) +
                                      (local_metric.gamma33 * u_phi * u_phi)) /
                            local_metric.alpha;

          // Position derivatives
          double drdt = local_metric.gamma11 * u_r / u_uppert;
          double dthdt = local_metric.gamma22 * u_th / u_uppert;
          double dphidt =
              (local_metric.gamma33 * u_phi / u_uppert) - local_metric.beta3;

          // Momentum derivatives
          double temp1 = (u_r * u_r * local_metric.d_gamma11_dr) +
                         (u_th * u_th * local_metric.d_gamma22_dr) +
                         (u_phi * u_phi * local_metric.d_gamma33_dr);
          double durdt =
              (-local_metric.alpha * u_uppert * local_metric.d_alpha_dr) +
              (u_phi * local_metric.d_beta3_dr) - (temp1 / (2.0 * u_uppert));

          double temp2 = (u_r * u_r * local_metric.d_gamma11_dth) +
                         (u_th * u_th * local_metric.d_gamma22_dth) +
                         (u_phi * u_phi * local_metric.d_gamma33_dth);
          double duthdt =
              (-local_metric.alpha * u_uppert * local_metric.d_alpha_dth) +
              (u_phi * local_metric.d_beta3_dth) - temp2 / (2.0 * u_uppert);

          // u_phi is conserved
          double duphidt = 0;

          return std::vector<double>{drdt,  dthdt,  dphidt,
                                     durdt, duthdt, duphidt};
        };

        // Calculate initial velocities
        double u_r =
            -std::sqrt(local_metric.g_11) * std::cos(beta) * std::cos(alpha);
        double u_theta = -std::sqrt(local_metric.g_22) * std::sin(alpha);
        double u_phi =
            std::sqrt(local_metric.g_33) * std::sin(beta) * std::cos(alpha);

        // Initial state vector
        std::vector<double> y0 = {r, theta, phi, u_r, u_theta, u_phi};

        // Stopping conditions
        auto stop_at_disk = [&config](double x, const std::vector<double> &y) {
          double r = y[0];
          double theta = y[1];
          return ((r >= config.innerRadius && r <= config.outerRadius) &&
                  (std::abs(theta - M_PI / 2.0) < 0.01));
        };

        auto stop_at_boundary = [&config](double x,
                                          const std::vector<double> &y) {
          double r = y[0];
          return (r < config.r_H_tol || r > config.farRadius);
        };

        // Perform integration
        DormandPrinceRK45 rk45(6, 1.0e-12, 1.0e-12);
        rk45.integrate(dydx, stop_at_disk, stop_at_boundary, 0.0, y0);

        // Calculate observed intensity
        float Iobs = 0.0f;
        if (rk45.brightness) { // Changed from rk45.brightness[0] to
                               // rk45.brightness
          // Get final state
          // double rf = rk45.result.back()[0];
          // double u_rf = -rk45.result.back()[3];
          // double u_thf = -rk45.result.back()[4];
          // double u_phif = -rk45.result.back()[5];
          double rf = rk45.y_next[0];
          double u_rf = -rk45.y_next[3];
          double u_thf = -rk45.y_next[4];
          double u_phif = -rk45.y_next[5];

          // Compute metric at final position
          // local_metric.compute_metric(rf, rk45.result.back()[1]);

          // Calculate final 4-velocity components
          double u_uppertf =
              std::sqrt((local_metric.gamma11 * u_rf * u_rf) +
                        (local_metric.gamma22 * u_thf * u_thf) +
                        (local_metric.gamma33 * u_phif * u_phif)) /
              local_metric.alpha;
          double u_lower_tf =
              (-local_metric.alpha * local_metric.alpha * u_uppertf) +
              (u_phif * local_metric.beta3);

          // Calculate redshift factor
          double omega =
              1.0 /
              (config.a + (std::pow(rf, 3.0 / 2.0) / std::sqrt(config.M)));
          double oneplusz = (1.0 + (omega * u_phif / u_lower_tf)) /
                            std::sqrt(-local_metric.g_00 -
                                      (omega * omega * local_metric.g_33) -
                                      (2 * omega * local_metric.g_03));

          // Calculate observed intensity with relativistic beaming
          Iobs = 1.0f / (oneplusz * oneplusz * oneplusz);
        }

        // Store result (flipping y-coordinate for image)
        final_screen[(config.pixelHeight - j - 1) * config.pixelWidth + i] = Iobs;
      }
    }
  }
  return final_screen;
}

// Main function
int main() {
  std::cout << "Black Hole Raytracer starting up...\n";

  // Configure black hole parameters
  BlackHoleConfig config;
  config.a = 0.99;                     // Spin parameter
  config.M = 1.0;                      // Mass
  config.D = 500.0;                    // Distance
  config.theta0 = 85.0 * M_PI / 180.0; // Inclination angle
  config.phi0 = 0.0;                   // Azimuthal angle
  config.innerRadius = 5.0 * config.M;        // Inner disk radius
  config.outerRadius = 20.0 * config.M;      // Outer disk radius
  config.epsilon = 1.0e-5;             // Integration tolerance
  config.r_H = config.M + std::sqrt(config.M * config.M -
                                    config.a * config.a); // Event horizon
  config.r_H_tol = 1.01 * config.r_H; // Horizon detection tolerance
  config.farRadius = config.outerRadius * 50.0; // Far field boundary
  config.aspectWidth = 16;                 // Width aspect ratio
  config.ratioy = 9;                  // Height aspect ratio
  config.imageScale = 10;               // Resolution scale
  config.pixelWidth = config.aspectWidth * config.imageScale;
  config.pixelHeight = config.ratioy * config.imageScale;

  // Calculate black hole image
  Timer timer;
  timer.start("Raytracing");
  std::vector<float> image_data = calculateBlackHoleImage(config);
  timer.stop();
  // Image render
  std::cout << "Raytracing complete. Rendering image...\n";
  // NORMALIZE IMAGE DATA
  float max_intensity = 0.0;
  // Update array access in normalization
  for (auto &intensity : image_data) {
    max_intensity = std::max(max_intensity, intensity);
  }
  
  for (auto &intensity : image_data) {
    intensity /= max_intensity;
  }

  // APPLY HOT colormap from matlab/matplotlib
  // https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py
  // _hot_data = {'red':   ((0., 0.0416, 0.0416),(0.365079, 1.000000, 1.000000),(1.0, 1.0, 1.0)),
            //  'green': ((0., 0., 0.),(0.365079, 0.000000, 0.000000),
            //            (0.746032, 1.000000, 1.000000),(1.0, 1.0, 1.0)),
            //  'blue':  ((0., 0., 0.),(0.746032, 0.000000, 0.000000),(1.0, 1.0, 1.0))} 

  // Write the final_screen to a ppm file
  std::ofstream output_file(PPM_DIR "bh_opengl.ppm");
  output_file << "P3\n";
  output_file << config.pixelWidth << " " << config.pixelHeight << "\n";
  output_file << "255\n";

  for (int i = 0; i < config.pixelHeight; i++) {
    for (int j = 0; j < config.pixelWidth; j++) {
      double value = image_data[i * config.pixelWidth + j];
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
  
  // Initialize GLFW for visualization
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }

  // Set up GLFW window hints
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

  // Create window
  int window_width = 800;
  int window_height = 450;
  GLFWwindow *window = glfwCreateWindow(window_width, window_height,
                                        "Black Hole Raytracer", NULL, NULL);
  if (!window) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }

  glfwMakeContextCurrent(window);
  glfwSetKeyCallback(window, key_callback);

  // Initialize GLAD
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    glfwTerminate();
    return -1;
  }
  std::cout << "OpenGL version: " << glGetString(GL_VERSION) << std::endl;


  // Create and bind texture
  GLuint texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // Allocate texture memory and upload image data
  int tex_width = config.pixelWidth;
  int tex_height = config.pixelHeight;

  // Convert 2D vector to 1D array for OpenGL
  // std::vector<float> flattened_data;
  // flattened_data.reserve(tex_width * tex_height);
  // for (const auto &row : image_data) {
  //   flattened_data.insert(flattened_data.end(), row.begin(), row.end());
  // }

  // Upload texture data
  glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, tex_width, tex_height, 0, GL_RED,
               GL_FLOAT, image_data.data());
  glGenerateMipmap(GL_TEXTURE_2D);

  // Create shader program
  Shader shader("simple.vert", "bhtexture.frag"); // Update path

  // Create vertex data for full-screen quad
  float vertices[] = {
      // positions  // texture coords
      -1.0f, -1.0f, 0.0f, 1.0f, // bottom left
      1.0f,  -1.0f, 1.0f, 1.0f, // bottom right
      1.0f,  1.0f,  1.0f, 0.0f, // top right
      -1.0f, 1.0f,  0.0f, 0.0f  // top left
  };
  unsigned int indices[] = {0, 1, 2, 0, 2, 3};

  // Set up vertex buffer and vertex array object
  GLuint VAO, VBO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
               GL_STATIC_DRAW);

  // Set up vertex attributes
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void *)0);
  glEnableVertexAttribArray(0);
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void *)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // Initial brightness and contrast values

  // Main render loop
  std::cout << "Starting render loop. Press ESC to exit." << std::endl;
  while (!glfwWindowShouldClose(window)) {
    glClear(GL_COLOR_BUFFER_BIT);

    // Render the quad with the black hole texture
    shader.use();
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // Clean up resources
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glDeleteTextures(1, &texture);
  glDeleteShader(shader.ID);

  glfwTerminate();
  return 0;
}