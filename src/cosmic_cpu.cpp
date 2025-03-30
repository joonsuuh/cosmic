#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <utility>

// External Libraries
#include <omp.h>

// Project Headers
#include "timer.h"
#include "metric.h"
#include "dormand_prince.h"
#include "ray_tracer.h"
#include "constants.h"
#include "config.h"


// ===== FORWARD DECLARATIONS =====
// Image processing functions
void normalizeScreenBuffer(double* screenBuffer, int numPixels);
void applyColormap(double value, int& r, int& g, int& b);
void writeImageToFile(const std::string& filename, double* screenBuffer, int pixelWidth, int pixelHeight);
void applyHotColorMap(double value, int& r, int& g, int& b);

// Output functions
inline void printDisplayResolution(int pixelWidth, int pixelHeight);
inline void printBufferSize(int numPixels);
inline void printNumberOfThreads();

// OpenMP config
enum ThreadCount {
  kDefault,
  kSingle,
  kManual = 8,
};
static inline int numThreads = ThreadCount::kManual;
inline void setOpenMPThreads(int numThreads);

// Ray tracing functions
void performRayTracing(RayTracer& rayTracer, double* screenBuffer, 
                      const Config::BlackHole& bhConfig,
                      int pixelWidth, int pixelHeight, Timer& timer);

int main(int argc, char* argv[]) {
  std::cout << "CPU version running" << std::endl;
  Timer timer;
  
  // ===== CONFIG SETUP =====
  Config::BlackHole bhConfig;
  bhConfig.spin = 0.99;
  bhConfig.setObserverAngle(85.0);  // Set angle in degrees
  
  Config::Image imgConfig;
  imgConfig.setAspectRatio(16, 9);
  imgConfig.scale = 20;
  // imgConfig.cameraScale = 1.5;
 
  // Set up output configuration
  Config::OutputConfig outConfig;
  outConfig.setDescriptiveFilename(bhConfig, imgConfig, "bh_cpu"); 
  
  // Create ray tracer object with config objects
  RayTracer rayTracer(bhConfig, imgConfig);
  
  setOpenMPThreads(numThreads);

  // ===== MEMORY ALLOCATION =====
  timer.start("INITIALIZING SCREEN BUFFER");
  const int pixelWidth = imgConfig.width();
  const int pixelHeight = imgConfig.height();
  const int numPixels = imgConfig.numPixels();
  
  // Allocate screen buffer
  double* screenBuffer = new double[numPixels](); // 1D row-major order
  timer.stop();

  // ===== OUTPUT CONFIG =====
  printDisplayResolution(pixelWidth, pixelHeight);
  printBufferSize(numPixels);
  printNumberOfThreads();

  // ===== RAY TRACING =====
  timer.start("COMPUTING PIXEL INTENSITY");
  performRayTracing(rayTracer, screenBuffer, bhConfig, pixelWidth, pixelHeight, timer);
  timer.stop();

  // ===== POST-PROCESSING =====
  // Normalize the intensity values in the screen buffer
  timer.start("NORMALIZING SCREEN BUFFER");
  normalizeScreenBuffer(screenBuffer, numPixels);
  timer.stop();
  
  // Write the image to file using the output config
  timer.start("WRITING IMAGE TO FILE");
  writeImageToFile(outConfig.getFullPath(), screenBuffer, pixelWidth, pixelHeight);
  timer.stop();
  
  // Free screen buffer memory
  delete[] screenBuffer;

  return 0;
}

// ===== RAY TRACING FUNCTIONS =====
void performRayTracing(RayTracer& rayTracer, double* screenBuffer, 
                      const Config::BlackHole& bhConfig,
                      int pixelWidth, int pixelHeight, Timer& timer) {
  #pragma omp parallel
  {
    // Each thread needs its own metric, integrator, and ray
    BoyerLindquistMetric thread_metric(bhConfig.spin, bhConfig.mass);
    DormandPrinceRK45 integrator(6, Constants::Integration::ABS_TOLERANCE, Constants::Integration::REL_TOLERANCE);
    
    // Allocate ray memory
    double* y {new double[6]{}};
    
    #pragma omp for collapse(2) schedule(dynamic) nowait
    for (int i = 0; i < pixelWidth; i++) {
      for (int j = 0; j < pixelHeight; j++) {
        double intensity = 0.0;
        
        if (rayTracer.traceRay(i, j, thread_metric, integrator, y, intensity)) {
          screenBuffer[j * pixelWidth + i] = intensity;
        }
      }
    }
    
    // Clean up memory
    delete[] y;
  }
}

// ===== IMAGE PROCESSING FUNCTIONS =====

void normalizeScreenBuffer(double* screenBuffer, int numPixels) {
  double max_intensity = 0.0;

  // First pass: find max intensity
  for (int i = 0; i < numPixels; i++) {
    max_intensity = std::max(max_intensity, screenBuffer[i]);
  }

  // Normalize the screen buffer
  for (int i = 0; i < numPixels; i++) {
    screenBuffer[i] /= max_intensity;
  }
}

/**
 * @brief matplotlib-style hot colormap
 * 
 * A sequential colormap that transitions from black -> red -> yellow -> white.
 * 
 * @param value Normalized intensity value between 0 and 1
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * 
 * @note THE FOLLOWING VALUES ARE BASED ON THE HOT COLORMAP FROM MATPLOTLIB:
 * ====================================
 * ========= PALETTE COLORMAP =========
 * ====================================
 * red:   (0.0      , 0.0416 , 0.0416),
 *        (0.365079 , 1.0    , 1.0   ),
 *        (1.0      , 1.0    , 1.0   )
 * ====================================
 * green: (0.0      , 0.0    , 0.0   ),
 *        (0.365079 , 0.0    , 0.0   ),
 *        (0.746032 , 1.0     , 1.0  ),
 *        (1.0      , 1.0     , 1.0  )
 * ===================================
 * blue:  (0.0      , 0.0     , 0.0  ),
 *        (0.746032 , 0.0     , 0.0  ),
 *        (1.0      , 1.0     , 1.0  ) 
 * ===================================
 * @see [matplotlib colormap](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py)
 */
void applyHotColorMap(double value, int& r, int& g, int& b) {
  // Define colormap thresholds
  const double RED_THRESHOLD = 0.365079;    // Threshold for black to red
  const double YELLOW_THRESHOLD = 0.746032; // Threshold for red to yellow

  // Implement hot colormap
  if (value < RED_THRESHOLD) {
    // Black to red
    r = static_cast<int>((value / RED_THRESHOLD) * 255);
    g = 0;
    b = 0;
  } else if (value < YELLOW_THRESHOLD) {
    // Red to yellow
    r = 255;
    g = static_cast<int>(((value - RED_THRESHOLD) / (YELLOW_THRESHOLD - RED_THRESHOLD)) * 255);
    b = 0;
  } else {
    // Yellow to white
    r = 255;
    g = 255;
    b = static_cast<int>(((value - YELLOW_THRESHOLD) / (1.0 - YELLOW_THRESHOLD)) * 255);
  }
}

// Write image to PPM file
void writeImageToFile(const std::string& filename, double* screenBuffer, int pixelWidth, int pixelHeight) {
  std::ofstream outputFile(filename);
  outputFile << "P3\n";
  outputFile << pixelWidth << " " << pixelHeight << "\n";
  outputFile << "255\n";

  for (int i = 0; i < pixelHeight; i++) {
    for (int j = 0; j < pixelWidth; j++) {
      double value = screenBuffer[i * pixelWidth + j];
      int r, g, b;

      applyHotColorMap(value, r, g, b);

      outputFile << r << " " << g << " " << b << " ";
    }
    outputFile << "\n";
  }
  outputFile.close();
}

// ===== OPENMP FUNCTIONS =====
inline void setOpenMPThreads(int numThreads) {
  const int maxThreads = omp_get_max_threads();
  switch (numThreads) {
    case ThreadCount::kDefault:
      omp_set_num_threads(maxThreads);
      break;
    case ThreadCount::kSingle:
      omp_set_num_threads(1);
      break;
    case ThreadCount::kManual:
      if (numThreads > maxThreads) {
        std::cerr << "Invalid number of threads specified. Using Default..." << std::endl;
        omp_set_num_threads(maxThreads);
      } else {
        omp_set_num_threads(numThreads);
      }
      break;
  }
}

// ===== OUTPUT FUNCTIONS =====

inline void printDisplayResolution(int pixelWidth, int pixelHeight) {
  std::cout << "Display Resolution: " << pixelWidth << "x" << pixelHeight << std::endl;
}

inline void printBufferSize(int numPixels) {
  std::cout << "Buffer Size: " << numPixels << " pixels" << "\n"
            << "Buffer Size: " << numPixels * sizeof(double) / (1024.0 * 1024.0) << " MB" << std::endl;
}

inline void printNumberOfThreads() {
  int numThreads = 1; // Default 1 thread
  #pragma omp parallel
  {
    #pragma omp single
    {
      numThreads = omp_get_num_threads();
    }
  }
  // Print number of threads used vs available
  std::cout << "Number of Threads: " << numThreads << " / " << omp_get_max_threads() << std::endl;
}