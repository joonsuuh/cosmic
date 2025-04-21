#include <algorithm>
#include <fstream>
#include <iostream>

// External Libraries
#include <omp.h>

// Project Headers
#include "timer.h"
#include "metric.h"
#include "dormand_prince.h"
#include "ray_tracer.h"
#include "constants.h"
#include "config.h"
#include "image_processing.h"
#include "omp_helper.h"


static inline int numThreads = ThreadCount::kManual;

// Ray tracing functions
void performRayTracing(RayTracer& rayTracer, float* screenBuffer, 
                      const BlackHole& bhConfig,
                      int pixelWidth, int pixelHeight, Timer& timer);

int main(int argc, char* argv[]) {
  std::cout << "CPU version running" << std::endl;
  Timer timer;
  
  // ===== CONFIG SETUP =====
  BlackHole bhConfig;
  bhConfig.setObserverAngle(85.0);  // Set angle in degrees
  
  Image imgConfig;

  // Set up output configuration
  OutputConfig outConfig;
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
  float* screenBuffer = new float[numPixels](); // 1D row-major order
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
void performRayTracing(RayTracer& rayTracer, float* screenBuffer, 
                      const BlackHole& bhConfig,
                      int pixelWidth, int pixelHeight, Timer& timer) {
  #pragma omp parallel
  {
    // Each thread needs its own metric, integrator, and ray
    BoyerLindquistMetric thread_metric(bhConfig.spin(), bhConfig.mass());
    DormandPrinceRK45 integrator(6, Constants::Integration::ABS_TOLERANCE,
                                Constants::Integration::REL_TOLERANCE,
                                Constants::Integration::MIN_STEP_SIZE,
                                Constants::Integration::MAX_STEP_SIZE,
                                Constants::Integration::INITIAL_STEP_SIZE,
                                Constants::Integration::MAX_ITERATIONS);
    
    // Allocate ray memory
    float* y {new float[6]{}};
    
    #pragma omp for collapse(2) schedule(dynamic) nowait
    for (int i = 0; i < pixelWidth; i++) {
      for (int j = 0; j < pixelHeight; j++) {
        float intensity = 0.0;
        
        if (rayTracer.traceRay(i, j, thread_metric, integrator, y, intensity)) {
          screenBuffer[j * pixelWidth + i] = intensity;
        }
      }
    }
    
    // Clean up memory
    delete[] y;
  }
}