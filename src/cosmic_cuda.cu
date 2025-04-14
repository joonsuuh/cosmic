#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

// CUDA HEADERS
#include <cuda_runtime.h>

// Project CUDA headers
#include "ray_tracer_cuda.cuh"

// Project headers
#include "config.h"
#include "perlin.h"
#include "timer.h"

// CUDA Error Handling
#define CUDA_CHECK(call)                                                   \
  do {                                                                     \
    cudaError_t error = call;                                              \
    if (error != cudaSuccess) {                                            \
      std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << ": " \
                << cudaGetErrorString(error) << std::endl;                 \
      exit(1);                                                             \
    }                                                                      \
  } while (0)

// Forward declarations
void normalizeScreenBuffer(float* screenBuffer, int numPixels);
void writeImageToFile(const std::string& filename, float* screenBuffer,
                      int pixelWidth, int pixelHeight);
void applyHotColorMap(float value, int& r, int& g, int& b);
void printDeviceInfo();

int main(int argc, char* argv[]) {
  std::cout << "CUDA version running" << std::endl;
  Timer timer;

  // Print CUDA device info
  printDeviceInfo();

  // ===== CONFIG SETUP =====
  Config::BlackHole bhConfig;
  bhConfig.spin = 0.85f;
  // bhConfig.theta = M_PI;
  // bhConfig.setObserverAngle(175.0f);

  Config::Image imgConfig;
  imgConfig.scale = 120;

  // Create camera parameters
  const int pixelWidth = imgConfig.width();
  const int pixelHeight = imgConfig.height();
  const int numPixels = imgConfig.numPixels();
  // Config::Image::CameraParams camParams = imgConfig.getCameraParams();

  // Set up output configuration
  Config::OutputConfig outConfig;
  outConfig.setDescriptiveFilename(bhConfig, imgConfig, "TEMP");

  // ===== MEMORY ALLOCATION =====
  timer.start("INITIALIZING MEMORY");
  float* h_screenBuffer = new float[numPixels]();
  float* d_screenBuffer;
  float* d_bloomBuffer;
  CUDA_CHECK(cudaMalloc(&d_screenBuffer, numPixels * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&d_bloomBuffer, numPixels * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_screenBuffer, 0, numPixels * sizeof(float)));
  CUDA_CHECK(cudaMemset(d_bloomBuffer, 0, numPixels * sizeof(float)));
  timer.stop();

  // ===== OUTPUT CONFIG =====
  std::cout << "Display Resolution: " << pixelWidth << "x" << pixelHeight
            << std::endl;
  std::cout << "Buffer Size: " << numPixels << " pixels" << "\n"
            << "Buffer Size: "
            << numPixels * sizeof(float) / (1024.0f * 1024.0f) << " MB"
            << std::endl;

  // ===== PERLIN NOISE SETUP =====
  timer.start("GENERATING PERLIN NOISE");
  const int noiseSize = 1024;
  float* noiseMap = generatePerlinNoise(noiseSize, noiseSize, 4, 0.5f);
  float* d_noiseMap;
  CUDA_CHECK(cudaMalloc(&d_noiseMap, noiseSize * noiseSize * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(d_noiseMap, noiseMap,
                        noiseSize * noiseSize * sizeof(float),
                        cudaMemcpyHostToDevice));
  timer.stop();

  // ===== COPY CONFIG TO CONSTANT MEMORY =====
  timer.start("SETUP CONSTANT MEMORY");
  float bhParams[9] = {
      bhConfig.spin,        bhConfig.mass,          bhConfig.distance,
      bhConfig.theta,       bhConfig.phi,           bhConfig.innerRadius,
      bhConfig.outerRadius, bhConfig.diskTolerance, bhConfig.farRadius};
  float imgParams[4] = {imgConfig.aspectWidth, imgConfig.aspectHeight,
                        imgConfig.scale, imgConfig.cameraScale};
  float cameraParams[4] = {imgConfig.offsetX, imgConfig.offsetY,
                           imgConfig.stepX, imgConfig.stepY};
  float integrationParams[6] = {Constants::Integration::ABS_TOLERANCE,
                                Constants::Integration::REL_TOLERANCE,
                                Constants::Integration::MIN_STEP_SIZE,
                                Constants::Integration::MAX_STEP_SIZE,
                                Constants::Integration::INITIAL_STEP_SIZE,
                                Constants::Integration::DISK_TOLERANCE};
  CUDA_CHECK(cudaMemcpyToSymbol(c_bhParams_data, bhParams, sizeof(bhParams)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(c_imgParams_data, imgParams, sizeof(imgParams)));
  CUDA_CHECK(
      cudaMemcpyToSymbol(c_camParams_data, cameraParams, sizeof(cameraParams)));
  CUDA_CHECK(cudaMemcpyToSymbol(c_integrationConstants, integrationParams,
                                sizeof(integrationParams)));

  timer.stop();

  // ===== RAY TRACING =====
  timer.start("COMPUTING PIXEL INTENSITY");

  dim3 blockSize(32, 4);
  dim3 gridSize((pixelWidth + blockSize.x - 1) / blockSize.x,
                (pixelHeight + blockSize.y - 1) / blockSize.y);

  rayTraceKernel<<<gridSize, blockSize>>>(d_screenBuffer, d_noiseMap, noiseSize,
                                          0.0f  // Initial time = 0
  );
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();

  // Copy results back to host
  timer.start("COPYING DATA FROM DEVICE");
  CUDA_CHECK(cudaMemcpy(h_screenBuffer, d_screenBuffer,
                        numPixels * sizeof(float), cudaMemcpyDeviceToHost));
  timer.stop();

  // ===== POST-PROCESSING =====
  timer.start("NORMALIZING SCREEN BUFFER");
  normalizeScreenBuffer(h_screenBuffer, numPixels);
  timer.stop();

  // Write the image to file
  timer.start("WRITING IMAGE TO FILE");
  writeImageToFile(outConfig.getFullPath(), h_screenBuffer, pixelWidth,
                   pixelHeight);
  timer.stop();

  // Free memory
  delete[] h_screenBuffer;
  delete[] noiseMap;
  CUDA_CHECK(cudaFree(d_screenBuffer));
  CUDA_CHECK(cudaFree(d_noiseMap));

  return 0;
}

// Print CUDA device information
void printDeviceInfo() {
  int deviceCount = 0;
  CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
  assert(deviceCount > 0 && "No CUDA devices found!");

  cudaDeviceProp deviceProp;
  CUDA_CHECK(cudaGetDeviceProperties(&deviceProp, 0));

  std::cout << "Device " << "0" << ": " << deviceProp.name << "\n"
            << "  Compute Capability: " << deviceProp.major << "."
            << deviceProp.minor << "\n"
            << "  Total Global Memory: "
            << deviceProp.totalGlobalMem / (1024 * 1024) << " MB" << "\n"
            << "  Multiprocessors: " << deviceProp.multiProcessorCount << "\n"
            << "  Max Threads per Block: " << deviceProp.maxThreadsPerBlock
            << "\n"
            << "  Max Threads Dimensions: (" << deviceProp.maxThreadsDim[0]
            << ", " << deviceProp.maxThreadsDim[1] << ", "
            << deviceProp.maxThreadsDim[2] << ")" << "\n"
            << "  Max Grid Size: (" << deviceProp.maxGridSize[0] << ", "
            << deviceProp.maxGridSize[1] << ", " << deviceProp.maxGridSize[2]
            << ")" << std::endl;
}

// Normalize the screen buffer values
void normalizeScreenBuffer(float* screenBuffer, int numPixels) {
  float max_intensity = 0.0f;

  // First pass: find max intensity
  for (int i = 0; i < numPixels; i++) {
    max_intensity = std::max(max_intensity, screenBuffer[i]);
  }

  // Normalize the screen buffer
  if (max_intensity > 0.0f) {
    for (int i = 0; i < numPixels; i++) {
      screenBuffer[i] /= max_intensity;
    }
  }
}

// Apply hot colormap (same as in CPU version)
void applyHotColorMap(float value, int& r, int& g, int& b) {
  // Define colormap thresholds
  const float RED_THRESHOLD = 0.365079f;     // Threshold for black to red
  const float YELLOW_THRESHOLD = 0.746032f;  // Threshold for red to yellow

  // Implement hot colormap
  if (value < RED_THRESHOLD) {
    // Black to red
    r = static_cast<int>((value / RED_THRESHOLD) * 255);
    g = 0;
    b = 0;
  } else if (value < YELLOW_THRESHOLD) {
    // Red to yellow
    r = 255;
    g = static_cast<int>(
        ((value - RED_THRESHOLD) / (YELLOW_THRESHOLD - RED_THRESHOLD)) * 255);
    b = 0;
  } else {
    // Yellow to white
    r = 255;
    g = 255;
    b = static_cast<int>(
        ((value - YELLOW_THRESHOLD) / (1.0f - YELLOW_THRESHOLD)) * 255);
  }
}

// Write image to PPM file (same as in CPU version)
void writeImageToFile(const std::string& filename, float* screenBuffer,
                      int pixelWidth, int pixelHeight) {
  std::ofstream outputFile(filename);
  outputFile << "P3\n";
  outputFile << pixelWidth << " " << pixelHeight << "\n";
  outputFile << "255\n";

  for (int i = 0; i < pixelHeight; i++) {
    for (int j = 0; j < pixelWidth; j++) {
      float value = screenBuffer[i * pixelWidth + j];
      int r, g, b;

      applyHotColorMap(value, r, g, b);

      outputFile << r << " " << g << " " << b << " ";
    }
    outputFile << "\n";
  }
  outputFile.close();
}
