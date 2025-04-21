#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

// CUDA HEADERS
#include <cuda_runtime.h>

// Project CUDA headers
#include "ray_tracer_cuda.cuh"
#include "cuda_helper.cuh"

// Project headers
#include "config.h"
#include "image_processing.h"
#include "perlin.h"
#include "timer.h"


void printDeviceInfo();

int main(int argc, char* argv[]) {
  std::cout << "CUDA version running" << std::endl;
  Timer timer;

  // Print CUDA device info
  printDeviceInfo();

  // ===== CONFIG SETUP =====
  BlackHole bhConfig;

  Image imgConfig;

  // Create camera parameters
  const int pixelWidth = imgConfig.width();
  const int pixelHeight = imgConfig.height();
  const int numPixels = imgConfig.numPixels();

  // Set up output configuration
  OutputConfig outConfig;
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
      bhConfig.spin(),        bhConfig.mass(),          bhConfig.distance(),
      bhConfig.theta(),       bhConfig.phi(),           bhConfig.innerRadius(),
      bhConfig.outerRadius(), bhConfig.diskTolerance(), bhConfig.farRadius()};
  float imgParams[3] = {imgConfig.aspectWidth(), imgConfig.aspectHeight(),
                        imgConfig.scale()};
  float cameraParams[4] = {imgConfig.offsetX(), imgConfig.offsetY(),
                           imgConfig.stepX(), imgConfig.stepY()};
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