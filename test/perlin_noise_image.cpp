#include <iostream>
#include <fstream>
#include "../include/perlin.h"

int main() {
  // testing if perlin noise works with test image
  int N = 1024;
  int width = N;
  int height = N;
  int numOctaves = 4;  // Increased for more detail
  float persistence = 0.5f;
  
  float* noiseData = generatePerlinNoise(N, N, numOctaves, persistence);
  
  // Debug - check noise range before writing
  float minVal = 1.0f, maxVal = 0.0f;
  for (int i = 0; i < width * height; i++) {
    minVal = std::min(minVal, noiseData[i]);
    maxVal = std::max(maxVal, noiseData[i]);
  }
  std::cout << "Noise range: " << minVal << " to " << maxVal << std::endl;
  
  // output to file
  std::ofstream outputFile("perlin_noise.pgm");
  outputFile << "P2\n";  // PGM format
  outputFile << width << " " << height << "\n";
  outputFile << "255\n";  // Max pixel value
  
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float value = noiseData[y * width + x];
      
      // Scale the noise value to the range [0, 255]
      int pixelValue = static_cast<int>(value * 255.0f);
      
      // Ensure values in correct range
      pixelValue = std::max(0, std::min(255, pixelValue));
      
      outputFile << pixelValue << " ";
    }
    outputFile << "\n";
  }
  outputFile.close();
  std::cout << "Perlin noise image generated: perlin_noise.pgm" << std::endl;
    
  // free memory
  delete[] noiseData;
  return 0;
}
