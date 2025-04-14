#ifndef PERLIN_H
#define PERLIN_H

#include <cmath>
#include <cassert>
#include <algorithm> // For std::min and std::max

// Ken Perlin's Improved Noise
// @see https://cs.nyu.edu/~perlin/noise/
class PerlinGenerator {

public:
  // Constructor
  PerlinGenerator() {
    // Initialize the permutation table
    for (int i = 0; i < 256; ++i) {
      p[i] = permutation[i];
      }

    // Shuffle the permutation table with LCG
    // @see https://en.wikipedia.org/wiki/Linear_congruential_generator
    unsigned int seed = 0; // Use a fixed seed for reproducibility
    for (int i = 255; i > 0; --i) {
      int j = seed % (i + 1); // Random index 
      // Swap the elements in the permutation table
      int temp = p[i];
      p[i] = p[j];
      p[j] = temp;
      // Update seed with (seed * a + c) % m
      // where modulo of powers of 2 is just bitwise AND: x % m = x & (m - 1)
      // @see https://en.wikipedia.org/wiki/Modulo#Performance_issues
        seed = (seed * 1103515245 + 12345) & 0x7FFFFFFF; // using glibc a, c, and m = 2^31 in hex
    }
    // Duplicate the permutation table
    for (int i = 0; i < 256; ++i) {
      p[i + 256] = permutation[i];
    }
  }
  

  // 2D Perlin noise function
  float perlinNoise(float x, float y) {
    int X = static_cast<int>(std::floor(x)) & 255; // unit square
    int Y = static_cast<int>(std::floor(y)) & 255; // that surrounds point
    x -= std::floor(x); // relative x,y
    y -= std::floor(y); // coord in the square
    float u = fade(x); // smoothstep function set for nice interpolation
    float v = fade(y); // at each corner
    int A = p[X] + Y; // coordinate hash
    int B = p[X + 1] + Y;
    // bilinear interpolation: 2 lerps in u direction then 1 lerp in v
    return lerp(v, 
                lerp(u, grad(p[A], x, y), grad(p[B], x - 1, y)), 
                lerp(u, grad(p[A + 1], x, y - 1), grad(p[B + 1], x - 1, y - 1)));
  }

private:
  // 5th order Hermite polynomial smoothstep function
  float fade(float t) {
    return t * t * t * (t * (t * 6.0f - 15.0f) + 10.0f);
  }

  // basic linear interpolation
  float lerp(float t, float a, float b) {
    return a + t * (b - a);
  }

  // For 3D gradient we have 12 possible directions:
// (1,1,0), (1,-1,0), (-1,1,0), (-1,-1,0),
// (1,0,1), (1,0,-1), (-1,0,1), (-1,0,-1),
// (0,1,1), (0,1,-1), (0,-1,1), (0,-1,-1)
// h = hash & 15; 
//    generates 0-15 i.e. use last 4 bits for random gradient (8 directions in 3D)
// u = h < 8 ? x : y; 
//    8 = 0b1000 
//    if the very first bit is 0 then u = x else u = y
// v = h < 4 ? y : (h == 12 || h == 14 ? x : z); //
//    4 = 0b0100, 12 = 0b1100, 14 = 0b1110
//    if the first two bits are 0 then v = y  
//    else if the first two bits are 1 and last bit is 0 then v = x
//    else v = z
// return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v);
//    1 = 0b0001, 2 = 0b0010
//    if the first bit is 0 then u = u else u = -u
//    if the second bit is 0 then v = v else v = -v
//    return sum
// For 2D gradient we set z = 0; 
  float grad(int hash, float x, float y) {
    int h = hash & 15; // use last 4 bits
    float u = h < 8 ? x : y; // 8 = 0b1000
    float v = h < 4 ? y : (h == 12 || h == 14 ? x : 0); // 4 = 0b0100, 12 = 0b1100, 14 = 0b1110
    return ((h & 1) == 0 ? u : -u) + ((h & 2) == 0 ? v : -v); // 1 = 0b0001, 2 = 0b0010
  }

  int p[512]; // permutation table
  const int permutation[256] = {151,160,137,91,90,15,
   131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,142,8,99,37,240,21,10,23,
   190, 6,148,247,120,234,75,0,26,197,62,94,252,219,203,117,35,11,32,57,177,33,
   88,237,149,56,87,174,20,125,136,171,168, 68,175,74,165,71,134,139,48,27,166,
   77,146,158,231,83,111,229,122,60,211,133,230,220,105,92,41,55,46,245,40,244,
   102,143,54, 65,25,63,161, 1,216,80,73,209,76,132,187,208, 89,18,169,200,196,
   135,130,116,188,159,86,164,100,109,198,173,186, 3,64,52,217,226,250,124,123,
   5,202,38,147,118,126,255,82,85,212,207,206,59,227,47,16,58,17,182,189,28,42,
   223,183,170,213,119,248,152, 2,44,154,163, 70,221,153,101,155,167, 43,172,9,
   129,22,39,253, 19,98,108,110,79,113,224,232,178,185, 112,104,218,246,97,228,
   251,34,242,193,238,210,144,12,191,179,162,241, 81,51,145,235,249,14,239,107,
   49,192,214, 31,181,199,106,157,184, 84,204,176,115,121,50,45,127, 4,150,254,
   138,236,205,93,222,114,67,29,24,72,243,141,128,195,78,66,215,61,156,180
   };

};

// Perlin noise generation function for rectangular dimensions
float* generatePerlinNoise(int width, int height, int octaves = 1, float persistence = 0.5f) {
  PerlinGenerator generator;
  float* noiseMap = new float[width * height];
  
  // Scale factor for differnt patterns
  // 0.01f - 0.005f works well
  const float scale = 0.0075f; // lower values for larger patterns
  
  // Store min and max values for normalization
  float minNoise = 1.0f;
  float maxNoise = -1.0f;
    
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      float noise = 0.0f;
      float amplitude = 1.0f;
      float freq = 1.0f;
      float maxAmplitude = 0.0f; // for normalization
      
      for (int o = 0; o < octaves; ++o) {
        // Apply proper scaling to get visible pattern
        float sampleX = x * scale * freq;
        float sampleY = y * scale * freq;
        
        // Getting noise - this function produces values in range [-1,1]
        float sampleValue = generator.perlinNoise(sampleX, sampleY);
        noise += sampleValue * amplitude;
        maxAmplitude += amplitude;
        
        amplitude *= persistence;
        freq *= 2.0f;
      }
      
      // Normalize based on total amplitude
      noise /= maxAmplitude;
      
      // Track min and max for normalization
      minNoise = std::min(minNoise, noise);
      maxNoise = std::max(maxNoise, noise);
            
      noiseMap[y * width + x] = noise;
    }
  }
  
  // // Normalize to ensure full [-1,1] range
  float range = maxNoise - minNoise;
  for (int i = 0; i < width * height; ++i) {
    // Map from [minNoise, maxNoise] to [-1,1]
    noiseMap[i] = 2.0f * (noiseMap[i] - minNoise) / range - 1.0f;

    // map to [0,1] for image output
    // noiseMap[i] = (noiseMap[i] - minNoise) / range; // [0,1]
  }
  
  return noiseMap;
}

// // Square grid overload
// float* generatePerlinNoise(int N, int octaves = 1, float persistence = 0.5f) {
//   return generatePerlinNoise(N, N, octaves, persistence);
// }

#endif // PERLIN_H