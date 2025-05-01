#ifndef IMAGE_PROCESSING_H
#define IMAGE_PROCESSING_H

#include <iostream>
#include <fstream>

// ===== IMAGE PROCESSING FUNCTIONS =====

void normalizeScreenBuffer(float* screenBuffer, int numPixels) {
  float max_intensity = 0.0;

  // First pass: find max intensity
  for (int i = 0; i < numPixels; i++) {
    max_intensity = std::max(max_intensity, screenBuffer[i]);
  }

  // Normalize the screen buffer
  for (int i = 0; i < numPixels; i++) {
    screenBuffer[i] /= max_intensity;
  }
}

// matplotlib-style hot colormap
//  
// A sequential colormap that transitions from black -> red -> yellow -> white.
// 
// value: normalized intensity value between 0 and 1
// r: Red component (0-255)
// g: Green component (0-255)
// b: Blue component (0-255)
// 
// THE FOLLOWING VALUES ARE BASED ON THE HOT COLORMAP FROM MATPLOTLIB:
// ========= PALETTE COLORMAP =========
// red:   (0.0      , 0.0416 , 0.0416),
//        (0.365079 , 1.0    , 1.0   ),
//        (1.0      , 1.0    , 1.0   )
// ====================================
// green: (0.0      , 0.0    , 0.0   ),
//        (0.365079 , 0.0    , 0.0   ),
//        (0.746032 , 1.0     , 1.0  ),
//        (1.0      , 1.0     , 1.0  )
// ===================================
// blue:  (0.0      , 0.0     , 0.0  ),
//        (0.746032 , 0.0     , 0.0  ),
//        (1.0      , 1.0     , 1.0  ) 
// ===================================
// source: https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py)
template<typename red, typename green, typename blue>
void applyHotColorMap(float value, red& r, green& g, blue& b) {
  // Define colormap thresholds
  const float RED_THRESHOLD = 0.365079;    // Threshold for black to red
  const float YELLOW_THRESHOLD = 0.746032; // Threshold for red to yellow

  // Implement hot colormap
  if (value < RED_THRESHOLD) {
    // Black to red
    r = static_cast<red>((value / RED_THRESHOLD) * 255);
    g = 0;
    b = 0;
  } else if (value < YELLOW_THRESHOLD) {
    // Red to yellow
    r = 255;
    g = static_cast<green>(((value - RED_THRESHOLD) / (YELLOW_THRESHOLD - RED_THRESHOLD)) * 255);
    b = 0;
  } else {
    // Yellow to white
    r = 255;
    g = 255;
    b = static_cast<blue>(((value - YELLOW_THRESHOLD) / (1.0 - YELLOW_THRESHOLD)) * 255);
  }
}

// Write image to PPM file
void writeImageToFile(const std::string& filename, float* screenBuffer, int pixelWidth, int pixelHeight) {
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

void writeGLImageToFile(const std::string& filename, float* screenBuffer, int pixelWidth, int pixelHeight) {
  std::ofstream outputFile(filename);
  outputFile << "P3\n";
  outputFile << pixelWidth << " " << pixelHeight << "\n";
  outputFile << "255\n";

  for (int i = 0; i < pixelHeight; i++) {
    for (int j = 0; j < pixelWidth; j++) {
      // Map to 0-255 range
      unsigned int r = static_cast<unsigned int>(screenBuffer[(i * pixelWidth + j) * 3] * 255.0f);
      unsigned int g = static_cast<unsigned int>(screenBuffer[(i * pixelWidth + j) * 3 + 1] * 255.0f);
      unsigned int b = static_cast<unsigned int>(screenBuffer[(i * pixelWidth + j) * 3 + 2] * 255.0f);

      outputFile << r << " " << g << " " << b << " ";
    }
    outputFile << "\n";
  }
  outputFile.close();
}

// ===== OUTPUT FUNCTIONS =====

inline void printDisplayResolution(int pixelWidth, int pixelHeight) {
  std::cout << "Display Resolution: " << pixelWidth << "x" << pixelHeight << std::endl;
}

inline void printBufferSize(int numPixels) {
  std::cout << "Buffer Size: " << numPixels << " pixels" << "\n"
            << "Buffer Size: " << numPixels * sizeof(float) / (1024.0 * 1024.0) << " MB" << std::endl;
}
#endif // IMAGE_PROCESSING_H
