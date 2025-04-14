#include <algorithm>
#include <fstream>
#include <iostream>

// External Libraries
#include <omp.h>

// OpenGL stuff
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include "shader.h"

// Project Headers
#include "timer.h"
#include "metric.h"
#include "dormand_prince.h"
#include "ray_tracer.h"
#include "constants.h"
#include "config.h"

// ===== CONSTANTS =====
// OpenGL constants
const int window_width = 1600;
const int window_height = 900;
bool useHotColormap = false; // use spacebar to toggle from grayscale to hot

// OpenMP config
enum ThreadCount {
  kDefault,
  kSingle,
  kManual = 10
};
static inline int numThreads = ThreadCount::kDefault;

// ===== FORWARD DECLARATIONS =====
// OpenMP functions
inline void setOpenMPThreads(int numThreads);

// OpenGL functions
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);

// Image processing functions
void normalizeScreenBuffer(float* screenBuffer, int numPixels);
void applyColormap(float value, int& r, int& g, int& b);
void writeImageToFile(const std::string& filename, float* screenBuffer, int pixelWidth, int pixelHeight);
void applyHotColorMap(float value, unsigned int& r, unsigned int& g, unsigned int& b);

// Output functions
inline void printDisplayResolution(int pixelWidth, int pixelHeight);
inline void printBufferSize(int numPixels);
inline void printNumberOfThreads();

// Ray tracing functions
void performRayTracing(RayTracer& rayTracer, float* screenBuffer, 
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
  imgConfig.scale = 12;
  // imgConfig.cameraScale = 1.5;
 
  // Set up output configuration
  Config::OutputConfig outConfig;
  outConfig.setDescriptiveFilename(bhConfig, imgConfig, "bh_cpu_gl"); 
  
  // Create ray tracer object with config objects
  RayTracer rayTracer(bhConfig, imgConfig);
  
  setOpenMPThreads(numThreads);

  // ===== MEMORY ALLOCATION =====
  timer.start("INITIALIZING SCREEN BUFFER");
  const int pixelWidth = imgConfig.width();
  const int pixelHeight = imgConfig.height();
  const int numPixels = imgConfig.numPixels();
  
  // Allocate screen buffer
  float* screenBuffer = new float[numPixels]{}; // 1D row-major order
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

  // Convert float buffer to float buffer for OpenGL compatibility
  timer.start("APPLYING COLOR MAP + MOVING TO FLOAT BUFFER");
  float* floatBuffer {new float[numPixels * 3]{}}; // RGB values (3 channels)
  unsigned int r, g, b;
  for (int i = 0; i < numPixels; i++) {
    r = g = b = static_cast<unsigned int>(screenBuffer[i] * 255.0f); // Grayscale for now
    
    // Store normalized RGB values (0.0-1.0) in the float buffer
    floatBuffer[i * 3] = r / 255.0f;        // R
    floatBuffer[i * 3 + 1] = g / 255.0f;    // G
    floatBuffer[i * 3 + 2] = b / 255.0f;    // B
  }

  float* hotFloatBuffer {new float[numPixels * 3]{}}; // RGB values (3 channels)
  for (int i = 0; i < numPixels; i++) {
    applyHotColorMap(screenBuffer[i], r, g, b);
    hotFloatBuffer[i * 3] = r / 255.0f;        // R
    hotFloatBuffer[i * 3 + 1] = g / 255.0f;    // G
    hotFloatBuffer[i * 3 + 2] = b / 255.0f;    // B
  }
  timer.stop();

  // ===== OPENGL SETUP =====
  timer.start("INITIALIZING OPENGL");
  // init glfw and config window
  if (!glfwInit()) {
    std::cerr << "Failed to initialize GLFW" << std::endl;
    return -1;
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  #ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
  #endif

  // glfw window creation
  // GLFWwindow* window = glfwCreateWindow(window_width, window_height, "Black Hole Raytracer", NULL, NULL);
  GLFWwindow* window = glfwCreateWindow(pixelWidth, pixelHeight, "Black Hole Raytracer", NULL, NULL);
  if (window == NULL) {
    std::cerr << "Failed to create GLFW window" << std::endl;
    glfwTerminate();
    return -1;
  }
  glfwMakeContextCurrent(window);
  glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

  // glad load opengl function pointers
  if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
    std::cerr << "Failed to initialize GLAD" << std::endl;
    return -1;
  }

  // compile shaders program
  Shader grayscaleShader("simple.vert", "grayscale.frag");
  Shader hotShader("simple.vert", "hot_colormap.frag");
  // TODO: hot colormap shader later...

  // Create vertex data for full-screen quad
  float vertices[] = {
    // positions  // texture coords
    -1.0f, -1.0f, 0.0f, 1.0f, // bottom left
    1.0f,  -1.0f, 1.0f, 1.0f, // bottom right
    1.0f,  1.0f,  1.0f, 0.0f, // top right
    -1.0f, 1.0f,  0.0f, 0.0f  // top left
  };
  unsigned int indices[] = {
    0, 1, 2,
    0, 2, 3
  };
  unsigned int VBO, VAO, EBO;
  glGenVertexArrays(1, &VAO);
  glGenBuffers(1, &VBO);
  glGenBuffers(1, &EBO);

  glBindVertexArray(VAO);

  glBindBuffer(GL_ARRAY_BUFFER, VBO);
  glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

  glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
  glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

  // position attribute
  glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
  glEnableVertexAttribArray(0);
  // texture coord attribute
  glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
  glEnableVertexAttribArray(1);

  // load & create texture
  unsigned int texture;
  glGenTextures(1, &texture);
  glBindTexture(GL_TEXTURE_2D, texture);
  // set texture wrapping param to prevent edge artifacts
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  // set texture filtering param
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR); // maybe GL_NEAREST
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

  // load data to texture - using RGB format for three-component color data
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pixelWidth, pixelHeight, 0, GL_RGB, GL_FLOAT, floatBuffer);
  glGenerateMipmap(GL_TEXTURE_2D);

  // Create another texture for hot colormap
  unsigned int hotTexture;
  glGenTextures(1, &hotTexture);
  glBindTexture(GL_TEXTURE_2D, hotTexture);
  // texture parameters
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
  // load hot colormap data to texture
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pixelWidth, pixelHeight, 0, GL_RGB, GL_FLOAT, hotFloatBuffer);
  glGenerateMipmap(GL_TEXTURE_2D);
  timer.stop();

  // main render loop
  while (!glfwWindowShouldClose(window)) {
    // input
    processInput(window);

    // render
    // glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    // Use the appropriate shader based on toggle state
    if (useHotColormap) {
      hotShader.use();
      glBindTexture(GL_TEXTURE_2D, hotTexture);
    } else {
      grayscaleShader.use();
      glBindTexture(GL_TEXTURE_2D, texture);
    }
    
    // bind vertex array and draw
    glBindVertexArray(VAO);
    glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

    // swap buffers and poll IO events
    glfwSwapBuffers(window);
    glfwPollEvents();
  }

  // cleanup
  glDeleteVertexArrays(1, &VAO);
  glDeleteBuffers(1, &VBO);
  glDeleteBuffers(1, &EBO);
  glDeleteTextures(1, &texture);
  glDeleteTextures(1, &hotTexture);

  glfwTerminate();
  // ===== CLOSED GLFW =====

  // ===== WRITE IMAGE TO FILE =====
  // Write the image to file using the output config
  timer.start("WRITING IMAGE TO FILE");
  writeImageToFile(outConfig.getFullPath(), screenBuffer, pixelWidth, pixelHeight);
  timer.stop();

  // Free screen buffer memory
  delete[] screenBuffer;
  delete[] floatBuffer;  // Don't forget to free the float buffer

  return 0;
}

// ===== RAY TRACING FUNCTIONS =====
void performRayTracing(RayTracer& rayTracer, float* screenBuffer, 
                      const Config::BlackHole& bhConfig,
                      int pixelWidth, int pixelHeight, Timer& timer) {
  #pragma omp parallel
  {
    // Each thread needs its own metric, integrator, and ray
    BoyerLindquistMetric thread_metric(bhConfig.spin, bhConfig.mass);
    DormandPrinceRK45 integrator(6, Constants::Integration::ABS_TOLERANCE,
                            Constants::Integration::REL_TOLERANCE,
                            Constants::Integration::MIN_STEP_SIZE,
                            Constants::Integration::MAX_STEP_SIZE,
                            Constants::Integration::INITIAL_STEP_SIZE,
                            Constants::Integration::MAX_ITERATIONS);
    float* y {new float[6]{}};
    
    #pragma omp for collapse(2) schedule(dynamic) nowait
    for (int i = 0; i < pixelWidth; i++) {
      for (int j = 0; j < pixelHeight; j++) {
        float intensity = 0.0;
        
        // Trace the ray and get intensity with the thread-local integrator
        if (rayTracer.traceRay(i, j, thread_metric, integrator, y, intensity)) {
          screenBuffer[j * pixelWidth + i] = intensity;
        }
      }
    }
    delete[] y;
  }
}

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

/**
 * @brief matplotlib-style hot colormap
 * 
 * A sequential colormap that transitions from black -> red -> yellow -> white
 * 
 * @param value Normalized intensity value between 0 and 1
 * @param r Red component (0-255)
 * @param g Green component (0-255)
 * @param b Blue component (0-255)
 * 
 * @see [matplotlib colormap](https://github.com/matplotlib/matplotlib/blob/main/lib/matplotlib/_cm.py)
 * 
 * @note THE FOLLOWING VALUES ARE BASED ON THE HOT COLORMAP FROM MATPLOTLIB
 * 
 * red:   (0.0      , 0.0416 , 0.0416),
 *        (0.365079 , 1.0    , 1.0   ),
 *        (1.0      , 1.0    , 1.0   )
 * green: (0.0      , 0.0    , 0.0   ),
 *        (0.365079 , 0.0    , 0.0   ),
 *        (0.746032 , 1.0     , 1.0  ),
 *        (1.0      , 1.0     , 1.0  )
 * blue:  (0.0      , 0.0     , 0.0  ),
 *        (0.746032 , 0.0     , 0.0  ),
 *        (1.0      , 1.0     , 1.0  ) 
 */
void applyHotColorMap(float value, unsigned int& r, unsigned int& g, unsigned int& b) {
  // Define colormap thresholds
  const float RED_THRESHOLD = 0.365079;    // Threshold for black to red
  const float YELLOW_THRESHOLD = 0.746032; // Threshold for red to yellow

  // Implement hot colormap
  if (value < RED_THRESHOLD) {
    // Black to red
    r = static_cast<unsigned int>((value / RED_THRESHOLD) * 255);
    g = 0;
    b = 0;
  } else if (value < YELLOW_THRESHOLD) {
    // Red to yellow
    r = 255;
    g = static_cast<unsigned int>(((value - RED_THRESHOLD) / (YELLOW_THRESHOLD - RED_THRESHOLD)) * 255);
    b = 0;
  } else {
    // Yellow to white
    r = 255;
    g = 255;
    b = static_cast<unsigned int>(((value - YELLOW_THRESHOLD) / (1.0 - YELLOW_THRESHOLD)) * 255);
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
      unsigned int r, g, b;

      applyHotColorMap(value, r, g, b);  // Uncomment this line

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
            << "Buffer Size: " << numPixels * sizeof(float) / (1024.0 * 1024.0) << " MB" << std::endl;
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

// ===== OPENGL FUNCTIONS =====
void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
  glViewport(0, 0, width, height);
}

void processInput(GLFWwindow* window) {
  static bool spacePressed = false;
  
  if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) {
    glfwSetWindowShouldClose(window, true);
  }
  
  // Toggle shader when spacebar is pressed (with debouncing)
  if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
    if (!spacePressed) {
      useHotColormap = !useHotColormap;
      spacePressed = true;
    }
  } else {
    spacePressed = false;
  }
}
