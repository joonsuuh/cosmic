#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>

// OpenGL Headers 
#include <glad/glad.h>
#include <GLFW/glfw3.h>

// CUDA HEADERS
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

// Project CUDA headers
#include "ray_tracer_cuda.cuh"
#include "cuda_helper.cuh"

// Project headers
#include "config.h"
#include "image_processing.h"
#include "perlin.h"
#include "timer.h"
#include "shader.h"

// ===== Constants =====
bool useHotColormap = true;  // use spacebar to toggle from grayscale to hot

// ===== CUDA Resources =====
cudaGraphicsResource_t cudaGrayscalePBO = nullptr;
cudaGraphicsResource_t cudaHotPBO = nullptr;
float* d_noiseMap = nullptr;

// Add global variables for frame recording
bool isRecording = true;
int frameCount = 0;
const std::string frameDir = "frames/";

// ===== Function Prototypes =====
void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void processInput(GLFWwindow* window);
void renderCudaFrame(float* d_screenBuffer, float* d_tempBuffer,
                     float* d_colorBuffer, float* d_hotColorBuffer, 
                     int pixelWidth, int pixelHeight, float time);

// Kernel to apply colormap
__global__ void applyGrayscaleColormapKernel(float* intensityBuffer, float* colorBuffer, 
                                            int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float intensity = intensityBuffer[idx];
        
        // RGB is the same for grayscale
        colorBuffer[idx * 3] = intensity;      // R
        colorBuffer[idx * 3 + 1] = intensity;  // G
        colorBuffer[idx * 3 + 2] = intensity;  // B
    }
}

__global__ void applyHotColormapKernel(float* intensityBuffer, float* colorBuffer, 
                                     int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int idx = y * width + x;
        float intensity = intensityBuffer[idx];
        
        // Hot colormap implementation
        float r = 0.0f, g = 0.0f, b = 0.0f;
        const float RED_THRESHOLD = 0.365079f;    // Threshold for black to red
        const float YELLOW_THRESHOLD = 0.746032f;
        
        if (intensity < RED_THRESHOLD) {
            // Black to red
            r = (intensity / RED_THRESHOLD);
            g = 0.0f;
            b = 0.0f;
        } else if (intensity < YELLOW_THRESHOLD) {
            // Red to yellow
            r = 1.0f;
            g = ((intensity - RED_THRESHOLD) / (YELLOW_THRESHOLD - RED_THRESHOLD));
            b = 0.0f;
        } else {
            // Yellow to white
            r = 1.0f;
            g = 1.0f;
            b = ((intensity - YELLOW_THRESHOLD) / (1.0f - YELLOW_THRESHOLD));
        }
        
        colorBuffer[idx * 3] = r;
        colorBuffer[idx * 3 + 1] = g;
        colorBuffer[idx * 3 + 2] = b;
    }
}

int main(int argc, char* argv[]) {
    std::cout << "CUDA-OpenGL version running with PBO" << std::endl;
    Timer timer;

    // Print CUDA device info
    printDeviceInfo();

    // ===== CONFIG SETUP =====
    BlackHole bhConfig;
    bhConfig.setObserverAngle(85.0);  // Set angle in degrees

    Image imgConfig;
    imgConfig.setScale(120.0f);
    
    // Create camera parameters
    const int pixelWidth = imgConfig.width();
    const int pixelHeight = imgConfig.height();
    const int numPixels = imgConfig.numPixels();

    // Set up output configuration
    OutputConfig outConfig;
    outConfig.setDescriptiveFilename(bhConfig, imgConfig, "cuda_gl_pbo");

    // ===== OPENGL SETUP =====
    timer.start("INITIALIZING OPENGL");
    // init glfw and config window
    if (!glfwInit()) {
        std::cerr << "Failed to initialize GLFW" << std::endl;
        return -1;
    }
    
    // Print GLFW version
    int major, minor, revision;
    glfwGetVersion(&major, &minor, &revision);
    std::cout << "GLFW Version: " << major << "." << minor << "." << revision << std::endl;
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // glfw window creation
    GLFWwindow* window = glfwCreateWindow(1920, 1080,
                                         "CUDA-GL PBO Black Hole", NULL, NULL);
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
    
    // Print OpenGL information
    std::cout << "OpenGL Version: " << glGetString(GL_VERSION) << std::endl;
    std::cout << "OpenGL Vendor: " << glGetString(GL_VENDOR) << std::endl;
    std::cout << "OpenGL Renderer: " << glGetString(GL_RENDERER) << std::endl;

    // compile shaders program
    Shader grayscaleShader("simple.vert", "grayscale.frag");
    Shader hotShader("simple.vert", "hot_colormap.frag");

    // Create vertex data for full-screen quad
    float vertices[] = {
        // positions  // texture coords
        -1.0f, -1.0f, 0.0f, 1.0f,  // bottom left
        1.0f,  -1.0f, 1.0f, 1.0f,  // bottom right
        1.0f,  1.0f,  1.0f, 0.0f,  // top right
        -1.0f, 1.0f,  0.0f, 0.0f   // top left
    };
    unsigned int indices[] = {0, 1, 2, 0, 2, 3};
    unsigned int VBO, VAO, EBO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);
    glGenBuffers(1, &EBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, EBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices,
                GL_STATIC_DRAW);

    // position attribute
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // texture coord attribute
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float),
                        (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // Create textures and PBOs for both colormaps
    unsigned int grayscaleTexture, hotTexture;
    unsigned int grayscalePBO, hotPBO;
    
    // Create and initialize Pixel Buffer Objects (PBOs)
    glGenBuffers(1, &grayscalePBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, grayscalePBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, numPixels * 3 * sizeof(float), NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    glGenBuffers(1, &hotPBO);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hotPBO);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, numPixels * 3 * sizeof(float), NULL, GL_STREAM_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
    
    // Regular grayscale texture
    glGenTextures(1, &grayscaleTexture);
    glBindTexture(GL_TEXTURE_2D, grayscaleTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pixelWidth, pixelHeight, 0, GL_RGB,
               GL_FLOAT, NULL);  // Allocate memory but don't upload data
    
    // Hot colormap texture
    glGenTextures(1, &hotTexture);
    glBindTexture(GL_TEXTURE_2D, hotTexture);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, pixelWidth, pixelHeight, 0, GL_RGB,
               GL_FLOAT, NULL);  // Allocate memory but don't upload data
    timer.stop();

    // ===== CUDA-GL INTEROP SETUP =====
    timer.start("SETTING UP CUDA-GL INTEROP");
    
    // Register PBOs with CUDA
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaGrayscalePBO, grayscalePBO, 
                                          cudaGraphicsMapFlagsWriteDiscard));
    CUDA_CHECK(cudaGraphicsGLRegisterBuffer(&cudaHotPBO, hotPBO, 
                                          cudaGraphicsMapFlagsWriteDiscard));
    
    // Allocate CUDA memory for raytracing
    float* d_screenBuffer;
    float* d_tempBuffer;
    float* d_colorBuffer;
    float* d_hotColorBuffer;

    CUDA_CHECK(cudaMalloc(&d_screenBuffer, numPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_tempBuffer, numPixels * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_colorBuffer, numPixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_hotColorBuffer, numPixels * 3 * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_screenBuffer, 0, numPixels * sizeof(float)));
    
    // ===== PERLIN NOISE SETUP =====
    timer.start("GENERATING PERLIN NOISE");
    const int noiseSize = 1024;
    float* noiseMap = generatePerlinNoise(noiseSize, noiseSize, 4, 0.5f);
    
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

    // ===== OUTPUT CONFIG =====
    std::cout << "Display Resolution: " << pixelWidth << "x" << pixelHeight
              << std::endl;
    std::cout << "Buffer Size: " << numPixels << " pixels" << "\n"
              << "Buffer Size: "
              << numPixels * sizeof(float) / (1024.0f * 1024.0f) << " MB"
              << std::endl;
              
    // Variable to track animation time
    float currentTime = 0.0f;
    float lastTime = 0.0f;
    
    // main render loop
    while (!glfwWindowShouldClose(window)) {
        // Calculate delta time for smooth animation
        float timeValue = glfwGetTime();
        float deltaTime = timeValue - lastTime;
        lastTime = timeValue;
        
        // Increment animation parameter slowly
        currentTime += deltaTime * 0.1f;  // Control animation speed
        printf("Current Time: %f\n", currentTime);
        
        // Input handling
        processInput(window);

        // Render new frame with CUDA
        renderCudaFrame(d_screenBuffer, d_tempBuffer, d_colorBuffer,
                        d_hotColorBuffer, pixelWidth, pixelHeight, currentTime);

        // Render
        glClear(GL_COLOR_BUFFER_BIT);

        // Use appropriate shader and texture based on colormap selection
        if (useHotColormap) {
            hotShader.use();
            glBindTexture(GL_TEXTURE_2D, hotTexture);
            
            // Update texture from PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, hotPBO);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pixelWidth, pixelHeight, 
                          GL_RGB, GL_FLOAT, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        } else {
            grayscaleShader.use();
            glBindTexture(GL_TEXTURE_2D, grayscaleTexture);
            
            // Update texture from PBO
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, grayscalePBO);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, pixelWidth, pixelHeight, 
                          GL_RGB, GL_FLOAT, 0);
            glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        }

        // Draw quad with the raytraced texture
        glBindVertexArray(VAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
        
        // Save frame if recording is active
        if (isRecording) {
            // Create directory if it doesn't exist
            std::string command = "mkdir -p " + frameDir;
            system(command.c_str());
            
            // Generate filename with leading zeros for proper sorting
            char filename[100];
            sprintf(filename, "%sframe_%06d.ppm", frameDir.c_str(), frameCount++);
            
            // Get current frame data from device - use the appropriate buffer based on current colormap
            float* h_frameBuffer = new float[numPixels * 3];
            if (useHotColormap) {
                CUDA_CHECK(cudaMemcpy(h_frameBuffer, d_hotColorBuffer, 3 * numPixels * sizeof(float),
                           cudaMemcpyDeviceToHost));
            } else {
                CUDA_CHECK(cudaMemcpy(h_frameBuffer, d_colorBuffer, 3 * numPixels * sizeof(float),
                           cudaMemcpyDeviceToHost));
            }
            
            // Write the frame to file using existing function
            writeGLImageToFile(filename, h_frameBuffer, pixelWidth, pixelHeight);
            
            // Free temporary buffer
            delete[] h_frameBuffer;
            
            // Update window title to show recording status
            char title[128];
            sprintf(title, "CUDA-GL Black Hole - Recording Frame: %d", frameCount);
            glfwSetWindowTitle(window, title);
        }

        // Swap buffers and poll events
        glfwSwapBuffers(window);
        glfwPollEvents();
    }

    // ===== CLEANUP =====
    // Export final image to file
    timer.start("WRITING IMAGE TO FILE");
    
    // Get the final rendered image back from device
    float* h_screenBuffer = new float[numPixels];
    CUDA_CHECK(cudaMemcpy(h_screenBuffer, d_screenBuffer, numPixels * sizeof(float), 
                       cudaMemcpyDeviceToHost));
    
    
    // Write image to file
    writeGLImageToFile(outConfig.getFullPath(), h_screenBuffer, pixelWidth, pixelHeight);
    timer.stop();

    // Clean up resources
    delete[] h_screenBuffer;
    delete[] noiseMap;
    
    // Free CUDA resources
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaGrayscalePBO));
    CUDA_CHECK(cudaGraphicsUnregisterResource(cudaHotPBO));
    CUDA_CHECK(cudaFree(d_screenBuffer));
    CUDA_CHECK(cudaFree(d_tempBuffer));
    CUDA_CHECK(cudaFree(d_colorBuffer));
    CUDA_CHECK(cudaFree(d_hotColorBuffer));
    CUDA_CHECK(cudaFree(d_noiseMap));
    
    // Free OpenGL resources
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);
    glDeleteBuffers(1, &EBO);
    glDeleteBuffers(1, &grayscalePBO);
    glDeleteBuffers(1, &hotPBO);
    glDeleteTextures(1, &grayscaleTexture);
    glDeleteTextures(1, &hotTexture);
    
    glfwTerminate();
    
    return 0;
}

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

void renderCudaFrame(float* d_screenBuffer, float* d_tempBuffer, 
                    float* d_colorBuffer, float* d_hotColorBuffer, 
                    int pixelWidth, int pixelHeight, float time) {
    // CUDA kernel configuration
    dim3 blockSize(32, 4);
    dim3 gridSize((pixelWidth + blockSize.x - 1) / blockSize.x,
                  (pixelHeight + blockSize.y - 1) / blockSize.y);
    
    // Ray trace kernel to generate intensities - use time parameter for animation
    rayTraceKernel<<<gridSize, blockSize>>>(d_screenBuffer, d_noiseMap, 1024, time);
    CUDA_CHECK(cudaGetLastError());
    
    // Make sure ray tracing is complete before proceeding
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // === STEP 1: Find max value for normalization ===
    // Initialize temp buffer to 0
    float initVal = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_tempBuffer, &initVal, sizeof(float), cudaMemcpyHostToDevice));
    
    // Use parallel reduction to find max value efficiently
    const int blockSize1D = 256;
    const int gridSize1D = (pixelWidth * pixelHeight + blockSize1D - 1) / blockSize1D;
    
    // Call reduction kernel with proper shared memory size
    maxReduceAtomic<<<gridSize1D, blockSize1D, blockSize1D * sizeof(float)>>>(
        d_screenBuffer, d_tempBuffer, pixelWidth * pixelHeight, 0.0f);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Get the max value from device memory
    float maxVal;
    CUDA_CHECK(cudaMemcpy(&maxVal, d_tempBuffer, sizeof(float), cudaMemcpyDeviceToHost));
    
    // === STEP 2: Normalize the intensity buffer ===
    dim3 normalizeBlockSize(256);
    dim3 normalizeGridSize((pixelWidth * pixelHeight + normalizeBlockSize.x - 1) / normalizeBlockSize.x);
    
    normalizeBufferKernel<<<normalizeGridSize, normalizeBlockSize>>>(
        d_screenBuffer, pixelWidth * pixelHeight, maxVal);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // === STEP 3: Apply colormaps to the normalized buffer ===
    applyGrayscaleColormapKernel<<<gridSize, blockSize>>>(
        d_screenBuffer, d_colorBuffer, pixelWidth, pixelHeight);
    CUDA_CHECK(cudaGetLastError());
    
    applyHotColormapKernel<<<gridSize, blockSize>>>(
        d_screenBuffer, d_hotColorBuffer, pixelWidth, pixelHeight);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // === STEP 4: Copy colormap data to PBOs ===
    float *d_grayscale_pbo, *d_hot_pbo;
    size_t grayscale_size, hot_size;
    
    // Map grayscale PBO
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaGrayscalePBO));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_grayscale_pbo, &grayscale_size, 
                                                  cudaGrayscalePBO));
    
    // Copy grayscale data to PBO
    CUDA_CHECK(cudaMemcpy(d_grayscale_pbo, d_colorBuffer, pixelWidth * pixelHeight * 3 * sizeof(float), 
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaGrayscalePBO));
    
    // Map hot colormap PBO
    CUDA_CHECK(cudaGraphicsMapResources(1, &cudaHotPBO));
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer((void**)&d_hot_pbo, &hot_size, 
                                                  cudaHotPBO));
    
    // Copy hot colormap data to PBO
    CUDA_CHECK(cudaMemcpy(d_hot_pbo, d_hotColorBuffer, pixelWidth * pixelHeight * 3 * sizeof(float), 
                       cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &cudaHotPBO));
}
