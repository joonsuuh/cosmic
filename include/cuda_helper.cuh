#ifndef CUDA_HELPER_CUH
#define CUDA_HELPER_CUH

#include <cassert>
#include <iostream>


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

#endif // CUDA_HELPER_CUH