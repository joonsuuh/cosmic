#ifndef OMP_HELPER_H
#define OMP_HELPER_H

#include <iostream>
#include <omp.h>

enum ThreadCount { kDefault, kSingle, kManual = 10 };

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

#endif // OMP_HELPER_H