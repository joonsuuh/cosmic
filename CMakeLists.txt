cmake_minimum_required(VERSION 3.31)
project(cosmic)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Special handling for OpenMP on macOS
# Configure OpenMP
if(APPLE)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
    
    if(CMAKE_C_COMPILER_ID MATCHES "Clang")
        set(OpenMP_C "${CMAKE_C_COMPILER}")
        set(OpenMP_C_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
    
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
        set(OpenMP_CXX_FLAGS "-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY /opt/homebrew/opt/libomp/lib/libomp.dylib)
    endif()
endif()


# Find required packages
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# Add glad lib
add_library(glad STATIC external/src/glad.c)
target_include_directories(glad PUBLIC external)

# Set Homebrew paths for Mac
if(APPLE)
    set(HOMEBREW_PATH /opt/homebrew)
    set(GLFW_DIR ${HOMEBREW_PATH}/lib/cmake/glfw3)
    
    include_directories(SYSTEM 
        ${HOMEBREW_PATH}/include
    )
    link_directories(${HOMEBREW_PATH}/lib)
endif()

# Set output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/bin)

# Add executable
add_executable(cosmic src/bh_gl_omp.cpp)

# Add include directories
target_include_directories(cosmic PUBLIC include/cosmic)
target_include_directories(cosmic PUBLIC external)

# Link all libraries
target_link_libraries(cosmic PRIVATE
    glad
    glfw
    OpenGL::GL
    OpenMP::OpenMP_CXX
)