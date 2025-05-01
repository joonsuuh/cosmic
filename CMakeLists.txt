cmake_minimum_required(VERSION 3.30)

########################################
# Project Options
########################################

# For macOS: use -DWITH_CUDA=OFF
option(USE_CLANG "Build with Clang compiler" OFF)
option(WITH_CUDA "Build with CUDA support" ON)

########################################
# Compiler Selection
########################################

if(USE_CLANG)
    set(CMAKE_C_COMPILER clang)
    set(CMAKE_CXX_COMPILER clang++)
else()
    if(APPLE)
        set(CMAKE_C_COMPILER gcc-14)
        set(CMAKE_CXX_COMPILER g++-14)
    else()
        set(CMAKE_C_COMPILER gcc)
        set(CMAKE_CXX_COMPILER g++)
    endif()
endif()

########################################
# Project Definition
########################################

if(WITH_CUDA)
    project(cosmic LANGUAGES CXX C CUDA)
    message(STATUS "CUDA support enabled")
else()
    project(cosmic LANGUAGES CXX C)
endif()

########################################
# Language Standards
########################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(WITH_CUDA)
    set(CMAKE_CUDA_STANDARD ${CMAKE_CXX_STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endif()

########################################
# Platform-specific Configuration
########################################

set(HOMEBREW_PATH /opt/homebrew)

if(APPLE)
    if(USE_CLANG)
        set(CMAKE_CXX_FLAGS "-O3")
        set(OMP_PATH ${HOMEBREW_PATH}/opt/libomp/include)
        set(OMP_LIB_PATH ${HOMEBREW_PATH}/opt/libomp/lib/libomp.dylib)
        set(OMP_FLAGS "-Xpreprocessor -fopenmp")
        
        set(OpenMP_C_FLAGS "${OMP_FLAGS} -I${OMP_PATH}")
        set(OpenMP_C_LIB_NAMES "omp")
        set(OpenMP_CXX_FLAGS "${OMP_FLAGS} -I${OMP_PATH}")
        set(OpenMP_CXX_LIB_NAMES "omp")
        set(OpenMP_omp_LIBRARY ${OMP_LIB_PATH})
    else()
        set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
    endif()
elseif(UNIX)
    set(OpenGL_GL_PREFERENCE "GLVND")
    set(CMAKE_CXX_FLAGS "-O3 -fopenmp")
endif()
message(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS}")

########################################
# CUDA Configuration
########################################

if(WITH_CUDA)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --resource-usage -O3")
    set(CMAKE_CUDA_ARCHITECTURES "89")
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_INCLUDES FALSE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_LIBRARIES FALSE)
    set(CMAKE_CUDA_USE_RESPONSE_FILE_FOR_OBJECTS FALSE)
    
    message(STATUS "CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")
    message(STATUS "CUDA flags: ${CMAKE_CUDA_FLAGS}")
endif()

########################################
# Dependencies
########################################

find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# GLAD library
add_library(glad STATIC external/glad/glad.c)
target_include_directories(glad PUBLIC ${CMAKE_SOURCE_DIR}/external)

########################################
# Project Structure
########################################

# Set output, ppm data, and shader dir
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(DATA_DIR ${CMAKE_BINARY_DIR}/data)
file(MAKE_DIRECTORY ${DATA_DIR})
set(SHADER_DIR "${CMAKE_SOURCE_DIR}/shaders/")

########################################
# CPU Targets
########################################

# CPU-only
add_executable(cosmic_cpu src/cosmic_cpu.cpp)
target_include_directories(cosmic_cpu PRIVATE ${CMAKE_SOURCE_DIR}/include)
target_link_libraries(cosmic_cpu PRIVATE
    OpenMP::OpenMP_CXX
)
target_compile_definitions(cosmic_cpu PRIVATE 
    PPM_DIR="${DATA_DIR}"
)

# CPU-only with OpenGL
add_executable(cosmic_cpu_gl src/cosmic_cpu_gl.cpp)
target_include_directories(cosmic_cpu_gl PRIVATE
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/external
)
target_link_libraries(cosmic_cpu_gl PRIVATE
    glad
    glfw
    OpenGL::GL
    OpenMP::OpenMP_CXX
)
target_compile_definitions(cosmic_cpu_gl PRIVATE 
    SHADER_DIR="${SHADER_DIR}"
    PPM_DIR="${DATA_DIR}"
)

########################################
# CUDA Targets
########################################

if(WITH_CUDA)
    add_executable(cosmic_cuda src/cosmic_cuda.cu)
    target_include_directories(cosmic_cuda PRIVATE ${CMAKE_SOURCE_DIR}/include)
    # target_link_libraries(cosmic_cuda PRIVATE)
    target_compile_definitions(cosmic_cuda PRIVATE 
        PPM_DIR="${DATA_DIR}"
    )

    add_executable(cosmic_cuda_gl src/cosmic_cuda_gl.cu)
    target_include_directories(cosmic_cuda_gl PRIVATE
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/external
    )
    target_link_libraries(cosmic_cuda_gl PRIVATE
        glad
        glfw
        OpenGL::GL
        OpenMP::OpenMP_CXX
    )
    target_compile_definitions(cosmic_cuda_gl PRIVATE 
        SHADER_DIR="${SHADER_DIR}"
        PPM_DIR="${DATA_DIR}"
    )
endif()