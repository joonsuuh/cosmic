cmake_minimum_required(VERSION 3.10)

#----------------------------------
# Compiler and Standard B4 Project
#----------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
option(USE_CLANG "Build with Clang compiler" OFF)

# Compiler selection must be before project()
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

# Debug output
message(STATUS "C compiler: ${CMAKE_C_COMPILER}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")

project(cosmic)

#----------------------------------
# Platform-specific Configuration
#----------------------------------
set(HOMEBREW_PATH /opt/homebrew)

message(STATUS "CXX FLAGS: ${CMAKE_CXX_FLAGS}")
if(APPLE)
    # Apple clang shenanigans
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

#----------------------------------
# Dependencies
#----------------------------------
find_package(OpenMP REQUIRED)
find_package(OpenGL REQUIRED)
find_package(glfw3 REQUIRED)

# GLAD library
add_library(glad STATIC external/src/glad.c)
target_include_directories(glad PUBLIC external)

#----------------------------------
# Project Structure
#----------------------------------
# Create required directories
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/bin)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)
endif()
if(NOT EXISTS ${CMAKE_SOURCE_DIR}/data)
    file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/data)
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#----------------------------------
# Targets
#----------------------------------
# Main executable
add_executable(cosmic src/bh_gl_omp.cpp)
target_include_directories(cosmic PUBLIC 
    include/cosmic
    external
)
target_link_libraries(cosmic PRIVATE
    glad
    glfw
    OpenGL::GL
    OpenMP::OpenMP_CXX
)

# CPU-only executable
add_executable(cosmic_cpu src/cosmic_cpu.cpp)
target_include_directories(cosmic_cpu PUBLIC include/cosmic)
target_link_libraries(cosmic_cpu PRIVATE
    OpenMP::OpenMP_CXX
)

# CPU-only executable with OpenGL
add_executable(cosmic_cpu_gl src/cosmic_cpu_gl.cpp)
target_include_directories(cosmic_cpu_gl PUBLIC
    include/cosmic
    external
)
target_link_libraries(cosmic_cpu_gl PRIVATE
    glad
    glfw
    OpenGL::GL
    OpenMP::OpenMP_CXX
)

#----------------------------------
# Definitions
#----------------------------------
target_compile_definitions(cosmic PRIVATE SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/")
target_compile_definitions(cosmic_cpu PRIVATE SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/")
target_compile_definitions(cosmic_cpu_gl PRIVATE SHADER_DIR="${CMAKE_SOURCE_DIR}/shaders/")

target_compile_definitions(cosmic PRIVATE PPM_DIR="${CMAKE_SOURCE_DIR}/data/")
target_compile_definitions(cosmic_cpu PRIVATE PPM_DIR="${CMAKE_SOURCE_DIR}/data/")
target_compile_definitions(cosmic_cpu_gl PRIVATE PPM_DIR="${CMAKE_SOURCE_DIR}/data/")

##########################
###### STUFF I USED ######
##########################

# glfw3 lib 
# wsl2 ubuntu: `sudo apt install libglfw3-dev`
# macOS hombrew: `brew install glfw`

# if(APPLE)
#     set(HOMEBREW_PATH /opt/homebrew)
#     set(GLFW_DIR ${HOMEBREW_PATH}/lib/cmake/glfw3)
    
#     include_directories(SYSTEM 
#         ${HOMEBREW_PATH}/include
#     )
#     link_directories(${HOMEBREW_PATH}/lib)
# endif()
# add_definitions(-DBUILD_DIR)
