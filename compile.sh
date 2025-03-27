#!/usr/bin/env zsh 

#
# Configuration
#
compiler="g++-14"

#
# Parse command line arguments
#
while [[ $# -gt 0 ]]; do
    case $1 in
        --compiler=*)
            # Extract compiler name after the = symbol
            compiler="${1#*=}"
            ;;
        --help|-h)
            # Show usage information
            echo "Usage: ./manual.sh [--compiler=<g++-14|clang++>]"
            exit 0
            ;;
    esac
    shift
done

#
# Setup directories
#
mkdir -p bin data


#
# Compilation
#
if [[ $(uname) == "Darwin" ]]; then
    echo "Compiling for MacOS using $compiler"

    GL_INC="/opt/homebrew/include"
    OMP_INC="/opt/homebrew/opt/libomp/include"
    GL_LIB="/opt/homebrew/lib"
    OMP_LIB="/opt/homebrew/opt/libomp/lib"
    INCLUDES="-I$GL_INC -I$OMP_INC"
    LIBS="-L$GL_LIB -L$OMP_LIB"
    GLAD_C="external/src/glad.c"
    STD_FLAG="std=c++17"
    OPT_FLAG="O3"

    # OpenGL version
    if [[ $compiler == "clang++" ]]; then
        # Three-step compilation with clang/clang++:
        # 1. Compile glad.c to object file
        clang -O3 -c $GLAD_C -o bin/glad.o -Iexternal
        
        # 2. Compile main source to object file
        clang++ -c src/bh_gl_omp.cpp -o bin/bh_gl_omp.o \
            -$STD_FLAG $INCLUDES

        # 3. Link object files
        clang++ bin/bh_gl_omp.o bin/glad.o -o bin/cosmic \
            $LIBS -lomp -lglfw \
            -framework OpenGL -framework Cocoa -framework IOKit \
            -$OPT_FLAG -Xpreprocessor -fopenmp
    else
        # Single-step compilation with g++-14
        $compiler src/bh_gl_omp.cpp $GLAD_C \
            -$STD_FLAG -o bin/cosmic \
            -Iinclude/cosmic -Iexternal \
            -I$GL_INC -L$GL_LIB \
            -I$OMP_INC -L$OMP_LIB \
            -lglfw -lomp \
            -framework OpenGL -framework Cocoa -framework IOKit \
            -$OPT_FLAG -fopenmp
    fi

    # CPU version (simplified)
    $compiler  -O3 src/bh_cpu.cpp \
        -o bin/cosmic_cpu \
        -Iinclude/cosmic $([[ $compiler == "clang++" ]] && echo "-Xpreprocessor") -fopenmp

elif [[ $(uname) == "Linux" ]]; then
    echo "Compiling for Linux:: NOT YET"
fi
