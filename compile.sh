#!/usr/bin/env zsh 

# 
# LIST OF CPP FILES TO COMPILE
# make directory list 
FILE_PATHS=(
    "src/bh_gl_omp.cpp"
    "src/cosmic_cpu.cpp"
    "src/cosmic_cpu_gl.cpp"
)

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
            echo "Usage: ./compile.sh [--compiler=<g++-14|clang++>]"
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

    # Compile each file in the FILE_PATHS array
    for source_file in "${FILE_PATHS[@]}"; do
        # Extract the base name without extension
        base_name=$(basename "$source_file" .cpp)
        output_name="bin/$base_name"
        
        echo "Compiling $source_file to $output_name"
        
        # Determine if this file needs OpenGL (GLAD)
        if [[ "$source_file" == *"gl"* ]]; then
            # File needs OpenGL
            if [[ $compiler == "clang++" ]]; then
                # Compile with clang++ in multiple steps
                clang -O3 -c $GLAD_C -o bin/glad.o -Iexternal
                
                clang++ -c $source_file -o "bin/${base_name}.o" \
                    -$STD_FLAG $INCLUDES
                
                clang++ "bin/${base_name}.o" bin/glad.o -o "$output_name" \
                    $LIBS -lomp -lglfw \
                    -framework OpenGL -framework Cocoa -framework IOKit \
                    -$OPT_FLAG -Xpreprocessor -fopenmp
            else
                # Compile with g++ in one step
                $compiler $source_file $GLAD_C \
                    -$STD_FLAG -o "$output_name" \
                    -Iinclude/cosmic -Iexternal \
                    -I$GL_INC -L$GL_LIB \
                    -I$OMP_INC -L$OMP_LIB \
                    -lglfw -lomp \
                    -framework OpenGL -framework Cocoa -framework IOKit \
                    -$OPT_FLAG -fopenmp
            fi
        else
            # CPU-only version
            $compiler -$OPT_FLAG $source_file \
                -o "$output_name" \
                -Iinclude/cosmic $([[ $compiler == "clang++" ]] && echo "-Xpreprocessor") -fopenmp
        fi
    done

elif [[ $(uname) == "Linux" ]]; then
    echo "Compiling for Linux:: NOT YET"
fi
