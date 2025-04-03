#!/bin/bash

# Add CUDA binaries to PATH
export PATH=/usr/local/cuda-12.8/bin${PATH:+:${PATH}}

# Add CUDA libraries to LD_LIBRARY_PATH (needed for runtime)
export LD_LIBRARY_PATH=/usr/local/cuda-12.8/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# PRINT WHERE ENV IS SET
echo "CUDA environment variables have been set."
echo "PATH now includes: /usr/local/cuda-12.8/bin"
echo "LD_LIBRARY_PATH now includes: /usr/local/cuda-12.8/lib64"

# Print CUDA version to verify installation
if command -v nvcc &> /dev/null; then
    echo "CUDA version:"
    nvcc --version
else
    echo "WARNING: nvcc not found in PATH. CUDA may not be properly installed."
fi
