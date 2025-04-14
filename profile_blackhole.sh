#!/bin/bash

# Create directory for profiling results if it doesn't exist
mkdir -p ./profile_results

# Define the executable path - adjust if your binary is located elsewhere
EXECUTABLE="../bin/cosmic_cuda"

# Check if executable exists
if [ ! -f "$EXECUTABLE" ]; then
    echo "Executable not found at $EXECUTABLE"
    echo "Please specify the correct path to your CUDA executable"
    exit 1
fi

echo "Profiling $EXECUTABLE..."

# Run Nsight Systems profiling with settings adjusted for limited permissions
# Skip GPU metrics which require root privileges
nsys profile \
  --stats=true \
  --force-overwrite=true \
  --cuda-memory-usage=true \
  --trace=cuda,nvtx \
  --sample=cpu \
  --cpuctxsw=none \
  --duration=30 \
  --output=./profile_results/blackhole_profile \
  $EXECUTABLE

# Check if the profiling was successful
if [ $? -eq 0 ] && [ -f "./profile_results/blackhole_profile.nsys-rep" ]; then
    echo "Profiling successful!"
    echo "Report saved to ./profile_results/blackhole_profile.nsys-rep"
    
    # Generate summary stats if the profiling was successful
    nsys stats ./profile_results/blackhole_profile.nsys-rep
else
    echo "Profiling failed or report file not created."
    echo "Try running 'sudo nvperm' to set up permissions for GPU monitoring."
    echo "See: https://developer.nvidia.com/ERR_NVGPUCTRPERM"
fi
