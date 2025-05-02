#!/bin/bash

# Use system clang with proper SDK path
export CC="clang"
export CXX="clang++"

# Check if libomp is installed
if [ ! -d "/opt/homebrew/opt/libomp" ]; then
  echo "Error: libomp not found. Installing libomp..."
  brew install libomp
fi

# Add proper flags for OpenMP support on macOS
export CFLAGS="-isysroot $(xcrun --show-sdk-path) -I/opt/homebrew/opt/libomp/include"
export CXXFLAGS="-isysroot $(xcrun --show-sdk-path) -I/opt/homebrew/opt/libomp/include"
export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp"

rm -rf build
# conan install . --build=missing
# Add specific CMake flags to resolve SDK linking issues
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH=/opt/homebrew/lib/cmake \
      -DCMAKE_OSX_SYSROOT="$(xcrun --show-sdk-path)" \
      -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0 \
      -DCMAKE_C_COMPILER="$(which clang)" \
      -DCMAKE_CXX_COMPILER="$(which clang++)" \
      -DOpenMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include" \
      -DOpenMP_C_LIB_NAMES="omp" \
      -DOpenMP_CXX_LIB_NAMES="omp" \
      -DOpenMP_omp_LIBRARY="/opt/homebrew/opt/libomp/lib/libomp.dylib"

cmake --build build

# Run the OpenVDB-based level set method for etching simulation
# ./build/vdbLevelset initial_struct_600_600.obj etched_result.obj 1.0 100

# Check if the executable was built successfully before running it
if [ -f "./build/levelset" ]; then
  echo "Running levelset executable..."
  ./build/levelset
else
  echo "Error: levelset executable was not built successfully."
  exit 1
fi

# Other executables (commented out)
# ./build/main
# ./build/alphawrap data/initial_struct.obj 600 600


