#!/bin/bash

# Detect operating system
OS="$(uname -s)"
echo "Detected OS: $OS"

# Set compiler variables
export CC="clang"
export CXX="clang++"

# OS-specific configurations
if [ "$OS" = "Darwin" ]; then
  # macOS specific configuration
  echo "Configuring for macOS..."
  
  # Check if libomp is installed
  if [ ! -d "/opt/homebrew/opt/libomp" ]; then
    echo "Error: libomp not found. Installing libomp..."
    brew install libomp
  fi
  
  # Add proper flags for OpenMP support on macOS
  export CFLAGS="-isysroot $(xcrun --show-sdk-path) -I/opt/homebrew/opt/libomp/include"
  export CXXFLAGS="-isysroot $(xcrun --show-sdk-path) -I/opt/homebrew/opt/libomp/include"
  export LDFLAGS="-L/opt/homebrew/opt/libomp/lib -lomp"
  
  # Set OpenMP library paths for macOS
  OPENMP_C_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
  OPENMP_CXX_FLAGS="-Xpreprocessor -fopenmp -I/opt/homebrew/opt/libomp/include"
  OPENMP_LIB_NAMES="omp"
  OPENMP_LIB_PATH="/opt/homebrew/opt/libomp/lib/libomp.dylib"
  CMAKE_PREFIX_PATH="/opt/homebrew/lib/cmake"
  CMAKE_OSX_FLAGS="-DCMAKE_OSX_SYSROOT=\"$(xcrun --show-sdk-path)\" -DCMAKE_OSX_DEPLOYMENT_TARGET=13.0"

elif [ "$OS" = "Linux" ]; then
  # Linux specific configuration
  echo "Configuring for Linux..."
  
  # Check if libomp is installed
  if ! dpkg -l | grep -q libomp-dev && command -v apt-get >/dev/null; then
    echo "Installing OpenMP development package..."
    sudo apt-get update && sudo apt-get install -y libomp-dev
  elif ! rpm -q libomp-devel >/dev/null 2>&1 && command -v yum >/dev/null; then
    echo "Installing OpenMP development package..."
    sudo yum install -y libomp-devel
  fi
  
  # Add proper flags for OpenMP support on Linux
  export CFLAGS="-fopenmp"
  export CXXFLAGS="-fopenmp"
  export LDFLAGS="-fopenmp"
  
  # Set OpenMP library paths for Linux
  OPENMP_C_FLAGS="-fopenmp"
  OPENMP_CXX_FLAGS="-fopenmp"
  OPENMP_LIB_NAMES="omp"
  OPENMP_LIB_PATH=""
  CMAKE_PREFIX_PATH="/usr/lib/cmake"
  CMAKE_OSX_FLAGS=""
else
  echo "Unsupported operating system: $OS"
  exit 1
fi

rm -rf build

# Uncomment if you need to use conan
# conan install . --build=missing

# Add specific CMake flags based on OS
cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$CMAKE_PREFIX_PATH" \
      $CMAKE_OSX_FLAGS \
      -DCMAKE_C_COMPILER="$(which $CC)" \
      -DCMAKE_CXX_COMPILER="$(which $CXX)" \
      -DOpenMP_C_FLAGS="$OPENMP_C_FLAGS" \
      -DOpenMP_CXX_FLAGS="$OPENMP_CXX_FLAGS" \
      -DOpenMP_C_LIB_NAMES="$OPENMP_LIB_NAMES" \
      -DOpenMP_CXX_LIB_NAMES="$OPENMP_LIB_NAMES" \
      ${OPENMP_LIB_PATH:+-DOpenMP_omp_LIBRARY="$OPENMP_LIB_PATH"}

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


