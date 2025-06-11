#!/bin/bash

# Clean previous build
rm -rf build

# Create build directory
mkdir -p build

# Install dependencies with Conan
conan install . --build=missing 

# Configure and build with CMake
cd build
cmake .. -DCMAKE_TOOLCHAIN_FILE=Release/generators/conan_toolchain.cmake -DCMAKE_BUILD_TYPE=Release

# Build using all available cores
cmake --build . 

echo "Build completed successfully!"
cd .. 
nohup ./build/levelset "data/initial_struct.bnd" "./out/" > check.txt &