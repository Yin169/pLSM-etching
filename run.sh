#!/bin/bash

rm -rf build
# conan install . --build=missing
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew/lib/cmake 
cmake --build build

# Run the OpenVDB-based level set method for etching simulation
# ./build/vdbLevelset initial_struct_600_600.obj etched_result.obj 1.0 100


# Uncomment to run other executables
# ./build/main
# ./build/alphawrap data/initial_struct.obj 600 600
./build/levelset


