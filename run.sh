#!/bin/bash

rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew/lib/cmake
cmake --build build

# ./build/main
# mkdir run
# mkdir run/constant
# mkdir run/system
# openfoam

# surfaceMeshConvert data/initial_struct.obj run/constant/initial_struct.stl
# surfaceCheck run/constant/initial_struct.stl

# # 1. First run blockMesh to create background mesh
# blockMesh

# # 2. Run surfaceFeatureExtract to extract features
# surfaceFeatureExtract

# # 3. Run snappyHexMesh with overwrite flag
# snappyHexMesh -overwrite

# # 4. Check mesh quality
# checkMesh
