#!/bin/bash

rm -rf build
cmake -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/opt/homebrew/lib/cmake
cmake --build build

# ./build/main
# ./build/alphawrap data/initial_struct.obj 600 600
./build/levelset


